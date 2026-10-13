#!/usr/bin/env python3
"""Bounded public lifecycle/OS-pipe debugging, with a private PostgreSQL cluster.

The short Python children below are pipe-protocol fixtures, not model workers.
This never calls run_worker, replaces a backend, or changes an activation gate.
"""
from __future__ import annotations
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import uuid
from private_rag.compute_process import _collect_worker_output
from private_rag.database import Repository
from private_rag.delivery import PreparationShare
from private_rag.errors import ErrorCode, PublicError
from private_rag.lifecycle import Lifecycle
from private_rag.scheduler import SchedulerLifecycleEvents
from private_rag.storage import ObjectStore


def expect(code,call):
    try: call()
    except PublicError as exc: assert exc.code==code,(exc.code,code)
    else: raise AssertionError('fixed public rejection absent')


def pipe_fixture(mode,callback,check=None):
    """Actual child+pipe resources; a debug alarm bounds a broken collector."""
    out_r,out_w=os.pipe();evt_r,evt_w=os.pipe();ack_r,ack_w=os.pipe()
    owned={out_r,out_w,evt_r,evt_w,ack_r,ack_w}; process=None
    source='''import os,sys
o,e,a=map(int,sys.argv[1:4]); mode=sys.argv[4]
if mode=='early_eof':
 os.close(e);os.read(a,1);raise SystemExit(7)
if mode=='exit_error':raise SystemExit(9)
os.write(e,b'\\x01');assert os.read(a,1)==b'\\x01'
if mode=='cancel':os.read(a,1);raise SystemExit(8)
os.write(e,b'\\x02');assert os.read(a,1)==b'\\x02'
data=b'P'*8201
while data:data=data[os.write(o,data):]
os.close(o);os.close(e);os.close(a)
'''
    def alarm(*unused): raise TimeoutError('public pipe fixture exceeded its debug bound')
    previous=signal.signal(signal.SIGALRM,alarm); signal.alarm(10)
    try:
        process=subprocess.Popen([sys.executable,'-c',source,str(out_w),str(evt_w),str(ack_r),mode],
                                 pass_fds=(out_w,evt_w,ack_r),stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        for fd in (out_w,evt_w,ack_r):os.close(fd);owned.remove(fd)
        return _collect_worker_output(process,8201,out_r,evt_r,ack_w,callback,check)
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=3)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        if process is not None: assert process.poll() is not None
        for fd in owned:os.close(fd)


def main():
    project=Path(__file__).resolve().parents[1];pg=Path('/usr/lib/postgresql/18/bin')
    results={'scope':'isolated_public_cancellation_and_pipe_debug','formal_acceptance':False,
             'private_model_executed':False,'production_gate_bypassed':False,'fake_private_backend_used':False,
             'pipe_children':'public event/ACK/output fixtures only, never run_worker executables'}
    with tempfile.TemporaryDirectory(prefix='prag-cancel-') as temporary:
        root=Path(temporary);data=root/'pg'
        subprocess.run([str(pg/'initdb'),'-D',str(data),'-A','trust','--no-locale','-E','UTF8'],check=True,capture_output=True)
        subprocess.run([str(pg/'pg_ctl'),'-D',str(data),'-l',str(root/'pg.log'),'-o',f"-k {root} -h '' -p 55486",'-w','start'],check=True,capture_output=True)
        try:
            repo=Repository(f'host={root} port=55486 dbname=postgres',ObjectStore(root/'objects'));repo.initialize();life=Lifecycle(repo)
            owner,other=str(uuid.uuid4()),str(uuid.uuid4())
            repo.enroll_account(owner,'D','1'*64);repo.enroll_account(other,'U','2'*64)
            doc=repo.create_document(owner,'public-debug','cancellation only');version=repo.create_version(owner,doc,1,'3'*64)
            # Exercise upgrade on an existing database, preserving metadata.
            old_session=repo.start_preparation(owner,doc,version,0)
            with repo.transaction() as c:
                c.execute('ALTER TABLE sessions DROP COLUMN worker_claimed, DROP COLUMN worker_owner, DROP COLUMN cancel_requested')
                c.execute('DROP TABLE session_cancel_intents')
            repo.initialize()
            with repo.transaction() as c:
                assert c.execute('SELECT count(*) AS n FROM documents').fetchone()['n']==1
                assert c.execute('SELECT worker_claimed,worker_owner FROM sessions WHERE session_id=%s',(old_session,)).fetchone()=={'worker_claimed':True,'worker_owner':None}
            assert repo.request_session_cancel(owner,old_session)['state']=='accepted'
            # This isolated legacy fixture never started a worker. In deployed
            # upgrades an operator must establish the equivalent before ending.
            expect(ErrorCode.BAD_STATE,lambda:life.recover_after_process_death(old_session,operator_confirmed_worker_and_lanes_stopped=False))
            assert life.recover_after_process_death(old_session,operator_confirmed_worker_and_lanes_stopped=True)==1
            def start(identity=None):return repo.start_preparation(owner,doc,version,0,session_id=identity)
            def state(session):return repo.session_status(owner,session)['state']
            early=str(uuid.uuid4());assert repo.request_session_cancel(owner,early)['state']=='aborted'
            expect(ErrorCode.DISCONNECTED,lambda:start(early))
            # A different account's future intent cannot cancel this owner.
            future=str(uuid.uuid4());repo.request_session_cancel(other,future);assert start(future)==future
            expect(ErrorCode.NOT_AUTHORIZED,lambda:repo.request_session_cancel(other,future))
            pending=life.reserve(future,0,'session_randomness',[1]);repo.request_session_cancel(owner,future)
            assert state(future)=='aborted'
            with repo.transaction() as c:
                assert c.execute('SELECT state FROM materials WHERE material_id=%s',(pending,)).fetchone()['state']=='burned'
            expect(ErrorCode.MATERIALS_NOT_READY,lambda:repo.claim_accepted_compute_session(str(uuid.uuid4())))
            claimed=start();token=str(uuid.uuid4());assert repo.claim_accepted_compute_session(token)['session_id']==claimed
            assert repo.claimed_session(token)==claimed
            reply=repo.request_session_cancel(owner,claimed);assert reply['state']=='accepted' and reply['cancel_requested']
            expect(ErrorCode.BUSY,start)
            expect(ErrorCode.DISCONNECTED,lambda:repo.check_session_cancellation(claimed,token))
            expect(ErrorCode.DISCONNECTED,lambda:life.transition(claimed,'computing'))
            life.finish(claimed,acknowledged=False,public_fault='DISCONNECTED')
            # Ordered actual pipe events commit to PG before receiving ACK.
            normal=start();token=str(uuid.uuid4());repo.claim_accepted_compute_session(token)
            output=pipe_fixture('success',SchedulerLifecycleEvents(repo,normal),lambda:repo.check_session_cancellation(normal,token))
            assert output==b'P'*8201 and state(normal)=='releasing'
            repo.complete_native_preparation(normal,PreparationShare(normal,'S',(5,)*1024,7))
            assert repo.request_session_cancel(owner,normal)=={'session_id':normal,'state':'completed','cancel_requested':False}
            # Cancel a live public fixture after AUTHENTICATED; the slot remains
            # occupied until the fixture helper has reaped its child and fds.
            cancelled=start();token=str(uuid.uuid4());repo.claim_accepted_compute_session(token)
            events=SchedulerLifecycleEvents(repo,cancelled)
            def cancel_at_event(event):
                events(event)
                assert event=='AUTHENTICATED'
                repo.request_session_cancel(owner,cancelled)
                expect(ErrorCode.BUSY,start)
            expect(ErrorCode.DISCONNECTED,lambda:pipe_fixture('cancel',cancel_at_event,lambda:repo.check_session_cancellation(cancelled,token)))
            assert state(cancelled)=='computing'
            life.finish(cancelled,acknowledged=False,public_fault='DISCONNECTED')
            for mode in ('early_eof','exit_error'):
                broken=start();token=str(uuid.uuid4());repo.claim_accepted_compute_session(token)
                try:pipe_fixture(mode,SchedulerLifecycleEvents(repo,broken),lambda:repo.check_session_cancellation(broken,token))
                except PublicError as exc:assert exc.code in (ErrorCode.BAD_STATE,ErrorCode.BAD_SHAPE)
                else:raise AssertionError('broken public pipe child was accepted')
                life.finish(broken,acknowledged=False,public_fault='DISCONNECTED')
            # Real concurrent transactions: either committed completion wins,
            # or cancellation keeps the slot until explicit coordinator cleanup.
            concurrent_outcomes=[]
            for _ in range(4):
                session=start();token=str(uuid.uuid4());repo.claim_accepted_compute_session(token)
                events=SchedulerLifecycleEvents(repo,session);events('AUTHENTICATED');events('RELEASING')
                rendezvous=threading.Barrier(2)
                def complete():
                    rendezvous.wait(timeout=5)
                    try:repo.complete_native_preparation(session,PreparationShare(session,'S',(11,)*1024,13));return 'completed'
                    except PublicError as exc:assert exc.code==ErrorCode.BAD_STATE;return 'cancel_won'
                def cancel():rendezvous.wait(timeout=5);return repo.request_session_cancel(owner,session)
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    a=pool.submit(complete);b=pool.submit(cancel);outcome=a.result(timeout=10);reply=b.result(timeout=10)
                if outcome=='completed':assert state(session)=='completed' and reply['state']=='completed'
                else:
                    assert state(session)=='releasing' and reply['cancel_requested'];expect(ErrorCode.BUSY,start)
                    life.finish(session,acknowledged=False,public_fault='DISCONNECTED')
                concurrent_outcomes.append(outcome)
            with repo.transaction() as c:
                assert c.execute("SELECT count(*) AS n FROM sessions WHERE state IN ('accepted','computing','releasing')").fetchone()['n']==0
            results.update({'schema_upgrade':'existing metadata preserved; missing columns/table added; legacy active session conservatively claimed until proven cleanup',
                'cancel_orders':'before acceptance, before claim, after claim, during computing, after completion; other account rejected',
                'active_slot':'retained on claimed cancellation until child/fds reaped and coordinator terminal transaction',
                'actual_pipe_checks':'8201-byte output drained with committed event ACKs; premature event EOF, child error and live cancellation terminate without ACK deadlock',
                'completion_cancel_concurrency':concurrent_outcomes,'remaining_active_sessions':0})
        finally:
            subprocess.run([str(pg/'pg_ctl'),'-D',str(data),'-m','fast','-w','stop'],check=True,capture_output=True)
            results['temporary_postgresql_stopped']=True
    (project/'records/cancellation_debug.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
