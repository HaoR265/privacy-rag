"""Production-facing management CLI. Reference and reconstruction have no flags here."""
import argparse
import asyncio
import json
import signal
from pathlib import Path
from .activation import report, require_private_backend
from .schedule import query_schedule, schedule_hash
from dataclasses import asdict


def main():
    parser=argparse.ArgumentParser(prog='private-rag')
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('status')
    p=commands.add_parser('schedule');p.add_argument('--authorized-blocks',type=int,required=True)
    p=commands.add_parser('init-pki');p.add_argument('--directory',type=Path,required=True)
    for name in ('init-db','enroll','serve','compute-once'):
        p=commands.add_parser(name);p.add_argument('--dsn',required=True);p.add_argument('--objects',type=Path,required=True)
        if name=='enroll':
            p.add_argument('--account-id',required=True);p.add_argument('--role',choices=['D','U'],required=True);p.add_argument('--certificate-sha256',required=True)
        if name in ('serve','compute-once'):
            p.add_argument('--cert',required=True);p.add_argument('--key',required=True);p.add_argument('--ca',required=True)
            p.add_argument('--host',default='127.0.0.1')
        if name=='serve': p.add_argument('--port',type=int,default=8443)
        if name=='compute-once':
            p.add_argument('--root',type=Path,required=True)
            p.add_argument('--peer-certificates',type=Path,required=True,help='operator-enrolled peer certificates named SHA256.pem')
            p.add_argument('--peer-name',default='localhost')
            p.add_argument('--ledger-directory',type=Path,required=True)
            p.add_argument('--compute-port',type=int,default=9443)
            p.add_argument('--client-heartbeat-port',type=int,default=9444)
            p.add_argument('--server-heartbeat-port',type=int,default=9445)
            p.add_argument('--admission-timeout',type=float,default=30)
    args=parser.parse_args()
    if args.command=='status':
        print(json.dumps(report(),indent=2));return
    if args.command=='schedule':
        print(json.dumps({'sha256':schedule_hash(args.authorized_blocks),'stages':[asdict(x) for x in query_schedule(args.authorized_blocks)]},indent=2));return
    if args.command=='init-pki':
        from .pki import create_local_pki
        print(json.dumps(create_local_pki(args.directory),indent=2));return
    if args.command=='compute-once':
        from .errors import PublicError,ErrorCode
        # An operator's normal process stop must unwind the worker wrapper and
        # coordinator before the global active session is released. SIGKILL
        # remains an operator recovery case, never an automatic old-run retry.
        previous_termination_handler=signal.getsignal(signal.SIGTERM)
        def terminate_compute(signum,frame):
            raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM,terminate_compute)
        try:
            # Check before constructing stores, opening a DB or listeners.
            require_private_backend()
            from .database import Repository
            from .storage import ObjectStore
            from .scheduler import ComputePorts,ComputeScheduler
            from .compute_process import WorkerSettings
            repo=Repository(args.dsn,ObjectStore(args.objects))
            def settings(admission):
                fingerprint=admission['peer_certificate_sha256']
                if len(fingerprint)!=64 or any(c not in '0123456789abcdef' for c in fingerprint):
                    raise PublicError(ErrorCode.BAD_FORMAT)
                peer=args.peer_certificates/(fingerprint+'.pem')
                if not peer.is_file() or peer.is_symlink(): raise PublicError(ErrorCode.MATERIALS_NOT_READY)
                return WorkerSettings(args.root,Path(args.ca),Path(args.cert),Path(args.key),peer,args.peer_name,fingerprint,args.ledger_directory)
            ports=ComputePorts(args.compute_port,args.client_heartbeat_port,args.server_heartbeat_port)
            session=ComputeScheduler(repo,settings,host=args.host,ports=ports).serve_one(
                admission_timeout=args.admission_timeout)
            print(json.dumps({'status':'completed','session_id':session}));return
        except PublicError as exc:
            print(json.dumps({'error':exc.code.value}));raise SystemExit(2) from None
        except KeyboardInterrupt:
            print(json.dumps({'error':ErrorCode.DISCONNECTED.value}));raise SystemExit(130) from None
        except Exception:
            print(json.dumps({'error':ErrorCode.BAD_STATE.value}));raise SystemExit(2) from None
        finally:
            signal.signal(signal.SIGTERM,previous_termination_handler)
    from .database import Repository
    from .storage import ObjectStore
    repo=Repository(args.dsn,ObjectStore(args.objects))
    if args.command=='init-db':
        repo.initialize();print('{"status":"database_initialized"}');return
    if args.command=='enroll':
        repo.enroll_account(args.account_id,args.role,args.certificate_sha256);print('{"status":"account_enrolled"}');return
    from .management import create_app
    from .tls_server import tls_context, serve
    async def run():
        server=await serve(create_app(repo,activation_check=require_private_backend),args.host,args.port,tls_context(args.cert,args.key,args.ca,server=True))
        async with server:
            await server.serve_forever()
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass

if __name__=='__main__':
    main()
