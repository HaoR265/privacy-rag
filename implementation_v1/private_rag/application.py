"""Owner-local application wiring; each instance runs exactly one party.

No query, answer, source text or D failure bit is sent through management HTTP.
There is no reference backend, automatic retry or automatic ACL/publication.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
import time
from .activation import require_private_backend
from .client import UserClient, ProviderClient, _uuid, _sha
from .compute_input import WorkerBinding
from .compute_process import (WorkerSettings, LocalAnswer, LocalPreparationResult,
                              run_worker, retain_provider_preparation)
from .configuration import check_configuration
from .delivery import PreparationShare, canonical_json, digest
from .errors import ErrorCode, PublicError
from .model_input import InputPreparer, DocumentChunk, InputRejected
from .scheduler import ComputePorts, connect_worker_sockets
from .vault import ProviderVault


@dataclass(frozen=True,repr=False)
class LocalResponse:
    status: str
    text: str
    reached_length_limit: bool=False


@dataclass(frozen=True,repr=False)
class PreparedBlock:
    """D only. Successful own share plus the exact selected local body."""
    binding: WorkerBinding
    chunk: DocumentChunk
    share: PreparationShare


@dataclass(frozen=True,repr=False)
class LocalPreparationOutcome:
    # The failed state is private to D; S persists only its own fixed output.
    status: str
    prepared: PreparedBlock | None


class _LocalEvents:
    def __init__(self): self.phase='accepted'
    def __call__(self,event: str):
        expected={'accepted':('AUTHENTICATED','computing'),'computing':('RELEASING','releasing')}
        if self.phase not in expected or event!=expected[self.phase][0]:
            raise PublicError(ErrorCode.REPLAY_OR_ORDER)
        self.phase=expected[self.phase][1]


@contextmanager
def _accepted_public_phase(client,session: str):
    """Cancel only a failed/cancelled public phase, before private branching."""
    try: yield
    except BaseException:
        # The worker wrapper has already reaped C and closed all its lanes.
        # One bounded public cancellation request; never retry this session.
        client.cancel_after_public_failure(session)
        raise


def document_authorization_binding(document: str,version: int) -> str:
    _uuid(document)
    if type(version) is not int or version<1: raise PublicError(ErrorCode.BAD_FORMAT)
    return canonical_json({'document_id':document,'document_version':version}).decode('ascii')


def _preparer(settings: WorkerSettings,configuration: str) -> InputPreparer:
    root=Path(settings.root)
    check_configuration(root,_sha(configuration),server_weights=False)
    return InputPreparer(root/'artifacts/models',root/'artifacts/template/template_v1.json')


def _decode_answer(preparer: InputPreparer,answer: LocalAnswer) -> LocalResponse:
    if answer.status==1:
        return LocalResponse('insufficient_evidence','当前授权资料中没有足够的有效证据回答该问题。')
    if answer.status==2:
        return LocalResponse('numeric_configuration_failure','当前数值配置无法完成本次回答。')
    if answer.status not in (0,3): raise PublicError(ErrorCode.BAD_FORMAT)
    # Native returns only the prescribed final U output. No incremental answer
    # or intermediate evidence is released by this local rendering function.
    text=preparer.generation.decode([t for t,v in zip(answer.tokens,answer.validity) if v],skip_special_tokens=False)
    return LocalResponse('length_limit' if answer.status==3 else 'answered',text,answer.status==3)


def _wait_for_server_completion(client: ProviderClient,session: str):
    """Read-only persistence acknowledgement; never rerun/replay computation.

    No total computation limit is added. Network faults stop this caller; the
    source chunk remains encrypted, but unretained output is not deliverable.
    An explicit cancellation also propagates.
    """
    while True:
        state=client.session_status(session)
        if state['state']=='completed':
            if not state['release_acknowledged'] or state['public_fault'] is not None:
                raise PublicError(ErrorCode.BAD_STATE)
            return
        if state['state']=='aborted':
            raise PublicError(ErrorCode(state['public_fault']) if state['public_fault'] else ErrorCode.BAD_STATE)
        time.sleep(0.25)


class UserApplication:
    def __init__(self,client: UserClient,settings: WorkerSettings,compute_host: str,
                 *,ports: ComputePorts=ComputePorts()):
        self.client=client; self.settings=settings; self.host=compute_host; self.ports=ports

    def ask(self,raw_question: bytes | str) -> LocalResponse:
        """Local preprocessing -> full sync -> acceptance -> one C worker."""
        preparer=_preparer(self.settings,self.client.configuration)
        question=preparer.question(raw_question)  # InputRejected stays local.
        collection=self.client.synchronize()
        if collection.manifest['N']:
            require_private_backend()  # do not allocate a doomed accepted run
        accepted=self.client.accept_query()
        if accepted['status']=='no_authorized_material':
            return LocalResponse('no_authorized_material','当前没有可用于回答的授权资料')
        snapshot=accepted['snapshot']
        with _accepted_public_phase(self.client,snapshot['session_id']):
            binding=WorkerBinding(snapshot['session_id'],self.client.configuration,
                                  digest(canonical_json(snapshot)),'C','query')
            channels=connect_worker_sockets(self.host,self.ports)
            result=run_worker(binding,self.settings,channels,snapshot=snapshot,on_event=_LocalEvents(),
                              collection=collection,question=question)
            if not isinstance(result,LocalAnswer): raise PublicError(ErrorCode.BAD_STATE)
        return _decode_answer(preparer,result)


class ProviderApplication:
    def __init__(self,client: ProviderClient,settings: WorkerSettings,configuration: str,
                 compute_host: str,*,ports: ComputePorts=ComputePorts()):
        self.client=client; self.settings=settings; self.configuration=_sha(configuration)
        self.host=compute_host; self.ports=ports

    def prepare_document(self,raw_document: bytes | str,document: str,version: int) -> tuple[DocumentChunk,...]:
        """D local chunk preparation only, bound to an already selected version."""
        return _preparer(self.settings,self.configuration).document_chunks(
            raw_document,document_authorization_binding(document,version))

    def prepare_chunk(self,document: str,version: int,block_index: int,chunk: DocumentChunk,
                      *,vault: ProviderVault,retained_chunk_path: Path,
                      retained_preparation_path: Path) -> LocalPreparationOutcome:
        """One selected block; no ACL mutation, delivery or publication occurs."""
        require_private_backend()
        if type(block_index) is not int or block_index<0 or chunk.authorization_binding!=document_authorization_binding(document,version):
            raise PublicError(ErrorCode.VERSION_MISMATCH)
        if Path(retained_chunk_path).resolve()==Path(retained_preparation_path).resolve():
            raise PublicError(ErrorCode.BAD_FORMAT)
        preparer=_preparer(self.settings,self.configuration)
        text=preparer._text(chunk.text)
        if (text!=chunk.text or chunk.character_start<0 or chunk.character_end-chunk.character_start!=len(text)
            or chunk.body!=preparer._slots(preparer._body_ids(text),256)
            or chunk.embedding!=preparer._embedding(text,257)):
            raise InputRejected('DOCUMENT_CHUNK_BINDING_MISMATCH')
        # The source/body and all local derivatives are encrypted before any
        # private computation starts. These JSON bytes never leave D's vault.
        identity=canonical_json({'document_id':document,'document_version':version,'block_index':block_index}).decode('ascii')
        retained={'format_version':1,'document_id':document,'document_version':version,'block_index':block_index,
                  'configuration':self.configuration,'text':text,'character_start':chunk.character_start,
                  'character_end':chunk.character_end,'body_tokens':chunk.body.tokens,'body_validity':chunk.body.validity,
                  'embedding_tokens':chunk.embedding.tokens,'embedding_validity':chunk.embedding.validity}
        vault.save(Path(retained_chunk_path),identity,canonical_json(retained))
        accepted=self.client.start_preparation(document,version,block_index)
        with _accepted_public_phase(self.client,accepted['session_id']):
            if accepted['snapshot']['model_numeric_version']!=self.configuration:
                raise PublicError(ErrorCode.VERSION_MISMATCH)
            binding=WorkerBinding(accepted['session_id'],self.configuration,accepted['snapshot_sha256'],'C','prepare')
            channels=connect_worker_sockets(self.host,self.ports)
            result=run_worker(binding,self.settings,channels,snapshot=accepted['snapshot'],on_event=_LocalEvents(),document=chunk.embedding)
            if not isinstance(result,LocalPreparationResult) or result.party!='C' or type(result.client_failed) is not bool:
                raise PublicError(ErrorCode.BAD_STATE)
            # Both private outcomes perform the identical public completion wait.
            # A success-only GET sequence would disclose the private failure bit.
            _wait_for_server_completion(self.client,binding.session_id)
        if result.client_failed:
            return LocalPreparationOutcome('numeric_configuration_failure',None)
        share=retain_provider_preparation(binding,result,vault,Path(retained_preparation_path))
        return LocalPreparationOutcome('prepared',PreparedBlock(binding,chunk,share))
