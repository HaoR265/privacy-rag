"""Memory-only scheduler/worker ABI; never put question input into management JSON.

The wire contents are local owner plaintext or this party's random shares. A
sealed Linux memfd transfers them to the same party's child process. It is not
an encrypted transport or a way to transfer the two shares to one process.
"""
from __future__ import annotations
from .memfd import create_memfd, seal_memfd
import fcntl
import os
import struct
import uuid
from dataclasses import dataclass
from .delivery import ShareRecord
from .errors import ErrorCode,PublicError
from .model_input import LocalTokenSlots,QuestionInput

HEADER=struct.Struct('<8sBB6x16s32s32sQ')
MAGIC=b'PRAGIN01'

@dataclass(frozen=True)
class WorkerBinding:
    session_id: str
    configuration: str
    snapshot_sha256: str
    party: str
    mode: str
    def encode(self,count: int)->bytes:
        if self.party not in ('C','S') or self.mode not in ('prepare','query') or count<0:
            raise PublicError(ErrorCode.BAD_FORMAT)
        try:
            session=uuid.UUID(self.session_id).bytes
            config=bytes.fromhex(self.configuration);snapshot=bytes.fromhex(self.snapshot_sha256)
        except ValueError:
            raise PublicError(ErrorCode.BAD_FORMAT) from None
        if len(config)!=32 or len(snapshot)!=32:raise PublicError(ErrorCode.BAD_FORMAT)
        return HEADER.pack(MAGIC,1 if self.mode=='prepare' else 2,0 if self.party=='C' else 1,session,config,snapshot,count)

def _slots(value: LocalTokenSlots,count: int,maximum: int)->bytes:
    if len(value.tokens)!=count or any(type(x)is not int or x<0 or x>maximum for x in value.tokens):
        raise PublicError(ErrorCode.BAD_SHAPE)
    return struct.pack(f'<{count}Q',*value.tokens)+bytes(value.validity)

def sealed_input(binding: WorkerBinding,records: tuple[ShareRecord,...],*,question: QuestionInput|None=None,document: LocalTokenSlots|None=None)->int:
    """Caller closes returned fd after spawn; no disk fallback exists."""
    header=binding.encode(len(records))
    if binding.mode=='prepare' and records:raise PublicError(ErrorCode.BAD_SHAPE)
    if binding.mode=='query' and not records:raise PublicError(ErrorCode.BAD_SHAPE)
    if binding.party=='S' and (question is not None or document is not None):raise PublicError(ErrorCode.BAD_STATE)
    slots=[]
    if binding.party=='C':
        if binding.mode=='prepare':
            if document is None or question is not None:raise PublicError(ErrorCode.BAD_STATE)
            slots=[_slots(document,257,151668)]
        else:
            if question is None or document is not None:raise PublicError(ErrorCode.BAD_STATE)
            slots=[_slots(question.embedding,160,151668),_slots(question.body,128,151642)]
    if any(r.party!=('U' if binding.party=='C' else 'S') for r in records):raise PublicError(ErrorCode.BAD_STATE)
    fd=create_memfd('private-rag-session-input',allow_sealing=True)
    try:
        with os.fdopen(os.dup(fd),'wb') as stream:
            stream.write(header)
            for part in slots:stream.write(part)
            for record in records:stream.write(record.encode())
        os.lseek(fd,0,os.SEEK_SET)
        seal_memfd(fd)
        return fd
    except BaseException:
        os.close(fd);raise
