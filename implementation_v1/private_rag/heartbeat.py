"""RUN-003: heartbeat worker independent from the compute subprocess.

Use an independent mTLS control connection; no total query timeout is imposed.
"""
import asyncio
import struct
import time
import uuid
from .errors import ErrorCode, PublicError

MESSAGE = struct.Struct("!4s16sQ4s") # fixed 32 bytes
INTERVAL = 30.0
INACTIVITY = 180.0

class Heartbeat:
    def __init__(self, session_id: str):
        self.session = uuid.UUID(session_id).bytes
        self.sent = self.received = 0
        self.last_received = time.monotonic()
        self.stopped = asyncio.Event()

    async def run(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        async def sender():
            deadline = time.monotonic()
            while not self.stopped.is_set():
                writer.write(MESSAGE.pack(b"PHB1", self.session, self.sent, bytes(4)))
                await writer.drain()
                self.sent += 1
                deadline += INTERVAL
                try:
                    await asyncio.wait_for(self.stopped.wait(), max(0, deadline-time.monotonic()))
                except asyncio.TimeoutError:
                    pass
        async def receiver():
            while not self.stopped.is_set():
                remaining = INACTIVITY - (time.monotonic()-self.last_received)
                try:
                    raw = await asyncio.wait_for(reader.readexactly(MESSAGE.size), max(0, remaining))
                except (asyncio.TimeoutError, asyncio.IncompleteReadError):
                    raise PublicError(ErrorCode.DISCONNECTED) from None
                magic, sid, sequence, reserved = MESSAGE.unpack(raw)
                if magic != b"PHB1" or sid != self.session or sequence != self.received or reserved != bytes(4):
                    raise PublicError(ErrorCode.REPLAY_OR_ORDER)
                self.received += 1
                self.last_received = time.monotonic()
        tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for task in done:
                task.result()
        finally:
            self.stopped.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            writer.close()
            await writer.wait_closed()
