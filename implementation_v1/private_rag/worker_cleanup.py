"""Local worker cancellation: disconnect, grace, then terminate and reap.

No computation deadline is applied. These waits begin only when run_worker is
already leaving. This Linux launcher owns one native child (no descendants).
"""
from contextlib import contextmanager
from dataclasses import dataclass
import os
import signal
import socket
import subprocess
import sys
import time


@contextmanager
def defer_cancellation_signals():
    """Protect fd ownership bookkeeping; preserve an exception already in flight.

    pthread_sigmask is per thread. This protects normal main-thread SIGINT/SIGTERM
    delivery, not arbitrary asynchronous exceptions or unrelated threads closing
    our fds. The caller must exclusively own its descriptors.
    """
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try:
        yield
    finally:
        failed = sys.exc_info()[0] is not None
        try:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)
        except KeyboardInterrupt:
            if not failed:
                raise


def close_owned(owned: set[int], fd: int):
    with defer_cancellation_signals():
        if fd not in owned:
            return
        owned.remove(fd)
        try:
            os.close(fd)
        except OSError:
            pass


def duplicate_lanes(descriptors: tuple[int, int, int], owned: set[int]) -> tuple[int, ...]:
    """Already-validated connected sockets only; never inherit the copies."""
    copies = []
    with defer_cancellation_signals():
        for fd in descriptors:
            duplicate = os.dup(fd)  # Python creates a non-inheritable fd.
            owned.add(duplicate)
            copies.append(duplicate)
    return tuple(copies)


@dataclass(frozen=True)
class CleanupOutcome:
    exit_code: int | None
    disconnected: bool
    term_sent: bool
    kill_sent: bool
    lane_shutdown_failures: int
    # This reports process handling only; it never asserts an aborted ledger.


def _wait_until(process, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while True:
        try:
            if process.poll() is not None:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            process.wait(timeout=min(remaining, 0.25))
            return True
        except subprocess.TimeoutExpired:
            pass
        except KeyboardInterrupt:
            # Repeated cancellation cannot skip the reap/owned-fd cleanup.
            pass


def cleanup_worker(process, owned: set[int], lanes: tuple[int, ...], *,
                   grace_seconds: float = 30.0, term_seconds: float = 5.0) -> CleanupOutcome:
    """Called after failure or after completed collection, never during work.

    Override durations are for isolated short OS diagnostics only. A normal
    exited worker is closed without shutdown; an active child is disconnected
    first, including ACK and output/event pipes, so native cleanup can run.
    """
    term = killed = disconnected = False
    lane_failures = 0
    with defer_cancellation_signals():
        try:
            if process is not None and process.poll() is None:
                for fd in lanes:
                    if fd not in owned:
                        continue
                    try:
                        channel = socket.socket(fileno=fd)
                    except OSError:
                        # A broken local lane must not skip child reaping.
                        # Keep fd ownership: the common close pass handles it.
                        lane_failures += 1
                        continue
                    try:
                        try:
                            channel.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            lane_failures += 1
                    finally:
                        channel.detach()
                disconnected = True
                for fd in tuple(owned):
                    close_owned(owned, fd)
                if not _wait_until(process, grace_seconds):
                    try:
                        process.terminate()
                        term = True
                    except ProcessLookupError:
                        pass
                    if not _wait_until(process, term_seconds):
                        try:
                            process.kill()
                            killed = True
                        except ProcessLookupError:
                            pass
                        while not _wait_until(process, 0.25):
                            pass
        finally:
            for fd in tuple(owned):
                close_owned(owned, fd)
        return CleanupOutcome(None if process is None else process.returncode,
                              disconnected, term, killed, lane_failures)
