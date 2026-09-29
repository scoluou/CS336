import sys
from multiprocessing import get_context

import pytest


def _probe_limit(sender):
    from .windows_memory_limit import limit_additional_commit

    try:
        with limit_additional_commit(1_000_000):
            small = bytearray(64_000)
            try:
                bytearray(2_000_000)
            except MemoryError:
                blocked = True
            else:
                blocked = False
        after_reset = bytearray(2_000_000)
        sender.send((len(small), blocked, len(after_reset)))
    except BaseException as exc:
        sender.send((type(exc).__name__, str(exc)))
    finally:
        sender.close()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Objects only")
def test_job_object_enforces_and_clears_memory_limit():
    context = get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_probe_limit, args=(sender,))
    try:
        process.start()
        sender.close()
        process.join(timeout=15)
        if process.is_alive():
            process.terminate()
            process.join()
            pytest.fail("memory limit probe timed out")
        assert receiver.poll(), f"memory limit probe exited without a result (exit code {process.exitcode})"
        assert receiver.recv() == (64_000, True, 2_000_000)
    finally:
        receiver.close()
        sender.close()
        if process.is_alive():
            process.terminate()
            process.join()
