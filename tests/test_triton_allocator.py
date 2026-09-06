"""Triton scratch allocator: installed in the launching context before every kernel launch.

Runs standalone (python tests/test_triton_allocator.py) or under pytest; set PYTHONPATH=src when the package is not installed.
"""

import sys
import threading
import time

import torch

from sdnq.common import set_triton_allocator, triton_scratch, torch_device


def current_allocator():
    from triton.runtime import _allocation
    return _allocation._allocator.get()


def test_install_in_current_context():
    set_triton_allocator()
    assert current_allocator() is triton_scratch, "the scratch allocator is not installed in the calling context"


def test_new_thread_starts_without_it():
    # Triton keeps the allocator in a ContextVar, so a launch from a fresh thread (a new request) starts from the null allocator
    seen = {}

    def probe():
        seen["before"] = current_allocator()
        set_triton_allocator()
        seen["after"] = current_allocator()

    set_triton_allocator()
    worker = threading.Thread(target=probe)
    worker.start()
    worker.join()
    assert seen["before"] is not triton_scratch, "a new thread inherited the allocator, so the per-launch install is not needed"
    assert seen["after"] is triton_scratch, "the install did not take effect inside the new thread"


def test_scratch_buffer():
    buf = triton_scratch(4096, 128, None)
    assert buf.dtype == torch.int8 and buf.numel() == 4096, f"unexpected scratch buffer {buf.dtype} {tuple(buf.shape)}"
    assert buf.device.type == torch_device.type, f"scratch buffer on {buf.device}, expected {torch_device.type}"


SUITES = {
    "allocator": [test_install_in_current_context, test_new_thread_starts_without_it, test_scratch_buffer],
}


def run_all() -> bool:
    started = time.time()
    passed = failed = 0
    for cat, tests in SUITES.items():
        print(f"=== {cat} ===")
        for fn in tests:
            try:
                fn()
                print(f"  PASS: {fn.__name__}")
                passed += 1
            except Exception as e: # pylint: disable=broad-except
                print(f"  FAIL: {fn.__name__} ({type(e).__name__}: {e})")
                failed += 1
    print(f"Total: {passed} passed, {failed} failed in {time.time() - started:.1f}s")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
