"""How many CPUs this process may actually use, and the thread environment that follows from it.

This module is imported at the VERY TOP of ``run_agent.py``, before anything that could pull in
PyTorch, CTranslate2 or ONNX Runtime. It is standard library only on purpose: importing the thing
that decides the thread count must never itself be what loads the native library whose thread pool
we are trying to size.

Two separate problems live here, and they are easy to confuse:

1. HOW MANY CPUs WE HAVE. ``os.cpu_count()`` answers "how many CPUs does this machine have", which
   on a PACE-ICE node is the whole node — 64, 128 — regardless of what Slurm gave this job.
   ``len(os.sched_getaffinity(0))`` answers "how many CPUs is this process allowed to run on",
   which under Slurm's cgroup is exactly ``--cpus-per-task``. Sizing a thread pool from
   ``cpu_count()`` on a shared node starts an order of magnitude more threads than there are cores
   to run them, and they then fight each other for the cores the job really owns: more context
   switching, more lock contention, and a WORSE p95 than a single thread would give.

2. TELLING THE NATIVE LIBRARIES. ``OMP_NUM_THREADS`` and ``MKL_NUM_THREADS`` are read by OpenMP and
   MKL when the native library is loaded and its thread pool is first built — not at each call. By
   the time ``import ctranslate2`` (or torch, or onnxruntime) has returned, the pool already exists
   and setting the variable afterwards changes nothing. That is why :func:`apply_thread_environment`
   runs before those imports, and why this module may not import them.

The environment variables are not the whole story either: CTranslate2 takes its intra-op thread
count from the ``cpu_threads`` argument of its constructor, and ONNX Runtime from its session
options. Those are passed explicitly in ``asr.py`` (and belong in the TTS stage too). The variables
here cover the libraries that only read the environment.
"""

from __future__ import annotations

import os
import sys

# The variables that have to be right BEFORE the native libraries load. OMP covers CTranslate2's
# OpenMP build, PyTorch and ONNX Runtime; MKL covers Intel's math kernels underneath them.
THREAD_ENVIRONMENT_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")

CPU_COUNT_FLAG = "--cpu-count"


def affinity_is_available() -> bool:
    """Return whether this platform can say which CPUs the process is actually allowed to use.

    ``os.sched_getaffinity`` is Linux only. PACE-ICE is Linux, so the graded run always takes the
    honest path; macOS does not have the call at all, so development on a laptop falls back.
    """
    return hasattr(os, "sched_getaffinity")


def available_cpu_count() -> int:
    """Return how many CPUs this process may use: its affinity set, or the machine as a fallback."""
    if affinity_is_available():
        return len(os.sched_getaffinity(0))

    # Fallback for macOS development only. It is the WRONG number on a shared cluster node, which is
    # exactly why the real run never reaches this line.
    return os.cpu_count() or 1


def cpu_count_source() -> str:
    """Return the name of the call :func:`available_cpu_count` used, for the banner and the report."""
    return "len(os.sched_getaffinity(0))" if affinity_is_available() else "os.cpu_count()"


def requested_cpu_count_from_argv(argv: list[str] | None = None) -> int | None:
    """Return the ``--cpu-count`` of the command line, or None, without running argparse.

    The thread environment has to be set before the heavy imports, and the heavy imports happen
    before ``main()`` can parse anything, so this one flag is read by hand first. ``argparse`` still
    declares it, so ``--help`` documents it and a bad value is still rejected the normal way; this
    function only has to be right about a value that parses.
    """
    argv = sys.argv[1:] if argv is None else argv

    for index, argument in enumerate(argv):
        value = None

        if argument == CPU_COUNT_FLAG and index + 1 < len(argv):
            value = argv[index + 1]

        elif argument.startswith(f"{CPU_COUNT_FLAG}="):
            value = argument.split("=", 1)[1]

        if value is not None:
            try:
                parsed = int(value)

            except ValueError:
                return None

            return parsed if parsed > 0 else None

    return None


def apply_thread_environment(cpu_count: int) -> list[tuple[str, str | None, str]]:
    """Set the thread variables to ``cpu_count`` and return (name, before, after) for each.

    The previous value is returned rather than thrown away so the banner can print what the
    environment said before this process touched it. That before/after pair is the evidence that the
    thread count came from the job and not from the node.
    """
    changes = []

    for name in THREAD_ENVIRONMENT_VARIABLES:
        before = os.environ.get(name)
        after = str(cpu_count)
        os.environ[name] = after
        changes.append((name, before, after))

    return changes
