#!/bin/bash
#
# Measure the ASR stage on PACE-ICE. Partner A, Phase 2.
#
# This deliberately does NOT start vLLM: the brain and the TTS are still stand-ins, so the only
# thing being measured is faster-whisper, and skipping the LLM makes the job start in seconds
# instead of minutes.
#
# It runs the agent EITHER in a virtual environment OR in the Apptainer image, because on ICE the
# login nodes carry no apptainer at all and the measurement must not wait for that to be sorted out.
#
#   RUNTIME=venv       (default) a plain venv in scratch, built by tools/setup_pace_venv.sh
#   RUNTIME=apptainer  the image built from recipe/agent.def
#
# The venv is the quicker path to a number; the image is what peer testers build, so it has to work
# before submission. The agent code is identical either way — only how Python is reached differs.
#
# VENV PATH, from the root of the repository on a login node (it needs the internet):
#   export SCRATCH_DIR=$HOME/scratch && mkdir -p "$SCRATCH_DIR"
#   bash tools/setup_pace_venv.sh
#   sbatch bench_asr.sbatch
#
# APPTAINER PATH, once apptainer is reachable:
#   export SCRATCH_DIR=$HOME/scratch
#   mkdir -p "$SCRATCH_DIR"
#   export APPTAINER_CACHEDIR=$SCRATCH_DIR/cache APPTAINER_TMPDIR=$SCRATCH_DIR/tmp
#   mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
#   apptainer build --fakeroot "$SCRATCH_DIR/agent.sif" recipe/agent.def
#   sbatch --export=ALL,RUNTIME=apptainer bench_asr.sbatch

#SBATCH --job-name=a4-asr
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=a4-asr-%j.out

set -euo pipefail

RUNTIME="${RUNTIME:-venv}"
SCRATCH_DIR="${SCRATCH_DIR:-$HOME/scratch}"
IMAGE="${IMAGE:-$SCRATCH_DIR/agent.sif}"
VENV="${VENV:-$SCRATCH_DIR/asrenv}"
RECORDINGS_DIR="${RECORDINGS_DIR:-recordings}"
OUTPUT_DIR="${OUTPUT_DIR:-run_output/asr_${SLURM_JOB_ID}}"
BENCHMARK_RUNS="${BENCHMARK_RUNS:-20}"

REPO_ROOT="${SLURM_SUBMIT_DIR}"

export HF_HOME=/storage/ice1/0/4/lheck6/CS6604_ECE6256/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "node: $(hostname)"
echo "slurm cpus-per-task: ${SLURM_CPUS_PER_TASK}"
echo "runtime: ${RUNTIME}"
nvidia-smi -L

# OMP_NUM_THREADS is deliberately NOT set here, and --cpu-count is deliberately NOT passed to the
# agent. Both are left out so the output shows run_agent.py deriving the count from
# sched_getaffinity on its own, and shows the variables as they were before it did.
#
# PYTHONNOUSERSITE=1 in both paths: packages in ~/.local would otherwise shadow the ones that were
# actually pinned, and then nobody can say which version produced the number.

if [ "${RUNTIME}" = "apptainer" ]; then
    module load apptainer 2>/dev/null || true
    WORK=/work

    # faster-whisper's GPU engine needs the CUDA libraries pip put inside the image. These folders
    # are namespace packages, so their location comes from __path__.
    CUDA_LIBS=$(apptainer exec --env PYTHONNOUSERSITE=1 "${IMAGE}" python3 - <<'PY'
import nvidia.cublas.lib as cublas
import nvidia.cudnn.lib as cudnn
print(":".join([list(cublas.__path__)[0], list(cudnn.__path__)[0]]))
PY
)

    agent() {
        apptainer exec --nv \
            --bind "${REPO_ROOT}:/work" \
            --bind "${HF_HOME}:${HF_HOME}:ro" \
            --env LD_LIBRARY_PATH="${CUDA_LIBS}" \
            --env PYTHONPATH=/work \
            --env HF_HOME="${HF_HOME}" \
            --env HF_HUB_OFFLINE=1 \
            --env PYTHONNOUSERSITE=1 \
            "${IMAGE}" "$@"
    }
else
    if [ ! -x "${VENV}/bin/python3" ]; then
        echo "FATAL: no virtual environment at ${VENV}."
        echo "Build it on a LOGIN node (it needs the internet): bash tools/setup_pace_venv.sh"
        exit 1
    fi

    WORK="${REPO_ROOT}"

    export PYTHONNOUSERSITE=1
    export PYTHONPATH="${REPO_ROOT}"

    # The same CUDA libraries, found the same way, but inside the venv instead of the image.
    CUDA_LIBS=$("${VENV}/bin/python3" - <<'PY'
import nvidia.cublas.lib as cublas
import nvidia.cudnn.lib as cudnn
print(":".join([list(cublas.__path__)[0], list(cudnn.__path__)[0]]))
PY
)
    export LD_LIBRARY_PATH="${CUDA_LIBS}:${LD_LIBRARY_PATH:-}"

    agent() {
        if [ "$1" = "python3" ]; then
            shift
            "${VENV}/bin/python3" "$@"
        else
            "$@"
        fi
    }
fi

echo ""
echo "================ 1. what this job may actually use ================"
# The evidence for "the thread count is read from sched_getaffinity": on a shared node these two
# numbers are different, and only the second one is this job's.
agent python3 -c "
import os
print('os.cpu_count()                    ', os.cpu_count(), ' <- the whole NODE')
print('len(os.sched_getaffinity(0))      ', len(os.sched_getaffinity(0)), ' <- what Slurm gave THIS job')
print('OMP_NUM_THREADS before the agent  ', os.environ.get('OMP_NUM_THREADS', '(unset)'))
print('MKL_NUM_THREADS before the agent  ', os.environ.get('MKL_NUM_THREADS', '(unset)'))
import ctranslate2, faster_whisper
print('ctranslate2                       ', ctranslate2.__version__)
print('faster_whisper                    ', faster_whisper.__version__)
print('ctranslate2.get_cuda_device_count ', ctranslate2.get_cuda_device_count())
"

echo ""
echo "================ 2. is the shared cache serving the model offline? ================"
ls "${HF_HOME}/hub" | grep -i whisper || echo "NO WHISPER MODEL FOUND IN ${HF_HOME}/hub"

echo ""
echo "================ 3. GPU decode, float16, GREEDY (the comparison) ================"
agent python3 ${WORK}/run_agent.py --benchmark-asr \
    --recordings "${WORK}/${RECORDINGS_DIR}" \
    --benchmark-runs "${BENCHMARK_RUNS}" \
    --asr-device cuda --asr-compute-type float16 --asr-beam-size 1 \
    || echo ">>> SECTION 3 FAILED <<<"

echo ""
echo "================ 4. GPU decode, float16, BEAM 5 (the shipped default) ================"
agent python3 ${WORK}/run_agent.py --benchmark-asr \
    --recordings "${WORK}/${RECORDINGS_DIR}" \
    --benchmark-runs "${BENCHMARK_RUNS}" \
    --asr-device cuda --asr-compute-type float16 --asr-beam-size 5 || echo ">>> SECTION 4 FAILED <<<"

echo ""
echo "================ 5. CPU fallback, thread sweep ================"
# Run even though the graded path is the GPU: it is what proves the thread count matters, and it is
# the number to quote if the GPU is ever unavailable.
for threads in 1 2 4 8; do
    echo "---- --cpu-count ${threads} ----"
    agent python3 ${WORK}/run_agent.py --benchmark-asr \
        --recordings "${WORK}/${RECORDINGS_DIR}" \
        --benchmark-runs 8 \
        --asr-device cpu --cpu-count "${threads}" \
        | grep -E "^(cpu count|OMP_NUM_THREADS|MKL_NUM_THREADS|asr  |p50|p95|real time factor)" \
        || echo ">>> ${threads}-thread run FAILED <<<"
done

echo ""
echo "================ 6. a whole session: real ASR, stand-in brain and TTS ================"
agent python3 ${WORK}/run_agent.py \
    --recordings "${WORK}/${RECORDINGS_DIR}" \
    --output-dir "${WORK}/${OUTPUT_DIR}" \
    --fake-brain --fake-tts || echo ">>> SECTION 6 FAILED <<<"

echo ""
echo "================ 7. the checker ================"
agent python3 ${WORK}/check_trace.py --trace "${WORK}/${OUTPUT_DIR}/trace.jsonl" || true

echo ""
echo "================ 8. feed lag with NO agent work, for comparison ================"
# The floor of the machine: one thread waking on 20 ms deadlines with an idle main thread. If this
# is already near the 5 ms limit, the limit is about the host and not about the agent.
agent python3 -c "
import sys, threading
from voice_kit.clock import now_nanoseconds
from voice_kit.source import sleep_until

def feed():
    sys.setswitchinterval(0.0005)
    lags = []
    start = now_nanoseconds()
    for index in range(1, 251):
        deadline = start + index * 20_000_000
        sleep_until(deadline)
        lags.append((now_nanoseconds() - deadline) / 1e6)
    lags.sort()
    print(f'idle feed thread: median {lags[len(lags)//2]:.3f} ms  p95 {lags[int(.95*len(lags))]:.3f} ms  worst {lags[-1]:.3f} ms')

thread = threading.Thread(target=feed)
thread.start()
thread.join()
"

echo ""
echo "done. trace: ${OUTPUT_DIR}/trace.jsonl"
