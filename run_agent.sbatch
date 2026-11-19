#!/bin/bash
#
# Run the full agent (real ASR, real LLM, real TTS) once on one H100 node of PACE-ICE, against an LLM
# served on the same node. Adapted from recipe/run_agent.sbatch.
#
# Build the image once, from the root of the repository, on a login node:
#   export SCRATCH_DIR=$HOME/scratch
#   mkdir -p "$SCRATCH_DIR"
#   export APPTAINER_CACHEDIR=$SCRATCH_DIR/cache APPTAINER_TMPDIR=$SCRATCH_DIR/tmp
#   mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
#   apptainer build --fakeroot "$SCRATCH_DIR/agent.sif" recipe/agent.def
#
# Run it, from the root of the repository:
#   sbatch run_agent.sbatch
#   sbatch --export=ALL,RECORDINGS_DIR=peer_recordings run_agent.sbatch
#   sbatch --export=ALL,LLM_MODEL=meta-llama/Llama-3.2-3B-Instruct,LLM_NLU=rules run_agent.sbatch
#   sbatch --export=ALL,EXTRA_AGENT_ARGS="--llm-kb-mode full" run_agent.sbatch
#
# Default: gpt-oss-20b at reasoning effort low, reading every utterance into JSON actions
# (--llm-nlu llm, intent_reader.py) before it phrases the reply.
#
# The trace and the captured TTS audio land in run_output/<job id>/ (trace.jsonl, tts/).

#SBATCH --job-name=a4-agent
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=a4-agent-%j.out

set -euo pipefail

# ---- Settings (every one can be overridden from the environment) ------------------------------
SCRATCH_DIR="${SCRATCH_DIR:-$HOME/scratch}"
IMAGE="${IMAGE:-$SCRATCH_DIR/agent.sif}"
LLM_MODEL="${LLM_MODEL:-openai/gpt-oss-20b}"
LLM_NLU="${LLM_NLU:-llm}"                # llm | hybrid | rules (see run_agent.py --llm-nlu)
RECORDINGS_DIR="${RECORDINGS_DIR:-recordings}"
OUTPUT_DIR="${OUTPUT_DIR:-run_output/${SLURM_JOB_ID}}"
EXTRA_AGENT_ARGS="${EXTRA_AGENT_ARGS:-}"

# gpt-oss-20b reasons before it answers: vLLM separates that channel (--reasoning-parser), the agent
# asks for the lowest effort, and the token cap leaves room for the reasoning before the answer.
case "${LLM_MODEL}" in
    *gpt-oss*) DEFAULT_VLLM_ARGS="--reasoning-parser openai_gptoss"
               MODEL_AGENT_ARGS="--llm-reasoning-effort low --llm-max-tokens 512" ;;
    *)         DEFAULT_VLLM_ARGS=""
               MODEL_AGENT_ARGS="" ;;
esac
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-${DEFAULT_VLLM_ARGS}}"
# ----------------------------------------------------------------------------------------------

REPO_ROOT="${SLURM_SUBMIT_DIR}"
CPU_COUNT="${SLURM_CPUS_PER_TASK}"

if [ ! -f "${IMAGE}" ]; then
    echo "FATAL: no image at ${IMAGE}. Build it first (see the top of this file)."
    exit 1
fi

# The professor's shared model cache. Nothing is downloaded at run time.
export HF_HOME=/storage/ice1/0/4/lheck6/CS6604_ECE6256/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "node: $(hostname)  model: ${LLM_MODEL}  cpus: ${CPU_COUNT}"
echo "recordings: ${RECORDINGS_DIR}  output: ${OUTPUT_DIR}  image: ${IMAGE}"
nvidia-smi -L
module load apptainer 2>/dev/null || true

# ---- Serve the LLM -----------------------------------------------------------------------------
# vLLM comes from the shared A4 environment, outside the image.
CONDA_SH=/usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
module load anaconda3 2>/dev/null || true
source "${CONDA_SH}"
conda activate /storage/ice1/0/4/lheck6/CS6604_ECE6256/envs/CAI_A4

# Ignore packages in ~/.local so that the shared environment is what runs.
export PYTHONNOUSERSITE=1

# Jobs that share a node share its network, so each job takes a port from its own job id.
PORT=$((20000 + SLURM_JOB_ID % 20000))
VLLM_LOG="vllm-${SLURM_JOB_ID}.log"

python -c "import vllm; print('vllm', vllm.__version__)" || true

# --seed 42 makes the server's sampling reproducible; the agent also sends temperature 0 and its own
# per-request seed (--llm-seed, default 20241118), so a rerun of one recording gives the same reply.
python -m vllm.entrypoints.openai.api_server \
    --model "${LLM_MODEL}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.60 \
    --seed 42 \
    ${VLLM_EXTRA_ARGS} \
    > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
trap 'kill "${VLLM_PID}" 2>/dev/null || true' EXIT

# Loading the weights takes a few minutes.
for attempt in $(seq 1 240); do
    if curl -s "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "FATAL: vLLM stopped. The end of its log:"
        tail -n 40 "${VLLM_LOG}"
        exit 1
    fi
    sleep 5
done

# The server on the port must be ours and must serve our model.
if ! curl -s "http://127.0.0.1:${PORT}/v1/models" | grep -q "\"${LLM_MODEL}\""; then
    echo "FATAL: the server on port ${PORT} does not serve ${LLM_MODEL}"
    curl -s "http://127.0.0.1:${PORT}/v1/models" || true
    exit 1
fi
echo "LLM server ready on port ${PORT}"

# ---- Run the agent -------------------------------------------------------------------------------
# faster-whisper's GPU engine needs the CUDA libraries that pip installed inside the image.
# These folders are namespace packages, so their location comes from __path__.
CUDA_LIBS=$(apptainer exec --env PYTHONNOUSERSITE=1 "${IMAGE}" python3 - <<'PY'
import nvidia.cublas.lib as cublas
import nvidia.cudnn.lib as cudnn
print(":".join([list(cublas.__path__)[0], list(cudnn.__path__)[0]]))
PY
)

# PYTHONNOUSERSITE: Apptainer mounts the home directory, and packages in ~/.local would shadow the
# image's. Thread counts: run_agent.py sets OMP/MKL from --cpu-count before any import and passes the
# same count to CTranslate2 (faster-whisper) and ONNX Runtime (Piper) in code.
# shellcheck disable=SC2086
apptainer exec --nv \
    --bind "${REPO_ROOT}:/work" \
    --bind "${HF_HOME}:${HF_HOME}:ro" \
    --env LD_LIBRARY_PATH="${CUDA_LIBS}" \
    --env PYTHONPATH=/work \
    --env HF_HOME="${HF_HOME}" \
    --env HF_HUB_OFFLINE=1 \
    --env PYTHONNOUSERSITE=1 \
    --env OMP_NUM_THREADS="${CPU_COUNT}" \
    "${IMAGE}" \
    python3 /work/run_agent.py \
    --recordings "/work/${RECORDINGS_DIR}" \
    --output-dir "/work/${OUTPUT_DIR}" \
    --llm-model "${LLM_MODEL}" \
    --llm-base-url "http://127.0.0.1:${PORT}/v1" \
    --cpu-count "${CPU_COUNT}" \
    --llm-nlu "${LLM_NLU}" \
    ${MODEL_AGENT_ARGS} \
    ${EXTRA_AGENT_ARGS}

echo ""
echo "================ checker ================"
apptainer exec \
    --bind "${REPO_ROOT}:/work" \
    --env PYTHONPATH=/work \
    --env PYTHONNOUSERSITE=1 \
    "${IMAGE}" \
    python3 /work/check_trace.py --trace "/work/${OUTPUT_DIR}/trace.jsonl" || true
