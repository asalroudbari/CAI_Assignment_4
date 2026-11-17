#!/bin/bash
#
# Build the virtual environment that bench_asr.sbatch uses when RUNTIME=venv (its default).
#
# RUN THIS ON A LOGIN NODE. It downloads packages, and compute nodes have no internet.
#
#   export SCRATCH_DIR=$HOME/scratch && mkdir -p "$SCRATCH_DIR"
#   bash tools/setup_pace_venv.sh
#
# WHY THIS EXISTS AT ALL: the graded path is the Apptainer image built from recipe/agent.def, and
# peer testers build that image themselves. But ICE login nodes carry no apptainer, and the ASR
# measurement should not wait for that to be resolved. This environment runs the SAME agent code
# against the SAME pinned versions, so the numbers it produces are the agent's numbers; the only
# thing it does not prove is that the container recipe builds.
#
# It installs the ASR packages only. piper-tts is in recipe/requirements.txt because the final image
# needs it, but this job never reaches the TTS stage (it runs with --fake-tts), so leaving it out
# keeps one more thing from going wrong on the way to a number.

set -euo pipefail

SCRATCH_DIR="${SCRATCH_DIR:-$HOME/scratch}"
VENV="${VENV:-$SCRATCH_DIR/asrenv}"

# The same pins as recipe/requirements.txt, so the venv and the image measure the same software.
FASTER_WHISPER_VERSION="${FASTER_WHISPER_VERSION:-1.2.1}"
CUBLAS_VERSION="${CUBLAS_VERSION:-12.8.4.1}"
CUDNN_VERSION="${CUDNN_VERSION:-9.10.2.21}"

mkdir -p "$SCRATCH_DIR"

echo "python:  $(command -v python3)  $(python3 -V 2>&1)"
echo "target:  ${VENV}"
echo ""

if [ -x "${VENV}/bin/python3" ]; then
    echo "An environment is already there. Delete it to rebuild:  rm -rf ${VENV}"
else
    python3 -m venv "${VENV}"
fi

# PYTHONNOUSERSITE so that packages in ~/.local cannot shadow the pinned ones during the install.
export PYTHONNOUSERSITE=1

"${VENV}/bin/python3" -m pip install --upgrade pip
"${VENV}/bin/python3" -m pip install \
    "faster-whisper==${FASTER_WHISPER_VERSION}" \
    "nvidia-cublas-cu12==${CUBLAS_VERSION}" \
    "nvidia-cudnn-cu12==${CUDNN_VERSION}"

echo ""
echo "================ what landed ================"
"${VENV}/bin/python3" - <<'PY'
import ctranslate2
import faster_whisper
import nvidia.cublas.lib as cublas
import nvidia.cudnn.lib as cudnn

print("faster_whisper      ", faster_whisper.__version__)
print("ctranslate2         ", ctranslate2.__version__)
print("cublas lib          ", list(cublas.__path__)[0])
print("cudnn lib           ", list(cudnn.__path__)[0])
# 0 here on a login node is expected and fine: login nodes have no GPU. What matters is that the
# same call inside the job, on the compute node, reports a device.
print("cuda devices here   ", ctranslate2.get_cuda_device_count(), "(0 is expected on a login node)")
PY

echo ""
echo "ready. submit with:  sbatch bench_asr.sbatch"
