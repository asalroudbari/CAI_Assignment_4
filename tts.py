"""The real speech synthesis stage, on Piper. Owned by Partner B.

WHAT THIS STAGE IS MEASURED ON
------------------------------
The shim stamps ``tts_first_audio`` on the first chunk written to it whose root mean square reaches
200, and that stamp is the end of the headline latency number. So what matters here is not how fast
a whole sentence is synthesised but how soon the FIRST audible bytes exist.

Three things follow, and they are the reason this file is not four lines around ``piper``:

* **One sentence at a time, never the whole reply.** Piper's ``synthesize`` produces one audio chunk
  per sentence, after the whole sentence has been through the model. The brain already yields whole
  sentences as they complete and the turn loop hands each one straight here, so the first audio is
  the synthesis time of the FIRST short sentence, not of the answer.
* **The thread count comes from the job.** ``PiperVoice.load`` builds its ONNX Runtime session with
  a default ``SessionOptions()``, and ONNX Runtime sizes its intra-op pool from the MACHINE — 64 on
  a PACE-ICE node, when Slurm gave the job 8. That is the trap the assignment README names. This
  file therefore builds the session itself, with ``intra_op_num_threads`` set from the CPUs the job
  actually has, and only falls back to ``PiperVoice.load`` (loudly) if that ever stops working.
* **Nothing is put in front of the audio.** No padding, no lead-in silence, no warm-up tone. The
  first chunk yielded is the first chunk Piper produced.

WHAT IT COSTS, MEASURED
-----------------------
On the development laptop, lessac medium on CPU at four threads, time from the call to the first
(and only) chunk of a sentence:

    "Booked."                                            32 ms   0.68 s of audio
    "Ten minutes for the tea, starting now."             85 ms   2.31 s of audio
    "Cedar Room on the first floor is free on Thursday
     at two in the afternoon."                          140 ms   3.85 s of audio

So the cost is roughly the length of the sentence divided by 27, and the FIRST sentence is the only
one on the critical path. That is the second reason the brain is told to make its first sentence the
short one; the first is that it is spoken sooner.

The thread count is worth as much as the sentence length. Same laptop, same sentence, five runs:

    1 thread 449.6 ms | 2 threads 244.2 ms | 4 threads 184.6 ms | 8 threads 264.5 ms

The rise at eight is oversubscription on a laptop that does not have eight fast cores; what it
shows is that the count has to be measured on the machine that will be graded, not assumed.

THE VOICE IS A SETTING, NOT A PATH IN THE CODE
----------------------------------------------
The graded run resolves the voice out of the professor's shared cache with ``HF_HUB_OFFLINE=1``;
development resolves it wherever it was downloaded. So the voice arrives as ``--tts-voice`` or
``$TTS_VOICE``, and failing that it is searched for under ``$HF_HOME`` with bounded globs — bounded
because the shared cache is large and a full walk of it is a start-up cost measured in seconds.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator

from cpu_affinity import available_cpu_count
from interfaces import Tts

# Piper's en_US lessac medium voice, the one the assignment says is in the shared cache.
DEFAULT_VOICE_NAME = "en_US-lessac-medium"

VOICE_ENVIRONMENT_VARIABLE = "TTS_VOICE"
DEVICE_ENVIRONMENT_VARIABLE = "TTS_DEVICE"

# The rate of that voice. It is only a fallback: the rate actually used is read from the voice's own
# JSON config, because the turn loop hands it to the shim and a wrong one writes a WAV that plays at
# the wrong speed.
PIPER_SAMPLE_RATE_HZ = 22_050

AUTOMATIC = "auto"

# Where a voice is looked for when no path was given, and the shapes its folder takes. Every pattern
# has a fixed depth: the Hugging Face cache is tens of gigabytes and ``rglob`` over it at start-up
# costs seconds on a network filesystem.
VOICE_SEARCH_PATTERNS = (
    "{name}.onnx",
    "*/{name}.onnx",
    "voices/{name}.onnx",
    "*/*/{name}.onnx",
    "models--*piper*/snapshots/*/*/*/*/*/{name}.onnx",
    "models--*piper*/snapshots/*/*/*/*/{name}.onnx",
    "hub/models--*piper*/snapshots/*/*/*/*/*/{name}.onnx",
    "hub/models--*piper*/snapshots/*/*/*/*/{name}.onnx",
)

# What warmup says. Short, because its only job is to pay the one-time costs: the espeak-ng data
# load, the first ONNX inference and whatever memory arena that first inference allocates.
WARMUP_SENTENCE = "Ready."

# How many times warmup synthesises it. Two for the same reason the ASR decodes twice: the first
# inference after a session is created is not the same cost as the second.
WARMUP_RUNS = 2

SAMPLE_WIDTH_BYTES = 2

# espeak-ng keeps its data path in a fixed-size buffer and SILENTLY falls back to the path it was
# compiled with when the one it is given is longer. The fallback does not exist on any machine but
# the build server, so phonemisation dies with "Error processing file
# '/Users/runner/work/.../espeak-ng-data/phontab': No such file or directory" — which names a path
# nobody chose and does not mention length at all. Found by installing Piper under a long
# scratch path; it cost an hour, so the check is here rather than in a comment.
ESPEAK_PATH_LIMIT = 160


def _onnxruntime() -> Any:
    """Import ONNX Runtime at call time, so a run with ``--tts fake`` never pays for it."""
    import onnxruntime

    return onnxruntime


def cuda_is_available() -> bool:
    """Return whether this ONNX Runtime build can actually use a GPU."""
    try:
        return "CUDAExecutionProvider" in _onnxruntime().get_available_providers()

    except Exception:  # noqa: BLE001 - a missing or broken onnxruntime is a CPU answer, not a crash.
        return False


def search_roots() -> list[Path]:
    """Return the directories a voice is looked for in, most specific first."""
    roots = []

    for variable in ("PIPER_VOICE_DIR", "TTS_VOICE_DIR", "HF_HOME"):
        value = os.environ.get(variable)

        if value:
            roots.append(Path(value))

    roots.append(Path("voices"))
    roots.append(Path.home() / ".local/share/piper")
    roots.append(Path.home() / ".cache/huggingface")

    return [root for root in roots if root.exists()]


def find_voice(name: str = DEFAULT_VOICE_NAME) -> Path | None:
    """Return the path of the named voice, or ``None`` when it is nowhere to be found."""
    for root in search_roots():
        for pattern in VOICE_SEARCH_PATTERNS:
            matches = sorted(root.glob(pattern.format(name=name)))

            if matches:
                return matches[0]

    return None


def resolve_voice(setting: str | None) -> Path | None:
    """Return the voice this run should use, from a path, a directory, a name, or a search."""
    if not setting:
        return find_voice()

    given = Path(setting).expanduser()

    if given.is_file():
        return given

    if given.is_dir():
        matches = sorted(given.glob("*.onnx"))

        return matches[0] if matches else None

    # A bare voice name, such as "en_US-lessac-medium" or "en_US-ryan-high".
    return find_voice(given.name)


def config_path_of(voice_path: Path) -> Path:
    """Return the JSON config beside the model, which is where the sample rate lives."""
    with_suffix = voice_path.with_suffix(voice_path.suffix + ".json")

    if with_suffix.exists():
        return with_suffix

    return voice_path.with_suffix(".json")


def sample_rate_of(voice_path: Path | None) -> int:
    """Return the voice's own sample rate, or the lessac default when it cannot be read.

    This is read at construction, from JSON, WITHOUT loading the model: the turn loop prints the
    rate and hands it to the shim before warmup has run, and a WAV written at the wrong rate plays
    back at the wrong speed and pitch.
    """
    if voice_path is None:
        return PIPER_SAMPLE_RATE_HZ

    try:
        config = json.loads(config_path_of(voice_path).read_text())

        return int(config["audio"]["sample_rate"])

    except Exception:  # noqa: BLE001 - an unreadable config is a fallback, not a failed run.
        return PIPER_SAMPLE_RATE_HZ


def trim_leading_silence(
    pcm_bytes: bytes, sample_rate_hz: int, window_milliseconds: int = 10
) -> bytes:
    """Return ``pcm_bytes`` without the silence Piper puts in front of a sentence.

    Nothing is added and nothing is re-encoded: the return value is a suffix of the input, cut at a
    whole sample, at the start of the first window that reaches the shim's own audible threshold.
    """
    from voice_kit.audio import DEFAULT_ENERGY_THRESHOLD, root_mean_square

    step = max(1, sample_rate_hz * window_milliseconds // 1000) * SAMPLE_WIDTH_BYTES

    for start in range(0, len(pcm_bytes), step):
        if root_mean_square(pcm_bytes[start:start + step]) >= DEFAULT_ENERGY_THRESHOLD:
            return pcm_bytes[start:]

    return pcm_bytes


@dataclass(frozen=True)
class TtsSettings:
    """Everything about this stage that a run is allowed to change."""

    voice: str | None = field(
        default_factory=lambda: os.environ.get(VOICE_ENVIRONMENT_VARIABLE) or None
    )
    device: str = field(
        default_factory=lambda: os.environ.get(DEVICE_ENVIRONMENT_VARIABLE, AUTOMATIC)
    )

    # 0 means "ask cpu_affinity", which answers with the CPUs this job was given rather than the
    # ones the node has. See the head of this file.
    cpu_threads: int = 0

    # None means "whatever the voice's own config says". length_scale below 1 speaks faster, which
    # shortens synthesis as well as the speech, but it is a quality decision and so not a default.
    length_scale: float | None = None
    noise_scale: float | None = None
    noise_w_scale: float | None = None

    # Piper returns one chunk per sentence. Splitting it before it goes to the shim changes nothing
    # about when the audio exists, so the default is not to: 0 means "yield what Piper produced".
    #
    # It is not neutral for the STAMP, though, and the reason is worth stating. Piper puts about
    # 60 ms of silence in front of every sentence. Written as one chunk, that chunk's root mean
    # square is the whole sentence's, well over the threshold, so tts_first_audio is stamped when
    # the audio was produced. Written as 20 ms chunks, the first three are silent, the shim skips
    # them, and the stamp lands 60 ms later — on the same audio. Neither is wrong; they measure
    # different instants, and the report has to say which one it quotes. This agent quotes the
    # first, and ships the silence rather than removing it (see trim_leading_silence).
    chunk_milliseconds: int = 0

    # Off by default: the audio handed to the shim is exactly what Piper produced, including its
    # own lead-in. Turning this on drops the leading silence, which makes the captured WAV start
    # speaking at once. It never ADDS anything — the README's warning is about padding put in
    # front, and this is the opposite — but it does change the audio, so it is a decision somebody
    # makes on purpose and reports, not a default.
    trim_leading_silence: bool = False

    def resolved(self) -> "TtsSettings":
        """Return these settings with ``auto`` replaced by what this machine actually offers."""
        device = self.device

        if device == AUTOMATIC:
            device = "cuda" if cuda_is_available() else "cpu"

        return replace(self, device=device, cpu_threads=self.cpu_threads or available_cpu_count())

    def describe(self) -> str:
        """Return a one-line summary for the banner."""
        voice = resolve_voice(self.voice)

        return (
            f"{voice if voice else '(voice NOT FOUND)'} device={self.device} "
            f"cpu_threads={self.cpu_threads} chunk_milliseconds={self.chunk_milliseconds} "
            f"trim_leading_silence={self.trim_leading_silence}"
            + (f" length_scale={self.length_scale}" if self.length_scale else "")
        )


class PiperTts:
    """Synthesises one sentence at a time with a local Piper voice."""

    def __init__(self, settings: TtsSettings | None = None) -> None:
        self.settings = (settings or TtsSettings()).resolved()
        self.voice_path = resolve_voice(self.settings.voice)

        # Read before the model is loaded, because the turn loop asks for it before warmup.
        self.sample_rate_hz = sample_rate_of(self.voice_path)
        self.voice: Any | None = None
        self.synthesis_config: Any | None = None

    def describe(self) -> str:
        """Return what this stage is running, for the banner."""
        return self.settings.describe()

    # -- the contract ---------------------------------------------------------------------------

    def warmup(self) -> None:
        """Load the voice and synthesise one short sentence, so no load cost lands inside a turn."""
        if self.voice_path is None:
            raise FileNotFoundError(
                f"No Piper voice found. Looked for {DEFAULT_VOICE_NAME}.onnx under "
                f"{[str(root) for root in search_roots()] or 'nowhere — no search root exists'}. "
                "Pass --tts-voice with the path to the .onnx file, or set $TTS_VOICE."
            )

        self._check_espeak_data_path()
        self.voice = self._load_voice()
        self.synthesis_config = self._build_synthesis_config()

        rate_from_model = int(self.voice.config.sample_rate)

        if rate_from_model != self.sample_rate_hz:
            # The shim was already told the rate read from the JSON. If the model disagrees, the
            # captured WAV would be written at the wrong rate, so this is a failure, not a notice.
            raise ValueError(
                f"{self.voice_path} says {rate_from_model} Hz but its config said "
                f"{self.sample_rate_hz} Hz, and the shim has already been given the config's rate."
            )

        for _ in range(WARMUP_RUNS):
            for _chunk in self.stream(WARMUP_SENTENCE):
                pass

    @staticmethod
    def _check_espeak_data_path() -> None:
        """Fail early, and legibly, when espeak-ng's data path is too long for espeak-ng."""
        from piper.phonemize_espeak import ESPEAK_DATA_DIR

        if len(str(ESPEAK_DATA_DIR)) > ESPEAK_PATH_LIMIT:
            raise RuntimeError(
                f"Piper is installed at {ESPEAK_DATA_DIR}, which is "
                f"{len(str(ESPEAK_DATA_DIR))} characters long. espeak-ng truncates paths at about "
                f"{ESPEAK_PATH_LIMIT} and then looks for its data where it was BUILT, which does "
                "not exist here, so synthesis would fail with a confusing error about a phontab "
                "file. Install Piper somewhere with a shorter path."
            )

    def stream(self, sentence: str) -> Iterator[bytes]:
        """Yield ``sentence`` as 16-bit mono PCM chunks at :attr:`sample_rate_hz`."""
        if self.voice is None:
            raise RuntimeError("PiperTts.stream was called before warmup loaded the voice.")

        text = sentence.strip()

        if not text:
            return

        first = True

        for audio in self.voice.synthesize(text, syn_config=self.synthesis_config):
            pcm_bytes = audio.audio_int16_bytes

            if self.settings.trim_leading_silence and first:
                pcm_bytes = trim_leading_silence(pcm_bytes, self.sample_rate_hz)

            first = False

            if not pcm_bytes:
                continue

            # Yielded as it comes. Nothing is prepended and nothing is held back: the first bytes
            # Piper produced are the first bytes the shim sees, and that is the stamp.
            if self.settings.chunk_milliseconds <= 0:
                yield pcm_bytes

                continue

            step = self._chunk_byte_count()

            for start in range(0, len(pcm_bytes), step):
                yield pcm_bytes[start:start + step]

    # -- loading --------------------------------------------------------------------------------

    def _chunk_byte_count(self) -> int:
        """Return how many bytes one configured chunk holds, as a whole number of samples."""
        samples = max(1, self.sample_rate_hz * self.settings.chunk_milliseconds // 1000)

        return samples * SAMPLE_WIDTH_BYTES

    def _session_options(self) -> Any:
        """Return ONNX Runtime options whose thread pool is the size of this JOB, not this node."""
        onnxruntime = _onnxruntime()
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = self.settings.cpu_threads

        # One sentence is synthesised at a time, so there is nothing for a second op-level thread
        # to overlap with; it would only take a core away from the one that is working.
        options.inter_op_num_threads = 1
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        # ONNX Runtime's intra-op workers spin-wait after each op by default. With the pool the
        # size of the job, those spinning workers hold every core the job has, and the shim's feed
        # thread (which still delivers the trailing silence of the recording while this stage runs)
        # wakes late: PACE jobs 7162952 and 7163145 showed max_feed_lag up to 14.8 ms, against
        # 0.1 ms in the ASR-only job 7124980. Workers that sleep when idle give the core back.
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")

        return options

    def _load_voice(self) -> Any:
        """Return the loaded voice, with the session built here so its thread count is ours."""
        from piper import PiperVoice
        from piper.config import PiperConfig

        onnxruntime = _onnxruntime()
        providers = (
            [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"})]
            if self.settings.device == "cuda"
            else ["CPUExecutionProvider"]
        )

        try:
            config = json.loads(config_path_of(self.voice_path).read_text())
            session = onnxruntime.InferenceSession(
                str(self.voice_path),
                sess_options=self._session_options(),
                providers=providers,
            )

            return PiperVoice(config=PiperConfig.from_dict(config), session=session)

        except Exception as error:  # noqa: BLE001
            # Piper's own loader works, but it builds the session with a default SessionOptions(),
            # which sizes the thread pool from the node. Saying so is the point of this branch: a
            # silent fallback would make the stage timings on PACE-ICE inexplicable.
            print(
                f"tts: building the ONNX session directly failed ({type(error).__name__}: {error}); "
                "falling back to PiperVoice.load, whose THREAD COUNT COMES FROM THE MACHINE, "
                "not from this job. Stage timings from this run are not comparable.",
                file=sys.stderr,
                flush=True,
            )

            return PiperVoice.load(
                str(self.voice_path), use_cuda=self.settings.device == "cuda"
            )

    def _build_synthesis_config(self) -> Any:
        """Return the synthesis settings, or Piper's own defaults when nothing was overridden."""
        from piper.config import SynthesisConfig

        return SynthesisConfig(
            length_scale=self.settings.length_scale,
            noise_scale=self.settings.noise_scale,
            noise_w_scale=self.settings.noise_w_scale,
        )


def build(settings: TtsSettings | None = None) -> Tts:
    """Return the real speech synthesis stage."""
    return PiperTts(settings)
