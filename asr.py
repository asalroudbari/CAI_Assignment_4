"""The real speech recognition stage, on faster-whisper. Owned by Partner A.

DECODE STRATEGY: WHOLE UTTERANCE, AFTER SPEECH ENDS.
-----------------------------------------------------
The turn loop hands this stage every byte the shim fed, from the start of the recording to the end
of user speech, and the whole buffer is decoded in one pass at that moment. Nothing is decoded while
the audio is still arriving.

The alternative is streaming: re-decode a growing prefix of the buffer every few hundred
milliseconds, so that when speech ends only the tail is left to do. It is not free, and the costs
are not obvious:

* faster-whisper has no incremental decoder. "Streaming" means re-running the encoder and decoder
  over a longer prefix again and again, so the CPU work over one utterance is several times the work
  of one whole-utterance pass.
* That work lands WHILE the shim's feed thread is pacing the recording at real time. The feed thread
  must reacquire the interpreter lock every 20 ms to deliver its next chunk, so the extra work shows
  up in ``extras.max_feed_lag_milliseconds`` — the one number that proves the agent is not distorting
  its own input.
* It only pays when the final pass is long enough that overlapping it with speech saves more than the
  repeated passes cost.

THE MEASUREMENT HAS BEEN MADE, AND IT KEEPS WHOLE-UTTERANCE DECODE.

On one H100 of PACE-ICE (job 7090009, 36 turns, Systran/faster-whisper-small.en, device=cuda,
compute_type=float16, greedy), the ``asr`` stage of a whole session was:

    p50 40.1 ms, p95 62.0 ms          (benchmark on one 3.76 s utterance: p50 29.1 ms, 129x real time)

Beam 5 was tried against this and rejected on the same evidence: it moved the p50 by 5 ms, the p95
from 62.0 ms to 114.0 ms and the worst turn to 440.3 ms, which also failed the shim's feed-lag
check. See ``beam_size`` below.

The budget that would have forced streaming was 400 ms: the assignment pays full credit at a p50
time to first audio of 1500 ms, and the LLM's first token plus Piper's first chunk need the rest.
Whole-utterance decode comes in an order of magnitude under that, so streaming would add the
repeated-prefix cost and the feed-thread contention described above to buy back at most 40 ms of a
1500 ms budget. It was not built, and this is why.

The same decode on the CPUs of that node, for the case where no GPU is available:

    1 thread 2387.8 ms | 2 threads 1381.6 ms | 4 threads 889.4 ms | 8 threads 587.8 ms

which is where streaming would start to be worth arguing about again, and is the reason the thread
count is taken from the job rather than from the node. The chosen strategy is stated here, in one
place, because the report has to say which one is running.

Everything that could change that measurement is configuration, not a constant in the code: the
model, the device, the compute type, the thread count and the decode options all arrive through
:class:`AsrSettings` (see ``run_agent.py --help``), because the graded run happens on an H100 node
and the development run happens on a laptop, and the same source has to be honest on both.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any

from cpu_affinity import available_cpu_count
from interfaces import Asr

# The repository id in the professor's shared cache. It is resolved offline through HF_HOME with
# HF_HUB_OFFLINE=1, and it is a default rather than a constant: --asr-model takes a different id or
# a path to a directory of CTranslate2 weights.
DEFAULT_MODEL = "Systran/faster-whisper-small.en"

MODEL_ENVIRONMENT_VARIABLE = "ASR_MODEL"
DEVICE_ENVIRONMENT_VARIABLE = "ASR_DEVICE"
COMPUTE_TYPE_ENVIRONMENT_VARIABLE = "ASR_COMPUTE_TYPE"

# The rate the shim feeds and Whisper expects. Both are fixed at 16 kHz, so this is not a setting.
SAMPLE_RATE_HZ = 16_000
SAMPLE_WIDTH_BYTES = 2
FULL_SCALE = 32_768.0

# "auto" means "work it out at load time": the laptop has no CUDA and the H100 node does, and a run
# must not need a different command line on each.
AUTOMATIC = "auto"

# float16 on an H100 is the compute type the GPU has hardware for. int8 on CPU is roughly twice as
# fast as float32 on this model with no accuracy loss worth the latency; both are overridable,
# because "what is fastest here" is a measurement, not a belief.
DEFAULT_COMPUTE_TYPE_BY_DEVICE = {"cuda": "float16", "cpu": "int8"}

# What a turn with no recognisable speech reports. It must not be empty: interfaces.Asr says an empty
# transcript makes check_trace.py reject the whole turn, so a stage that heard nothing says so.
NOTHING_HEARD = "(no speech detected)"

# How much audio warmup decodes. Whisper pads every window to 30 seconds, so the encoder cost is the
# same whatever this is; what it buys is a decoder that has already generated tokens before the first
# timed turn asks it to.
WARMUP_SECONDS = 5.0
WARMUP_NOISE_AMPLITUDE = 600
WARMUP_SEED = 20_241_118


def _numpy() -> Any:
    """Import numpy at call time, so ``--asr fake`` never pays for it."""
    import numpy

    return numpy


def cuda_device_count() -> int:
    """Return how many CUDA devices CTranslate2 can see; 0 when it is a CPU-only build or box."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count()

    except Exception:  # noqa: BLE001 - a CPU-only wheel raises rather than returning 0.
        return 0


@dataclass(frozen=True)
class AsrSettings:
    """Everything about this stage that a run is allowed to change.

    The defaults come from the environment where there is a sensible variable for it, so the sbatch
    script can set the model and the device without the command line growing, and the command line
    still wins over both.
    """

    model: str = field(
        default_factory=lambda: os.environ.get(MODEL_ENVIRONMENT_VARIABLE, DEFAULT_MODEL)
    )
    device: str = field(
        default_factory=lambda: os.environ.get(DEVICE_ENVIRONMENT_VARIABLE, AUTOMATIC)
    )
    compute_type: str = field(
        default_factory=lambda: os.environ.get(COMPUTE_TYPE_ENVIRONMENT_VARIABLE, AUTOMATIC)
    )

    # 0 means "ask cpu_affinity", which means the CPUs this job was given rather than the node's.
    # Ignored on the GPU path, where the decode does not run on the CPU at all.
    cpu_threads: int = 0

    # One utterance is decoded at a time, so a second worker would be a second copy of the model
    # splitting the same cores and the same memory bandwidth with nothing to do.
    num_workers: int = 1

    # Greedy, and this was measured twice because the first measurement said the opposite.
    #
    # Decoding ONE utterance 20 times said beam 5 costs 3.8 ms (p50 29.1 ms against 32.9 ms) — too
    # little to care about. Running all 36 utterances said something else entirely:
    #
    #     greedy (job 7090009):  asr p50 40.1 ms   p95  62.0 ms   max  72.1 ms   checker PASS
    #     beam 5 (job 7090291):  asr p50 45.1 ms   p95 114.0 ms   max 440.3 ms   checker FAIL
    #
    # The median moved by the 5 ms the benchmark predicted. The TAIL did not: beam search keeps
    # five hypotheses alive until they all finish, so on an utterance the model is unsure about it
    # generates for far longer, and two of the 36 turns blew out to 440 ms and 201 ms. The 440 ms
    # one also held the interpreter lock long enough to push the shim's feed lag to 20.4 ms and
    # fail check_trace.py, so the beam did not merely cost latency, it corrupted the input timing.
    #
    # The accuracy argument for the beam was never strong enough to pay for that: 7.59% against
    # 8.13% pooled word error rate is two words in 369, inside the noise of a set this size.
    #
    # THE LESSON, because it is worth more than the setting: a benchmark that decodes the same
    # utterance again and again measures the decoder's variance ON THAT UTTERANCE. It cannot see
    # the across-utterance tail, which is exactly where beam search spends its time. Percentiles
    # from --benchmark-asr are for comparing configurations; percentiles for the report come from
    # a whole session.
    beam_size: int = 1

    # A single temperature, not faster-whisper's ladder of six. The ladder re-decodes the whole
    # window when a heuristic says the result looks bad, which is a 2x-6x outlier exactly in the p95
    # the report has to quote.
    temperature: float = 0.0

    # Each turn is decoded on its own. Carrying the previous transcript in as a prompt costs tokens
    # on every step, and the dialogue state belongs to the brain, not to the recogniser.
    condition_on_previous_text: bool = False

    # Segment timestamps are tokens the decoder has to emit, and nothing downstream reads them.
    without_timestamps: bool = True

    # The shim already cut the buffer at the end of speech, so Silero VAD would be a second ONNX
    # model loaded and run to find a boundary that is already known.
    vad_filter: bool = False

    # small.en is English-only, so this skips the language-detection pass rather than choosing a
    # language. A multilingual model would make this a real setting.
    language: str = "en"

    # How many times warmup decodes the dummy utterance. Two rather than one because the first
    # decode after a load is measurably slower — CUDA still selects kernels and allocates workspace
    # on the first call through a shape — and warmup is not timed, so the second decode costs the
    # trace nothing.
    #
    # KNOWN LIMIT, measured: the dummy is noise, and noise decodes to a single token ("you"), while
    # a real utterance decodes to ten or more. So this warms the model load, the feature extractor
    # and the encoder properly, but only the first step of the generation loop. It is enough in
    # practice — the first timed turn of a session came in at 114.5% of the median (greedy) and
    # 102.2% (beam 5), both well inside the 20% the warmup is there to buy — but the first decode
    # in the FIRST GPU process of a job still runs long (92 ms against a 31 ms median), so this is
    # not fully paid off. Warming on a real utterance would close it.
    warmup_decodes: int = 2

    def resolved(self) -> "AsrSettings":
        """Return these settings with every ``auto`` replaced by what this machine actually offers."""
        device = self.device

        if device == AUTOMATIC:
            device = "cuda" if cuda_device_count() > 0 else "cpu"

        compute_type = self.compute_type

        if compute_type == AUTOMATIC:
            compute_type = DEFAULT_COMPUTE_TYPE_BY_DEVICE.get(device, "default")

        cpu_threads = self.cpu_threads or available_cpu_count()

        return replace(
            self, device=device, compute_type=compute_type, cpu_threads=cpu_threads
        )

    def describe(self) -> str:
        """Return a one-line summary for the banner and for the head of a benchmark report."""
        return (
            f"{self.model} device={self.device} compute_type={self.compute_type} "
            f"cpu_threads={self.cpu_threads} num_workers={self.num_workers} "
            f"beam_size={self.beam_size} warmup_decodes={self.warmup_decodes} "
            f"vad_filter={self.vad_filter} "
            f"condition_on_previous_text={self.condition_on_previous_text}"
        )


def pcm_to_float32(pcm_bytes: bytes) -> Any:
    """Return 16 kHz mono 16-bit little-endian PCM as the float32 array faster-whisper wants.

    Handing it a numpy array rather than a path or a file-like object keeps the audio in memory:
    writing a temporary WAV and letting PyAV decode it again would add a disk round trip and an
    audio decoder to a stage whose whole job is to be fast.
    """
    numpy = _numpy()

    # A half sample cannot be decoded; truncating is better than raising on a buffer that is one
    # byte short because the feed was cut between the two halves of a sample.
    usable = len(pcm_bytes) - (len(pcm_bytes) % SAMPLE_WIDTH_BYTES)
    samples = numpy.frombuffer(pcm_bytes[:usable], dtype=numpy.int16)

    return samples.astype(numpy.float32) / FULL_SCALE


def warmup_audio_bytes(
    seconds: float = WARMUP_SECONDS, amplitude: int = WARMUP_NOISE_AMPLITUDE
) -> bytes:
    """Return a deterministic buffer to warm the decoder on, as 16-bit little-endian PCM.

    Noise rather than silence: a silent buffer is cut short by Whisper's own no-speech check, so the
    decoder would barely run and the first timed turn would still be paying for its first real
    decode. The seed is fixed so that two warmups cost the same.
    """
    numpy = _numpy()
    generator = numpy.random.default_rng(WARMUP_SEED)
    sample_count = int(seconds * SAMPLE_RATE_HZ)
    samples = generator.standard_normal(sample_count) * amplitude

    return samples.astype(numpy.int16).tobytes()


class FasterWhisperAsr:
    """Transcribes the replayed utterance with a local faster-whisper model, in one pass."""

    def __init__(self, settings: AsrSettings | None = None) -> None:
        self.settings = (settings or AsrSettings()).resolved()
        self.model: Any | None = None
        self.warmup_transcript: str | None = None

    def warmup(self) -> None:
        """Load the model and decode one dummy utterance, so no start-up cost lands inside a turn.

        Both halves matter. Loading the weights is seconds; but the first decode is also slower than
        every later one, because CTranslate2 builds its thread pool and allocates its workspace on
        the first call, the GPU path compiles and caches its kernels, and the log-mel filterbank is
        built once. Loading without decoding would move a visible cost into the first timed turn, and
        the first turn's asr stage is checked against the median to prove it did not.
        """
        self.model = self._load_model()

        # Built once and decoded repeatedly: the point is to repeat the DECODE, and regenerating the
        # noise each time would only measure numpy.
        audio = warmup_audio_bytes()

        for _ in range(max(1, self.settings.warmup_decodes)):
            self.warmup_transcript = self.transcribe(audio)

    def transcribe(self, pcm_bytes: bytes) -> str:
        """Return the final transcript of the whole utterance, as a non-empty string.

        The turn loop stamps ``asr_final`` the moment this returns, so everything here is counted as
        ASR time. faster-whisper's ``transcribe`` returns a GENERATOR of segments and does the real
        decoding only as it is consumed, so the join below is where the time actually goes — which
        is what makes the stamp honest.
        """
        if self.model is None:
            # Loading here would silently charge seconds of model load to one turn's asr stage.
            raise RuntimeError(
                "FasterWhisperAsr.transcribe was called before warmup(). The model is loaded in "
                "warmup() on purpose, so that load time can never land inside a timed turn."
            )

        if not pcm_bytes:
            return NOTHING_HEARD

        segments, _info = self.model.transcribe(
            pcm_to_float32(pcm_bytes),
            language=self.settings.language,
            beam_size=self.settings.beam_size,
            temperature=self.settings.temperature,
            condition_on_previous_text=self.settings.condition_on_previous_text,
            without_timestamps=self.settings.without_timestamps,
            vad_filter=self.settings.vad_filter,
        )

        transcript = " ".join(segment.text.strip() for segment in segments).strip()

        return transcript or NOTHING_HEARD

    def _load_model(self) -> Any:
        """Build the CTranslate2 model with this run's device, compute type and thread count.

        faster-whisper is imported here rather than at the top of the module so that a run with
        ``--asr fake`` never loads CTranslate2 at all, and so that this import happens after
        ``run_agent.py`` has set the thread environment.
        """
        from faster_whisper import WhisperModel

        settings = self.settings

        return WhisperModel(
            settings.model,
            device=settings.device,
            compute_type=settings.compute_type,
            # Passed explicitly: CTranslate2 sizes its intra-op pool from this argument, and when it
            # is left at 0 it asks the MACHINE how many cores it has, not this job.
            cpu_threads=settings.cpu_threads,
            num_workers=settings.num_workers,
        )


def build(settings: AsrSettings | None = None) -> Asr:
    """Return the real speech recognition stage."""
    return FasterWhisperAsr(settings)
