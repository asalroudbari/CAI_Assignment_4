"""Replays recordings through the agent, times every stage and writes the trace.

    PYTHONPATH=. python3 run_agent.py --recordings recordings --output-dir run_output

Which code backs each stage is always explicit. With no flags the agent uses the real modules
(``asr.py``, ``brain.py``, ``tts.py``); ``--fake-asr``, ``--fake-brain``, ``--fake-tts`` and
``--fake-all`` swap in the stand-ins from ``fakes.py``. Nothing is ever substituted silently: the
banner names the class behind each stage and every turn record carries
``extras.stage_implementations``.

To tune the ASR stage without replaying a whole session:

    PYTHONPATH=. python3 run_agent.py --benchmark-asr --benchmark-runs 20 --asr-cpu-threads 4

Exit code is 0 only when every turn ran to the end.
"""

from __future__ import annotations

# ------------------------------------------------------------------------------------------------
# THREAD ENVIRONMENT. THIS BLOCK RUNS BEFORE EVERY OTHER IMPORT, AND THAT ORDER IS THE WHOLE POINT.
#
# WHY HERE: OpenMP and MKL read OMP_NUM_THREADS and MKL_NUM_THREADS once, when the native library is
# loaded and builds its thread pool. CTranslate2 (faster-whisper), ONNX Runtime (Piper) and PyTorch
# all sit on one of them, so by the time ``import asr`` has pulled CTranslate2 in, the pool already
# exists and assigning to os.environ afterwards changes nothing at all. Above the imports is the
# only placement that has any effect; below them it is a comment that looks like code.
#
# WHY sched_getaffinity AND NOT cpu_count: on PACE-ICE these are different numbers. os.cpu_count()
# reports the CPUs the NODE has; len(os.sched_getaffinity(0)) reports the CPUs this process is
# allowed to run on, which under Slurm's cgroup is exactly --cpus-per-task. Sizing the pool from
# cpu_count() on a 64-core node running an 8-core job starts 64 threads for 8 cores: they preempt
# each other, and the ASR stage becomes both slower and far less predictable in the p95 than a
# smaller pool would be. cpu_affinity.py holds that call, and the macOS fallback for development.
#
# WHY sys.argv AND NOT argparse: argparse cannot run this early. main() starts only after every
# import has already happened, which is already too late for the variables to matter.
# ------------------------------------------------------------------------------------------------
import cpu_affinity

REQUESTED_CPU_COUNT = cpu_affinity.requested_cpu_count_from_argv()
CPU_COUNT = REQUESTED_CPU_COUNT or cpu_affinity.available_cpu_count()

# Where the number came from, kept separately from the number itself. The banner prints this, and a
# banner that says "from len(os.sched_getaffinity(0))" when the value actually came from
# --cpu-count is worse than no banner: it is a provenance claim, and the report quotes it.
CPU_COUNT_SOURCE = (
    "--cpu-count" if REQUESTED_CPU_COUNT else cpu_affinity.cpu_count_source()
)

THREAD_ENVIRONMENT = cpu_affinity.apply_thread_environment(CPU_COUNT)

import argparse
import math
import json
import queue
import re
import sys
import time
from pathlib import Path

from voice_kit import (
    ASR_FINAL,
    DEFAULT_CHUNK_MILLISECONDS,
    END_OF_USER_SPEECH,
    LLM_COMPLETE,
    LLM_FIRST_TOKEN,
    TTS_FIRST_AUDIO,
    ReplaySession,
    Turn,
    percentile,
)
from voice_kit.audio import SAMPLE_RATE_HZ, SAMPLE_WIDTH_BYTES, chunk_byte_count
from voice_kit.source import Recording, load_recording

import asr as real_asr
import brain as real_brain
import fakes
import llm_client
import tts as real_tts
from interfaces import Asr, Brain, Tts

NANOSECONDS_PER_MILLISECOND = 1_000_000

# How long a thread may hold the interpreter lock before it is asked to yield. The default is 5 ms,
# which is a problem here: the shim's feed thread sleeps until a deadline 20 ms away and then has to
# reacquire the lock to deliver the chunk, so main-thread work can hold it off for a whole switch
# interval and that delay lands in extras.max_feed_lag_milliseconds. Measured on this machine, a busy
# main thread takes the median feed lag from 3.4 ms to 9.0 ms at the default, and back to 4.5 ms at
# 0.5 ms. Shorter intervals cost a little throughput in exchange for a feed thread that wakes on time.
# Reproduce it with: PYTHONPATH=. python3 tools/measure_feed_lag.py
# (re-measured 2026-11-15: idle 3.41 ms; busy 8.99 ms at the default, 4.40 ms at 0.5 ms).
FEED_THREAD_SWITCH_INTERVAL_SECONDS = 0.0005


# A recording set is several short conversations. ``alex_d01_t02.wav`` is dialogue 1, turn 2, and
# sorted file names therefore replay each dialogue in order. A sidecar ``dialogue_id`` overrides it.
DIALOGUE_IN_FILE_NAME = re.compile(r"(?:^|_)d(\d+)(?:_|$)", re.IGNORECASE)

# What a recording with no dialogue marker in its name is treated as belonging to.
UNMARKED_DIALOGUE = "unmarked"


class ForcedTurnFailure(RuntimeError):
    """Raised by --fail-turn, to prove that a failed turn still writes a record."""


class EmptyReplyError(RuntimeError):
    """Raised when the brain yields no sentence at all, which would reject the turn silently."""


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Return the settings of one run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    parser.add_argument(
        "--recordings",
        type=Path,
        default=Path("recordings"),
        help="Directory of 16 kHz mono 16-bit WAV files, replayed in file name order.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("run_output"),
        help="Where the trace and the captured TTS audio are written.",
    )
    parser.add_argument(
        "--cpu-count",
        type=int,
        default=None,
        help="How many CPUs this run may use. The default is len(os.sched_getaffinity(0)), the CPUs "
        "the job was actually given. It sets OMP_NUM_THREADS and MKL_NUM_THREADS, and the ASR "
        "thread count unless --asr-cpu-threads overrides it. Read from sys.argv before the imports "
        "(see the top of this file), so argparse only documents and validates it here.",
    )

    recognition = parser.add_argument_group(
        "speech recognition settings",
        "The model, the device and the decode strategy of the ASR stage. None of these is a "
        "constant in the code: the graded run is an H100 node and development is a laptop.",
    )
    recognition.add_argument(
        "--asr-model",
        default=None,
        help="Hugging Face id or local directory of the faster-whisper model "
        f"(default: ${real_asr.MODEL_ENVIRONMENT_VARIABLE} or {real_asr.DEFAULT_MODEL}).",
    )
    recognition.add_argument(
        "--asr-device",
        choices=("auto", "cuda", "cpu"),
        default=None,
        help="Where the model runs; auto picks cuda when CTranslate2 can see a GPU (default: auto).",
    )
    recognition.add_argument(
        "--asr-compute-type",
        default=None,
        help="CTranslate2 compute type, e.g. float16, int8_float16, int8, float32 "
        "(default: float16 on cuda, int8 on cpu).",
    )
    recognition.add_argument(
        "--asr-cpu-threads",
        type=int,
        default=None,
        help="Intra-op threads for the CPU decode (default: --cpu-count).",
    )
    recognition.add_argument(
        "--asr-num-workers",
        type=int,
        default=None,
        help="Parallel model replicas; one utterance is decoded at a time, so this stays 1.",
    )
    recognition.add_argument(
        "--asr-beam-size",
        type=int,
        default=None,
        help="Beam width of the decode; 1 is greedy (default: 1, against Whisper's own 5).",
    )
    recognition.add_argument(
        "--asr-warmup-decodes",
        type=int,
        default=None,
        help="How many times warmup decodes its dummy utterance (default: 2). The second one is "
        "not waste: on CUDA the first decode after a load still pays for kernel selection.",
    )
    recognition.add_argument(
        "--asr-vad-filter",
        action="store_true",
        help="Run Silero VAD before decoding. Off by default: the shim already cut the buffer at "
        "the end of speech, so this only adds a second model to the measured stage.",
    )

    dialogue = parser.add_argument_group(
        "dialogue model settings",
        "The model, the server and how much of the knowledge base each turn carries. The README "
        "requires the model name and the server URL to be settings rather than constants, and the "
        "knowledge-base policy is the one prompt decision that shows up directly in llm_first_token.",
    )
    dialogue.add_argument(
        "--llm-model",
        default=None,
        help="Model name the server is serving "
        f"(default: ${real_brain.MODEL_ENVIRONMENT_VARIABLE} or {real_brain.DEFAULT_MODEL}).",
    )
    dialogue.add_argument(
        "--llm-base-url",
        default=None,
        help="Base URL of the OpenAI-compatible server, including /v1 "
        f"(default: ${llm_client.BASE_URL_ENVIRONMENT_VARIABLE} or {llm_client.DEFAULT_BASE_URL}).",
    )
    dialogue.add_argument(
        "--llm-temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: 0.0, so two runs of one recording agree).",
    )
    dialogue.add_argument(
        "--llm-max-tokens",
        type=int,
        default=None,
        help="Cap on the whole answer (default: 120, about two spoken sentences).",
    )
    dialogue.add_argument(
        "--llm-seed",
        type=int,
        default=None,
        help="Seed sent with every request, so a greedy run is reproducible (default: 20241118).",
    )
    dialogue.add_argument(
        "--llm-kb-mode",
        choices=real_brain.KB_MODES,
        default=None,
        help="How much of the knowledge base each prompt carries: relevant (default), full, none.",
    )
    dialogue.add_argument(
        "--llm-kb-facts",
        type=int,
        default=None,
        help="How many facts --llm-kb-mode relevant retrieves (default: 4).",
    )
    dialogue.add_argument(
        "--llm-history-turns",
        type=int,
        default=None,
        help="How many earlier turns of the dialogue go into the prompt (default: 6).",
    )
    dialogue.add_argument(
        "--llm-reasoning-effort",
        default=None,
        help="For a model that thinks before it answers (gpt-oss-20b): low, medium or high. Those "
        "tokens are never spoken, so they are pure latency; say in the report if you set this.",
    )

    synthesis = parser.add_argument_group(
        "speech synthesis settings",
        "The voice, the device and the thread count of the TTS stage. ONNX Runtime sizes its "
        "thread pool from the machine unless it is told otherwise, which on a 64-CPU node running "
        "an 8-CPU job is the trap the README warns about.",
    )
    synthesis.add_argument(
        "--tts-voice",
        default=None,
        help="Path to the Piper .onnx voice, a directory holding one, or a voice name "
        f"(default: ${real_tts.VOICE_ENVIRONMENT_VARIABLE}, else a search for "
        f"{real_tts.DEFAULT_VOICE_NAME} under $HF_HOME).",
    )
    synthesis.add_argument(
        "--tts-device",
        choices=("auto", "cuda", "cpu"),
        default=None,
        help="Where the voice runs; auto picks cuda when ONNX Runtime has a GPU provider "
        "(default: auto).",
    )
    synthesis.add_argument(
        "--tts-cpu-threads",
        type=int,
        default=None,
        help="Intra-op threads for synthesis (default: --cpu-count).",
    )
    synthesis.add_argument(
        "--tts-length-scale",
        type=float,
        default=None,
        help="Speaking rate; below 1 is faster (default: the voice's own setting).",
    )
    synthesis.add_argument(
        "--tts-chunk-milliseconds",
        type=int,
        default=None,
        help="Split each sentence of audio into chunks this long before writing it to the shim. "
        "0, the default, writes what Piper produced, which is one chunk per sentence.",
    )

    synthesis.add_argument(
        "--tts-trim-leading-silence",
        action="store_true",
        help="Drop the roughly 60 ms of silence Piper puts in front of each sentence, so the "
        "captured audio starts speaking at once. Off by default: what Piper produced is what the "
        "shim is given. Nothing is ever added in front either way.",
    )

    benchmark = parser.add_argument_group(
        "asr benchmark",
        "Decode one recording again and again and print the percentiles, with no session, no "
        "brain and no TTS, so a thread count or a decode setting can be measured in seconds.",
    )
    benchmark.add_argument(
        "--benchmark-asr",
        action="store_true",
        help="Run the ASR benchmark instead of a session.",
    )
    benchmark.add_argument(
        "--benchmark-wav",
        type=Path,
        default=None,
        help="Which recording to decode (default: the first one in --recordings).",
    )
    benchmark.add_argument(
        "--benchmark-runs",
        type=int,
        default=10,
        help="How many times to decode it, after warmup (default: 10).",
    )

    for stage in ("asr", "brain", "tts"):
        parser.add_argument(
            f"--{stage}",
            choices=("real", "fake"),
            default="real",
            help=f"Which {stage} implementation to use (default: real).",
        )
        parser.add_argument(
            f"--fake-{stage}",
            action="store_true",
            help=f"Shorthand for --{stage} fake.",
        )

    parser.add_argument(
        "--fake-all",
        action="store_true",
        help="Shorthand for --asr fake --brain fake --tts fake.",
    )
    parser.add_argument(
        "--fail-turn",
        type=int,
        default=None,
        metavar="N",
        help="Force turn N (1-based) to fail after end of speech, to prove the failure path.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Keep an existing trace instead of truncating it. The shim appends, so without this "
        "a second run into the same directory would double the turn count.",
    )

    arguments = parser.parse_args(argv)

    # The shorthand flags win over the defaults of the choice flags, and --fake-all wins over both.
    for stage in ("asr", "brain", "tts"):
        if arguments.fake_all or getattr(arguments, f"fake_{stage}"):
            setattr(arguments, stage, "fake")

    return arguments


def asr_settings_from(arguments: argparse.Namespace) -> real_asr.AsrSettings:
    """Return the ASR settings of this run: command line over environment over defaults.

    Only a flag the user actually passed becomes an override, so an unset flag leaves the field at
    whatever ``AsrSettings`` resolves from the environment rather than silently overwriting it.
    """
    overrides: dict[str, object] = {}

    for name, value in (
        ("model", arguments.asr_model),
        ("device", arguments.asr_device),
        ("compute_type", arguments.asr_compute_type),
        ("num_workers", arguments.asr_num_workers),
        ("beam_size", arguments.asr_beam_size),
        ("warmup_decodes", arguments.asr_warmup_decodes),
    ):
        if value is not None:
            overrides[name] = value

    if arguments.asr_vad_filter:
        overrides["vad_filter"] = True

    # --asr-cpu-threads is the knob for this stage alone; --cpu-count is the whole job's and covers
    # this stage too, so the sbatch script only has to pass one number. Neither given means
    # AsrSettings asks cpu_affinity itself, which is the same answer.
    cpu_threads = (
        arguments.asr_cpu_threads
        if arguments.asr_cpu_threads is not None
        else arguments.cpu_count
    )

    if cpu_threads is not None:
        overrides["cpu_threads"] = cpu_threads

    return real_asr.AsrSettings(**overrides)


def brain_settings_from(arguments: argparse.Namespace) -> real_brain.BrainSettings:
    """Return the dialogue model settings of this run: command line over environment over defaults.

    The same rule as the ASR: only a flag that was actually passed becomes an override, so an unset
    flag leaves the field at whatever the environment resolved it to.
    """
    overrides: dict[str, object] = {}

    for name, value in (
        ("model", arguments.llm_model),
        ("base_url", arguments.llm_base_url),
        ("temperature", arguments.llm_temperature),
        ("max_tokens", arguments.llm_max_tokens),
        ("seed", arguments.llm_seed),
        ("kb_mode", arguments.llm_kb_mode),
        ("kb_facts", arguments.llm_kb_facts),
        ("history_turns", arguments.llm_history_turns),
        ("reasoning_effort", arguments.llm_reasoning_effort),
    ):
        if value is not None:
            overrides[name] = value

    return real_brain.BrainSettings(**overrides)


def tts_settings_from(arguments: argparse.Namespace) -> real_tts.TtsSettings:
    """Return the speech synthesis settings of this run: command line over environment over defaults."""
    overrides: dict[str, object] = {}

    for name, value in (
        ("voice", arguments.tts_voice),
        ("device", arguments.tts_device),
        ("length_scale", arguments.tts_length_scale),
        ("chunk_milliseconds", arguments.tts_chunk_milliseconds),
    ):
        if value is not None:
            overrides[name] = value

    if arguments.tts_trim_leading_silence:
        overrides["trim_leading_silence"] = True

    # --tts-cpu-threads is this stage's own knob; --cpu-count is the job's and covers it too, so
    # the sbatch script only has to pass one number.
    cpu_threads = (
        arguments.tts_cpu_threads
        if arguments.tts_cpu_threads is not None
        else arguments.cpu_count
    )

    if cpu_threads is not None:
        overrides["cpu_threads"] = cpu_threads

    return real_tts.TtsSettings(**overrides)


def build_stages(arguments: argparse.Namespace) -> tuple[Asr, Brain, Tts]:
    """Return the three stages this run will use, real or fake as asked."""
    speech_recognition = (
        fakes.build_asr()
        if arguments.asr == "fake"
        else real_asr.build(asr_settings_from(arguments))
    )
    dialogue_manager = (
        fakes.build_brain()
        if arguments.brain == "fake"
        else real_brain.build(brain_settings_from(arguments))
    )
    speech_synthesis = (
        fakes.build_tts()
        if arguments.tts == "fake"
        else real_tts.build(tts_settings_from(arguments))
    )

    return speech_recognition, dialogue_manager, speech_synthesis


def stage_implementations(
    speech_recognition: Asr, dialogue_manager: Brain, speech_synthesis: Tts
) -> dict[str, str]:
    """Return the concrete class name behind each stage, for the banner and for every record."""
    return {
        "asr": type(speech_recognition).__name__,
        "brain": type(dialogue_manager).__name__,
        "tts": type(speech_synthesis).__name__,
    }


def print_thread_environment() -> None:
    """Print the thread count, where it came from, and what it changed in the environment.

    The before/after pair is the evidence for the report: it shows the variables the native thread
    pools read, as they were and as this process set them, rather than asserting a number.
    """
    print(f"cpu count         {CPU_COUNT} from {CPU_COUNT_SOURCE}")

    # Only when the fallback was actually what answered: with --cpu-count the platform never came
    # into it, so the warning would be about a call this run did not make.
    if REQUESTED_CPU_COUNT is None and not cpu_affinity.affinity_is_available():
        print(
            "                  WARNING: this platform has no sched_getaffinity, so that is the\n"
            "                  whole machine, not the CPUs a job was given. Development only;\n"
            "                  the PACE-ICE run takes the sched_getaffinity path."
        )

    for name, before, after in THREAD_ENVIRONMENT:
        print(f"{name:<18}{before if before is not None else '(unset)'} -> {after}")


def print_banner(
    arguments: argparse.Namespace,
    implementations: dict[str, str],
    recording_count: int,
    sink_sample_rate_hz: int,
) -> None:
    """Say out loud what is about to run, so no run is ever ambiguous about its own stages."""
    print("=" * 78)
    print(f"recordings        {recording_count} from {arguments.recordings}")
    print(f"output            {arguments.output_dir}")
    print(f"tts sample rate   {sink_sample_rate_hz} Hz")
    print_thread_environment()

    for stage in ("asr", "brain", "tts"):
        print(f"{stage:<18}{implementations[stage]} ({getattr(arguments, stage)})")

    if arguments.asr == "real":
        print(f"asr settings      {asr_settings_from(arguments).resolved().describe()}")

    if arguments.brain == "real":
        print(f"brain settings    {brain_settings_from(arguments).resolved().describe()}")

    if arguments.tts == "real":
        print(f"tts settings      {tts_settings_from(arguments).resolved().describe()}")

    if any(getattr(arguments, stage) == "fake" for stage in ("asr", "brain", "tts")):
        print("")
        print("*** FAKE STAGES ACTIVE — THESE NUMBERS ARE NOT A REAL MEASUREMENT ***")

    print("=" * 78)


def prepare_output_directory(output_directory: Path, append: bool) -> None:
    """Clear the previous run unless asked to append, because the shim opens the trace with "a"."""
    output_directory.mkdir(parents=True, exist_ok=True)

    if append:
        return

    trace_path = output_directory / "trace.jsonl"

    if trace_path.exists():
        trace_path.unlink()

    for wav_path in (output_directory / "tts").glob("turn_*.wav"):
        wav_path.unlink()


def validate_recordings(recording_paths: list[Path]) -> None:
    """Load every recording before the first timed turn, so a bad file cannot kill a run midway.

    ReplaySession.turns() is a generator that loads each recording as it yields it, so a WAV in the
    wrong format would raise inside the loop, after some turns had already been traced.
    """
    for wav_path in recording_paths:
        try:
            load_recording(wav_path)

        except Exception as error:  # noqa: BLE001
            raise SystemExit(
                f"{wav_path} cannot be replayed: {error}\n"
                "Every recording must be 16 kHz mono 16-bit PCM with audible speech in it. "
                "Run tools/check_recordings.py for a per-file report."
            ) from error


def dialogue_id_of(wav_path: Path) -> str:
    """Return which dialogue a recording belongs to, so the brain can be reset between dialogues."""
    sidecar_path = wav_path.with_suffix(".json")

    if sidecar_path.exists():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

        except ValueError:
            sidecar = {}

        if isinstance(sidecar, dict) and "dialogue_id" in sidecar:
            return str(sidecar["dialogue_id"])

    match = DIALOGUE_IN_FILE_NAME.search(wav_path.stem)

    return match.group(1) if match else UNMARKED_DIALOGUE


def warn_about_unmarked_dialogues(recording_paths: list[Path]) -> None:
    """Say so when no recording carries a dialogue marker, because then nothing is ever reset."""
    if any(dialogue_id_of(path) != UNMARKED_DIALOGUE for path in recording_paths):
        return

    print(
        "note: no recording name carries a dialogue marker, so all turns are treated as ONE\n"
        "      conversation and the brain is never reset. Name files <name>_d01_t01.wav, or put\n"
        '      {"dialogue_id": "01"} in the sidecar JSON, to mark where conversations end.\n',
        file=sys.stderr,
    )


def warm_up_stages(
    speech_recognition: Asr, dialogue_manager: Brain, speech_synthesis: Tts
) -> None:
    """Run every stage once before the first timed turn, so no start-up cost lands inside a turn."""
    print("warming up ...", flush=True)
    started_nanoseconds = time.monotonic_ns()

    for name, stage in (
        ("asr", speech_recognition),
        ("brain", dialogue_manager),
        ("tts", speech_synthesis),
    ):
        stage_started_nanoseconds = time.monotonic_ns()
        stage.warmup()
        elapsed = (time.monotonic_ns() - stage_started_nanoseconds) / NANOSECONDS_PER_MILLISECOND
        print(f"  {name:<6} warmed up in {elapsed:8.1f} ms")

    total = (time.monotonic_ns() - started_nanoseconds) / NANOSECONDS_PER_MILLISECOND
    print(f"warm up finished in {total:.1f} ms\n")


def stamp_once(turn: Turn, event: str) -> bool:
    """Stamp an event unless it is already stamped; return whether this call stamped it.

    Turn.stamp raises on a second stamp of the same event, so every path that could run twice — a
    retry, an error handler, a loop that yields more than once — goes through this.
    """
    if event in turn.events:
        return False

    turn.stamp(event)

    return True


def drain_audio(chunks: queue.SimpleQueue) -> bytes:
    """Return everything the feed thread has handed over so far, in order."""
    pcm_bytes = bytearray()

    while True:
        try:
            pcm_bytes += chunks.get_nowait()

        except queue.Empty:
            return bytes(pcm_bytes)


def run_turn(
    turn: Turn,
    turn_number: int,
    speech_recognition: Asr,
    dialogue_manager: Brain,
    speech_synthesis: Tts,
    implementations: dict[str, str],
    fail_turn: int | None,
    dialogue_id: str,
    starts_new_dialogue: bool,
) -> None:
    """Run one turn end to end; the record is written whether or not it succeeds."""
    # Set before the try block, so even a turn that fails on its first line carries it.
    turn.extras["stage_implementations"] = implementations
    turn.extras["dialogue_id"] = dialogue_id

    # Done before the recording starts feeding, so none of it lands inside the measured turn. The
    # first dialogue is reset too: warmup ran a whole reply through the brain and may have left
    # state behind.
    if starts_new_dialogue:
        dialogue_manager.reset()

    chunks: queue.SimpleQueue = queue.SimpleQueue()

    try:
        # on_chunk does nothing but hand the bytes over. ReplaySource calls it inline on the feed
        # thread, between deadlines that are 20 ms apart, and only announces end of speech after it
        # returns. Any work here would both show up in extras.max_feed_lag_milliseconds and delay the
        # start of ASR while end_of_user_speech stays backdated to before the chunk was delivered —
        # that is, it would quietly pad the measured ASR stage.
        turn.start(on_chunk=chunks.put)

        # Nothing may be stamped before this line: an event earlier than end_of_user_speech fails
        # the checker, and until this returns the shim has not stamped end_of_user_speech at all.
        turn.end_of_speech_reached.wait()

        if fail_turn is not None and fail_turn == turn_number:
            raise ForcedTurnFailure(
                f"--fail-turn {fail_turn}: failing this turn on purpose, after end of speech."
            )

        pcm_bytes = drain_audio(chunks)

        transcript = speech_recognition.transcribe(pcm_bytes)
        stamp_once(turn, ASR_FINAL)
        turn.extras["transcript"] = transcript

        spoken_sentences: list[str] = []

        for sentence in dialogue_manager.reply(transcript):
            sentence = sentence.strip()

            if not sentence:
                continue

            if not spoken_sentences:
                # The first sentence of the answer: this, not any earlier token, is the event.
                stamp_once(turn, LLM_FIRST_TOKEN)

            spoken_sentences.append(sentence)

            for chunk in speech_synthesis.stream(sentence):
                turn.sink.write(chunk)

            # Honest note: the next sentence is pulled from the brain only once this sentence has
            # been synthesised, so generation is serialised behind synthesis. That inflates the
            # llm_remaining stage, which is off the time-to-first-audio path (see STAGES in
            # voice_kit/trace.py), so it does not touch the headline number — but it is not free, and
            # the fix is to run the brain on its own thread feeding a sentence queue. Not done yet.

        stamp_once(turn, LLM_COMPLETE)

        if not spoken_sentences:
            raise EmptyReplyError(
                f"{type(dialogue_manager).__name__}.reply yielded no sentence, so there is nothing "
                "to speak and extras.reply would be empty."
            )

        turn.extras["reply"] = " ".join(spoken_sentences)

        # Required by the README only for a model that produces reasoning tokens before its answer.
        # The brain sets it to None for a model that does not, and then it is left out rather than
        # written as null: the checker reads the key, not its value.
        reasoning_nanoseconds = getattr(
            dialogue_manager, "first_reasoning_token_nanoseconds", None
        )

        if reasoning_nanoseconds is not None:
            turn.extras["first_reasoning_token_nanoseconds"] = reasoning_nanoseconds

    finally:
        # In a finally block on purpose: a turn that failed still has to write its record, so the
        # checker can reject it and we can see which turn it was.
        turn.finish()


def describe_turn(turn: Turn) -> str:
    """Return a one-line summary of what the turn cost, from whatever events it managed to stamp."""
    events = turn.events

    def delta(later: str, earlier: str) -> str:
        if later not in events or earlier not in events:
            return "    -   "

        return f"{(events[later] - events[earlier]) / NANOSECONDS_PER_MILLISECOND:8.1f}"

    return (
        f"asr {delta(ASR_FINAL, END_OF_USER_SPEECH)} ms | "
        f"llm_start {delta(LLM_FIRST_TOKEN, ASR_FINAL)} ms | "
        f"speech_start {delta(TTS_FIRST_AUDIO, LLM_FIRST_TOKEN)} ms | "
        f"ttfa {delta(TTS_FIRST_AUDIO, END_OF_USER_SPEECH)} ms"
    )


def shorten(text: object, limit: int = 48) -> str:
    """Return text clipped to one readable line."""
    if not isinstance(text, str):
        return "-"

    text = " ".join(text.split())

    return text if len(text) <= limit else text[: limit - 1] + "…"


def speech_bytes_of(recording: Recording) -> bytes:
    """Return exactly the bytes a turn's ASR stage would be handed for this recording.

    Not simply "the file up to speech_end_seconds". The shim feeds in whole 20 ms chunks and only
    announces the end of speech AFTER delivering the chunk the end falls in, so the buffer the turn
    loop drains runs to the end of that chunk. Rounding up to the same boundary here is what makes a
    benchmark number comparable with the asr stage in the trace instead of merely similar to it.
    """
    bytes_per_second = SAMPLE_RATE_HZ * SAMPLE_WIDTH_BYTES
    step = chunk_byte_count(DEFAULT_CHUNK_MILLISECONDS)
    chunks_to_speech_end = math.ceil(
        recording.speech_end_seconds * bytes_per_second / step
    )

    return recording.pcm_bytes[: min(chunks_to_speech_end * step, len(recording.pcm_bytes))]


def benchmark_asr(arguments: argparse.Namespace) -> int:
    """Decode one recording N times and print the percentiles, with no session around it.

    This exists so that a thread count or a decode setting can be measured in seconds instead of in
    a three-minute replay, and so that the ASR stage can be tuned while the brain and the TTS are
    still stand-ins. It decodes the same buffer every time, so the spread is the decoder's own
    variance and nothing else.
    """
    if arguments.asr == "fake":
        raise SystemExit(
            "--benchmark-asr measures the real recogniser; --asr fake has nothing to measure."
        )

    wav_path = arguments.benchmark_wav

    if wav_path is None:
        candidates = sorted(arguments.recordings.glob("*.wav"))

        if not candidates:
            raise SystemExit(
                f"No .wav files in {arguments.recordings} to benchmark on. "
                "Pass one with --benchmark-wav."
            )

        wav_path = candidates[0]

    recording = load_recording(wav_path)
    pcm_bytes = speech_bytes_of(recording)
    bytes_per_second = SAMPLE_RATE_HZ * SAMPLE_WIDTH_BYTES
    audio_seconds = len(pcm_bytes) / bytes_per_second
    file_seconds = len(recording.pcm_bytes) / bytes_per_second

    settings = asr_settings_from(arguments).resolved()
    speech_recognition = real_asr.build(settings)

    print("=" * 78)
    print(f"benchmark         {arguments.benchmark_runs} decodes of {wav_path.name}")
    print(
        f"audio             {audio_seconds:.2f} s to end of speech "
        f"(the file is {file_seconds:.2f} s)"
    )
    print_thread_environment()
    print(f"asr               {settings.describe()}")
    print("=" * 78)

    warmup_started_nanoseconds = time.monotonic_ns()
    speech_recognition.warmup()
    warmup_milliseconds = (
        time.monotonic_ns() - warmup_started_nanoseconds
    ) / NANOSECONDS_PER_MILLISECOND
    print(
        f"warmup            {warmup_milliseconds:9.1f} ms  "
        f"(model load + {settings.warmup_decodes} dummy decode(s))"
    )
    print("")

    durations_milliseconds: list[float] = []
    transcripts: list[str] = []

    for run_number in range(1, arguments.benchmark_runs + 1):
        started_nanoseconds = time.monotonic_ns()
        transcript = speech_recognition.transcribe(pcm_bytes)
        elapsed_milliseconds = (
            time.monotonic_ns() - started_nanoseconds
        ) / NANOSECONDS_PER_MILLISECOND

        durations_milliseconds.append(elapsed_milliseconds)
        transcripts.append(transcript)
        print(f"  run {run_number:<4}{elapsed_milliseconds:9.1f} ms")

    p50 = percentile(durations_milliseconds, 0.5)
    p95 = percentile(durations_milliseconds, 0.95)
    first = durations_milliseconds[0]

    print("")
    print(f"p50               {p50:9.1f} ms")
    print(f"p95               {p95:9.1f} ms")
    print(f"min               {min(durations_milliseconds):9.1f} ms")
    print(f"max               {max(durations_milliseconds):9.1f} ms")
    print(
        f"mean              "
        f"{sum(durations_milliseconds) / len(durations_milliseconds):9.1f} ms"
    )

    # Faster than real time means the decode costs less than the utterance lasted; it is the number
    # that says whether streaming could ever have kept up with the audio in the first place.
    print(f"real time factor  {audio_seconds * 1000 / p50:9.2f}x  (audio seconds per decode second)")

    # The first decode after warmup is where a half-warmed stage shows itself, so it is reported
    # next to the median rather than left for someone to work out from the run list.
    print(
        f"first vs p50      {first / p50 * 100:9.1f} %   "
        f"({first:.1f} ms against a median of {p50:.1f} ms)"
    )

    distinct = sorted(set(transcripts))
    print("")
    print(f"transcript        {shorten(distinct[0], 60)}")

    for other in distinct[1:]:
        print(f"  ALSO            {shorten(other, 60)}  <-- decode is not deterministic")

    return 0


def main(argv: list[str] | None = None) -> int:
    """Replay every recording as one turn and write the trace."""
    arguments = parse_arguments(argv)

    if arguments.benchmark_asr:
        return benchmark_asr(arguments)

    sys.setswitchinterval(FEED_THREAD_SWITCH_INTERVAL_SECONDS)

    recording_paths = sorted(arguments.recordings.glob("*.wav"))

    if not recording_paths:
        raise SystemExit(
            f"No .wav files in {arguments.recordings}. "
            "Record utterances first, then convert them to 16 kHz mono 16-bit WAV."
        )

    speech_recognition, dialogue_manager, speech_synthesis = build_stages(arguments)
    implementations = stage_implementations(
        speech_recognition, dialogue_manager, speech_synthesis
    )

    print_banner(
        arguments,
        implementations,
        len(recording_paths),
        speech_synthesis.sample_rate_hz,
    )

    validate_recordings(recording_paths)
    warn_about_unmarked_dialogues(recording_paths)
    prepare_output_directory(arguments.output_dir, arguments.append)
    warm_up_stages(speech_recognition, dialogue_manager, speech_synthesis)

    session = ReplaySession(
        recording_paths,
        arguments.output_dir,
        sink_sample_rate_hz=speech_synthesis.sample_rate_hz,
    )

    failed_turn_ids: list[str] = []
    previous_dialogue_id: str | None = None

    for turn_number, turn in enumerate(session.turns(), start=1):
        dialogue_id = dialogue_id_of(recording_paths[turn_number - 1])
        starts_new_dialogue = dialogue_id != previous_dialogue_id
        previous_dialogue_id = dialogue_id

        if starts_new_dialogue:
            print(f"--- dialogue {dialogue_id} ---")

        try:
            run_turn(
                turn,
                turn_number,
                speech_recognition,
                dialogue_manager,
                speech_synthesis,
                implementations,
                arguments.fail_turn,
                dialogue_id,
                starts_new_dialogue,
            )

        except Exception as error:  # noqa: BLE001
            # One bad turn does not end the session: its record is already written by finish().
            failed_turn_ids.append(turn.turn_id)
            print(
                f"{turn.turn_id}  FAILED  {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )
            continue

        print(
            f"{turn.turn_id}  {describe_turn(turn)}\n"
            f"             transcript: {shorten(turn.extras.get('transcript'))}\n"
            f"             reply:      {shorten(turn.extras.get('reply'))}",
            flush=True,
        )

    trace_path = session.trace_path
    print(f"\ntrace written to {trace_path}")
    print(f"captured audio in {arguments.output_dir / 'tts'}")

    if failed_turn_ids:
        print(
            f"{len(failed_turn_ids)} turn(s) failed and were traced anyway: "
            f"{', '.join(failed_turn_ids)}",
            file=sys.stderr,
        )

    print(f"check it with: PYTHONPATH=. python3 check_trace.py --trace {trace_path}")

    return 1 if failed_turn_ids else 0


if __name__ == "__main__":
    raise SystemExit(main())
