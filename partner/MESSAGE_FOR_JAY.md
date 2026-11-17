# What I need from you for A4

## First, the thing that blocks everything else: brain.py and tts.py

`brain.py` and `tts.py` are both marked "Owned by Partner B" and both still raise
`NotImplementedError` (`brain.py:31,38,45`, `tts.py:31,38`). Until they're real, the agent can only
run with `--fake-brain --fake-tts`, and `check_trace.py` stamps every such run:

    *** THIS TRACE WAS PRODUCED WITH FAKE STAGES — NOT A REAL MEASUREMENT ***

So we currently have **no end-to-end latency number at all** — not a slow one, none. That's 40 of
the 100 points, and the report needs p50/p95 over at least 30 turns from a real run. My side (the
recordings, sidecars, references, WER) is done and verified; it can't produce a waterfall on its own.

Two things that are easy to get wrong and are worth reading before you start, both from the
interface docstrings:

- **`Brain.reply` must yield WHOLE SENTENCES as soon as each is complete.** The turn loop stamps
  `llm_first_token` on the first thing you yield, so yielding the whole answer at the end destroys
  the headline metric.
- **The first chunk `Tts.stream` yields must already be audible.** The shim stamps
  `tts_first_audio` on the first chunk whose RMS reaches 200, so any padding silence in front is
  added straight onto our time to first audio.

Also: `Brain.reset()` is what I rely on for dialogue state (see §3 below), so please implement it
properly rather than leaving it a no-op.

If you want to split this, I'll take `tts.py` — say so and I'll start on it today.

## Second: your recordings

Everything below is so our two recording sets drop into ONE run and score as two speakers. The
tooling is already written and in the repo — you don't have to build anything, just run it.

## 1. What to record

At least 20 utterances, spread over at least 10 short dialogues, across the three families in
`tasks/task_families.md`: timers/reminders, lookup, two-constraint booking. Include one dialogue per
family where the agent **cannot** give you what you asked for (a missing value, a fact that isn't in
`toy_kb/knowledge_base.json`, a booking where the two constraints can't both hold). Everything
factual has to come from the toy KB — check your lookup answers against it before you record.

One ask on top of that: **keep your utterances roughly as long and as hard as mine.** We're reporting
WER on your voice against WER on mine, and since we're using different sentences, a gap between us
could just as easily be text difficulty as voice. Mine average about 10 words (369 reference words
over 36 utterances), with a handful of proper nouns and times — names like Ravi and Meriam, "ten
o'clock", "five pm", "the eighteenth of November". If yours are all short and plain while mine are
long and full of names, the comparison tells us nothing about our voices. Send me your reference
text and I'll check the two sets line up on length and word count before we run anything.

## 2. File naming — this is load-bearing, not cosmetic

Name every file:

    jay_d21_t01.wav     jay_d21_t02.wav     jay_d22_t01.wav   ...

- `d<NN>` is the dialogue, `t<NN>` is the turn inside it. **Zero-pad both to two digits.**
  The shim replays `sorted(glob("*.wav"))`, which is plain lexicographic order, so `d9` would sort
  after `d10` and shuffle your conversations.
- **Start your dialogue numbers at 21.** Mine are `d01`–`d13`. The agent resets its state when the
  dialogue number *changes*, so if your first dialogue were also `d01` and it landed right after my
  last one, the two would be read as one continuous conversation.
- Keep the `jay_` prefix on every file. It sorts all of yours together after all of mine, and it's
  what tells the WER scorer whose voice it is.

## 3. Why the naming matters: dialogue state reset

`run_agent.py` pulls the dialogue number out of the filename (`run_agent.py:401`, regex
`(?:^|_)d(\d+)(?:_|$)`). It tracks the previous one, and when it changes it calls
`brain.reset()` *before* the next recording starts feeding (`run_agent.py:497-500`) — so the reset
costs nothing inside the measured turn.

If a file carries no `d<NN>` marker, every turn is treated as **one single conversation** and the
brain is never reset. Turn 1 of your dialogue 7 would then still be holding the room booking from
dialogue 6, and it will confidently answer about the wrong room. The run prints a warning when no
file carries a marker, but it cannot warn you about a *missing* reset between two of your dialogues.

A sidecar `"dialogue_id"` overrides the filename if you ever need it, but the filename is enough.

## 4. Audio format — 16 kHz, mono, 16-bit, and one macOS trap

The shim refuses anything else. If you're on macOS:

```bash
afconvert -f WAVE -d LEI16@16000 -c 1 input.aifc jay_d21_t01.wav
```

**Then you must normalize the header.** macOS `afconvert` writes `WAVE_FORMAT_EXTENSIBLE`
(format tag 65534). It opens fine on a recent Python and raises `wave.Error: unknown format: 65534`
on Python 3.9/3.10 — which is PACE-ICE's Python and the Ubuntu 22.04 image we're graded in. The
samples are identical; only the header differs, so this is lossless and does not change WER:

```bash
PYTHONPATH=. python3 tools/normalize_recordings.py --recordings <your dir> --in-place
```

## 5. Leave trailing silence, and keep it actually silent

After your last word, **stop talking and wait about a second** before you stop the recorder. The
shim stamps `end_of_user_speech` from inside the file, not from the end of it, and it needs room
after the speech. At least 0.3 s.

But keep that tail clean. The shim's fallback is an energy rule: the end of the last 20 ms chunk
whose RMS reaches 200. A breath, a lip smack, or a chair creak clears 200 easily, and the rule then
puts end-of-speech 100–200 ms late — which is subtracted straight out of our reported latency and
makes our agent look faster than it is. Six of my 36 had exactly this. So: don't sigh at the end,
don't put the mic where it'll catch you moving, record in a quiet room.

## 6. Check the files before you trust them

```bash
PYTHONPATH=. python3 tools/check_recordings.py --recordings <your dir>
```

It must end with `all N recording(s) are ready for the shim`. It checks the header, the format, the
peak (clipping), the loudest chunk, where the energy rule thinks speech ends, and the trailing
silence. Fix anything it flags and re-record if you have to.

## 7. Sidecars — do NOT skip the listening step

```bash
PYTHONPATH=. python3 make_sidecars.py propose --recordings <your dir>
```

This writes `sidecar_review.json` and a `review_audio/` folder. For each file it gives you the
energy rule's guess, a "the sentence actually looks finished here" candidate when it sees a quiet
detached burst after your last word, and `review_audio/<name>.marked.wav` — your recording with a
1 kHz tick mixed in at the proposed end of speech.

**Play the marked file. The tick must land immediately after your last word**, not after a breath.
Start with the ones it lists as SUSPECT. Then, in `sidecar_review.json`, fix `speech_end_seconds` if
the tick was late, fill in `scenario_node` (a short label for the turn, like `booking.confirm` —
see `tasks/scenario_nodes.json` for mine), and set `"confirmed": true`.

```bash
PYTHONPATH=. python3 make_sidecars.py status      # what's left
PYTHONPATH=. python3 make_sidecars.py write       # writes <name>.json next to each wav
```

`write` **refuses** every entry still marked `confirmed: false`. That's deliberate: a sidecar should
only exist because a person listened to it.

## 8. Your reference transcripts

Type what you **actually said**, not what you meant to say. WER punishes a wrong reference exactly
as hard as it punishes the ASR. Send me your block in this shape and I'll merge it:

```json
"Jay": {
  "jay_d21_t01.wav": "set a timer for ten minutes",
  "jay_d21_t02.wav": "..."
}
```

The scorer ignores case and punctuation and turns digits into words on both sides, so `5` matches
"five" and `3:30` matches "three thirty". But these four are **not** normalized away and are where
points actually get lost:

- **A contraction is not its expansion.** `it's` ≠ `it is`, `she's` ≠ `she is`, `I'd` ≠ `I would`.
  The apostrophe survives. Write whichever one came out of your mouth — this one bit me.
- **A year is not read as a year.** `2026` becomes "two thousand twenty six", never "twenty twenty
  six". If you said it as a year, write the words.
- **Write `am` and `pm` as those two plain letters.** (`p.m.` is read as `pm` too.)
- **A colon makes a time, a dot makes a decimal.** `3:00` → "three o'clock", `3:05` → "three oh
  five", `3.5` → "three point five".

Run this before you send it — it prints exactly what the scorer will compare, and flags all of the
above:

```bash
PYTHONPATH=. python3 tools/lint_references.py --references references.json
```

And once we have a real trace, `--trace run_output/trace.jsonl` adds a word-level diff of your
reference against what the ASR heard. When they differ by one small function word, **listen again** —
the reference is as likely to be the wrong one as the ASR.

## 9. What to send me

1. Your `.wav` files (named as in §2, header-normalized as in §4)
2. Their `.json` sidecars, every one confirmed by ear
3. Your `"Jay": { ... }` reference block
4. The last line of your `tools/check_recordings.py` run, so I know it passed on your machine

I'll drop them into `recordings/` alongside mine and run the whole thing as one pass — one trace,
both speakers scored separately, and enough turns for the latency percentiles.
