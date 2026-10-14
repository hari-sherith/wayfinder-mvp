# Wayfinder MVP

An audio-first, zero-literacy life-skills simulator for newcomers.

The app teaches real-world interactions — boarding a bus, buying a transit pass —
through voice-only roleplay with an AI community guide (Buddy) and scene characters
(a bus driver). No reading required. Every step starts with the agent speaking first.

---

## Authorship

- **Voice pipeline, FSM, API server, pronunciation scorer, demo UI**
  (`server.py`, `fsm.py`, `scoring.py`, `demo/index.html`, `warm_cache.py`):
  written by the backend author.
- **Scenario content and persona prompts** under `content/` (scenario JSON files
  and character prompt markdown): written by a teammate and carried over here as-is.

---

## Repository structure

```
wayfinder-mvp/
├── server.py           FastAPI app — full request pipeline
├── fsm.py              Scenario loader, step evaluator, content matcher
├── scoring.py          Goodness-of-Pronunciation (GOP) scorer
├── warm_cache.py       Pre-fill /start_step cache before a demo
├── diag_alignment.py   Viterbi alignment diagnostic (development tool)
├── pipeline.py         Standalone pipeline runner (local testing)
├── requirements.txt    Python dependencies
├── Procfile            Heroku/Railway process definition
├── runtime.txt         Python version pin
├── .env.example        Required environment variables
├── cache/              Generated step-opener audio (gitignored)
├── demo/
│   └── index.html      Self-contained demo page (no build step)
└── content/
    ├── prompts/        Character system prompts (markdown)
    │   ├── buddy_agent.md
    │   └── scene_bus_driver.md
    ├── scenarios/      Scenario definitions (JSON)
    │   └── grt_bus_ticket.json
    └── assets/
        └── placeholders/   Background images used by the demo
```

---

## Setup

```bash
cp .env.example .env
# fill in ELEVENLABS_API_KEY and GROQ_API_KEY
pip install -r requirements.txt
python3 -m uvicorn server:app --port 8000 --reload
python3 warm_cache.py        # pre-fill step openers (recommended)
# open demo/index.html in a browser
```

---

## Pre-warming the step-opener cache

`/start_step` caches each step's opening line in `cache/start_<scenario>_<step>.mp3`
and a matching `.txt`.  On a cache hit the server returns the file directly with no
LLM or TTS call, making every step transition instant and the openers deterministic.

```bash
python warm_cache.py                 # warms all steps, skips already-cached
python warm_cache.py --no-cache      # force-regenerates every opener
```

To regenerate a single bad line: delete its `cache/start_*.mp3` and reload
that step in the browser, or rerun with `--no-cache`.

To always bypass the cache, set `START_STEP_NO_CACHE=1` in `.env`.

---

## Technical pipeline

### Per-turn flow (`POST /process_turn`)

```
Browser MediaRecorder → WebM audio blob
        ↓
FastAPI /process_turn
        ↓
Whisper base  — STT (language="en", fp16=False)
        ↓
FSM (fsm.py)  — fuzz.partial_ratio(transcript, target_phrase) >= 75
        ↓
    PASS ──────────────────────────────────────────────────────────┐
        ↓                                                           │
    GOP scorer (scoring.py)                                        │
      · librosa.load + resample to 16 kHz                         │
      · librosa.effects.trim (top_db=25) — remove silence         │
      · wav2vec2 → (T, V) log-softmax posteriors                  │
      · mask word-boundary token "|" (id=0) to -inf               │
      · CTC Viterbi forced alignment → per-phone frame ranges     │
      · GOP = mean(posterior(target) / max_posterior) per frame   │
      · scaled 0–100 per phone, mean for overall                  │
        ↓                                                           │
    return scores + next_step  ←───────────────────────────────────┘
    (no LLM, no TTS on pass — saves 3–5 s per correct answer)
        ↓
    FAIL
        ↓
    Groq LLM (qwen/qwen3.8-27b)
      · buddy_agent prompt with SITUATION, GOAL, TARGET_PHRASE, FAIL_COUNT
      · FAIL_COUNT = attempt number (fail_count_from_frontend + 1)
      · attempt 1 → gentle repeat; attempt 2+ → syllable breakdown
        ↓
    ElevenLabs TTS (eleven_multilingual_v2, speed=0.75)
        ↓
    return correction audio + scores
```

### Proactive step intro (`GET /start_step`)

Called by the frontend whenever a new step loads — on page open, after Continue,
and after tap-to-advance.  The agent speaks before the user does.

```
GET /start_step?scenario_id=X&step_id=Y
        ↓
cache hit?  → return cache/start_X_Y.mp3 immediately (~35 ms)
        ↓ miss
Groq LLM  → ElevenLabs TTS
        ↓
write cache/start_X_Y.mp3 + .txt, return audio
```

### FSM (fsm.py)

Scenarios are JSON files under `content/scenarios/`.  Each step defines:

- `target_phrase` — what the user must say (null for non-verbal steps)
- `scene_context` — agent, situation, goal (drives the LLM prompt)
- `transitions` — `on_pass`, `on_fail_1x`, `on_fail_2x`
- `visual_hints`, `background`, `ambient_sound`, `ui_hint`

Content matching uses `fuzz.partial_ratio` so the target phrase passes
when spoken as part of a longer utterance (e.g. "I need a bus ticket"
matches target "bus ticket").  Threshold is 75 (uncalibrated).

### Fail-count escalation

The frontend sends `fail_count` = number of prior failures at the current step
(starts at 0).  The server computes `attempt = fail_count + 1` and uses that
for both FSM routing and the `{{FAIL_COUNT}}` value in the Buddy prompt:

| attempt | FSM branch | Buddy behaviour |
|---------|-----------|-----------------|
| 1 | `on_fail_1x` | Gentle repeat |
| 2+ | `on_fail_2x` | Syllable breakdown of target phrase |

### Pronunciation scoring (scoring.py)

**Model**: `bookbot/wav2vec2-ljspeech-gruut` (wav2vec2-base, 94 M params, ~376 MB).
Fine-tuned on LJSpeech with gruut IPA phoneme labels.
Outputs frame-level CTC posteriors over a 44-phone IPA vocabulary.

**G2P**: `gruut` en-us.  IPA stress markers (`ˈ`, `ˌ`) are stripped before
vocab lookup — the model was trained without them.

**Word-boundary mask**: the model uses `|` (id=0) as an inter-phone separator,
appearing in ~30 % of frames by argmax.  Without masking, these frames are
absorbed into adjacent phone states and depress GOP scores.  The `|` column
is set to −∞ in `log_probs` before alignment so the Viterbi cannot route
through it.

**Alignment**: CTC Viterbi forced alignment over the extended label sequence
`[blank, p0, blank, p1, ..., pN, blank]`.  Each phone is assigned a frame
range; the GOP ratio at each frame is `posterior(target) / max_posterior`.

**Content-match gate**: pronunciation is not scored when the transcript
failed the content check.  Forced alignment produces valid-looking scores
for any audio against any phrase; showing them when the wrong words were
spoken would give misleading per-phone feedback.

---

## Demo UI (demo/index.html)

Single static HTML file, no build step, dark high-contrast theme.

| Feature | Detail |
|---------|--------|
| Start overlay | Full-screen tap required before any audio — satisfies browser autoplay policy |
| Proactive audio | Agent speaks on every step load; record/tap button hidden until audio ends |
| Phrase hint | "Try saying: `bus ticket`" shown above record button on verbal steps |
| Scoring panel | Transcript · word-match score + PASS/FAIL · per-phoneme colour bars · audio compare · latency breakdown |
| Audio compare | Play your own recording vs ElevenLabs model pronunciation side by side |
| Scene image | Background changes per step |
| Tap-to-continue | Non-verbal steps advance directly client-side (no API call) |

---

## Pronunciation scoring — limitations stated plainly

1. **GOP scores are not validated against human raters.**
   No labelled pronunciation data exists for this project.
   Scores derive from a model trained on a single American-English speaker
   and have not been calibrated against human judgements on L2 speech.

2. **Validation used synthetic audio only, and the ordering is wrong for this product.**
   The verification gate used macOS `say` and ElevenLabs TTS as proxies.
   No actual learners were recorded.  Scores after the `|`-mask fix:

   | Sample | Overall |
   |--------|---------|
   | clean.wav — macOS TTS, neutral American | 86 |
   | mangled.wav — macOS TTS, deliberate mispronunciation ("buh suh tee ket") | 67 |
   | accented.wav — ElevenLabs French voice, correct words | 40 |

   The scorer rates deliberate mispronunciation 27 points above accented-but-correct
   speech.  Every user of this app is an accented speaker saying the right words.
   This ordering is the wrong direction for a pronunciation-coaching product.

   Root cause: the model was trained on one neutral American-English speaker (LJSpeech).
   For the French-accented sample, the model's argmax never outputs `b` — it outputs
   `j` (palatal approximant) with 0.06 % posterior for `b`.  The mangled sample uses
   native-distribution phones for wrong words, so the model recognises them
   confidently.  The scorer measures acoustic similarity to LJSpeech, not
   phonemic correctness.

3. **The content-match threshold is uncalibrated.**
   `CONTENT_MATCH_THRESHOLD = 75` was set heuristically and has never been
   tuned against labelled pass/fail data.

4. **Whisper WER on Rohingya-accented L2 English is unmeasured.**
   The STT step uses `openai/whisper-base`.  Its word error rate on
   Rohingya-accented English is unknown.

5. **The scoring model is not adapted to L2 English.**
   A model trained on one native speaker will assign low posteriors to L2
   phoneme realisations that are acoustically valid but outside its training
   distribution.  Scores may be systematically low for non-native speakers
   regardless of actual pronunciation quality.

6. **Forced alignment cannot be used as a validity gate.**
   The CTC Viterbi aligner produces a plausible-looking score for any audio
   against any phrase.  The API refuses to score when the content check fails
   (`status: "unavailable", reason: "content_mismatch"`), but this means
   phoneme-level feedback is withheld on every failed attempt.
