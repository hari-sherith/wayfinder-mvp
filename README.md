# Wayfinder MVP — Backend

This backend is forked from the **Wayfinder Waterloo** project, built at the
AI for Good Hackathon (Waterloo).

## Authorship

- **Voice pipeline, FSM, and API server** (`server.py`, `fsm.py`, `pipeline.py`):
  written by the backend author during the hackathon.
- **Scenario content and persona prompts** under `content/` (scenario JSON files
  and character prompt markdown): written by a teammate and carried over here
  as-is.

## Repository structure

```
wayfinder-mvp/
├── server.py           FastAPI app — STT → FSM → LLM → TTS pipeline
├── fsm.py              Scenario loader and step evaluator
├── pipeline.py         Standalone pipeline runner (used for local testing)
├── scoring.py          Goodness-of-Pronunciation (GOP) scorer
├── requirements.txt    Python dependencies
├── Procfile            Heroku/Railway process definition
├── runtime.txt         Python version pin
├── .env.example        Required environment variables
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

## Setup

```bash
cp .env.example .env
# fill in ELEVENLABS_API_KEY and GROQ_API_KEY
pip install -r requirements.txt
python3 -m uvicorn server:app --port 8000 --reload
# open demo/index.html in a browser
```

## Pre-warming the step-opener cache (recommended before a demo)

`/start_step` caches each step's opening line in `cache/start_<scenario>_<step>.mp3`
and a matching `.txt`.  On a cache hit the server returns the file directly with no
LLM or TTS call, making every step transition instant and the openers deterministic.

```bash
python warm_cache.py                 # warms all steps, skips already-cached
python warm_cache.py --no-cache      # force-regenerates every opener
```

To regenerate a single bad line, delete its `cache/start_*.mp3` file and reload
that step in the browser (or rerun `warm_cache.py --no-cache`).

To always bypass the cache at the server level, set `START_STEP_NO_CACHE=1` in
`.env` before starting uvicorn.

## Pronunciation scoring

### Model and G2P

- **Acoustic model**: `bookbot/wav2vec2-ljspeech-gruut`
  (wav2vec2-base, 94 M params, ~376 MB download on first run).
  Fine-tuned on LJSpeech with gruut IPA phoneme labels.
  Outputs frame-level CTC posteriors over a 44-phone IPA vocabulary.

- **G2P**: `gruut` en-us (pure Python, no system dependencies).
  Converts the target phrase to the same gruut IPA phoneme set as the model.

- **Alphabet**: both systems use gruut's en-us IPA inventory.
  IPA stress markers (`ˈ`, `ˌ`) are stripped from gruut output before
  lookup because the model was trained without them.
  `verify_alphabet()` in `scoring.py` prints both sets and aborts on mismatch.

- **Scoring method**: CTC Viterbi forced alignment, then per-phone GOP ratio
  = mean(posterior of target phone / max posterior at that frame), scaled 0–100.

### Limitations — stated plainly

1. **GOP scores are not validated against human raters.**
   No labelled pronunciation data exists for this project.
   The scores are derived from a model trained on a single American-English
   speaker (LJSpeech) and have not been calibrated or compared to human
   judgements on L2 English speech.

2. **Validation used synthetic audio only, and the ordering is wrong for this product.**
   The verification gate used macOS `say` (system TTS) and ElevenLabs TTS
   as clean/accented/mangled proxies.  No actual learners were recorded.
   Scores after the `|`-mask fix:

   | Sample | Overall | Notes |
   |--------|---------|-------|
   | clean.wav (macOS TTS, neutral American) | 86 | |
   | mangled.wav (macOS TTS, deliberate mispronunciation: "buh suh tee ket") | 67 | |
   | accented.wav (ElevenLabs French voice, correct words) | 40 | |

   The scorer rates a deliberate mispronunciation 27 points above accented-but-correct
   speech.  Every user of this app is an accented speaker saying the right words.
   This ordering is the wrong direction for a pronunciation-coaching product.

   Root cause: the model was trained on a single neutral American-English speaker
   (LJSpeech).  For the French-accented sample, the model's argmax frames never
   output `b` — they output `j` (palatal approximant) with posterior 0.0006 for
   `b`, meaning the model has never seen this realisation of the phone.  The
   mangled sample uses native-distribution phones even for wrong words, so the
   model recognises them confidently.  The scorer measures acoustic similarity to
   LJSpeech, not phonemic correctness.

3. **The content-match threshold is uncalibrated.**
   `CONTENT_MATCH_THRESHOLD = 75` in `fsm.py` was set heuristically during
   the hackathon and has never been tuned against labelled pass/fail data.

4. **Whisper WER on Rohingya-accented L2 English is unmeasured.**
   The STT step uses `openai/whisper-base`.  Its word error rate on
   Rohingya-accented English is unknown.  Transcripts that feed the content
   check are of unknown quality for the target user population.

5. **The scoring model is not adapted to L2 English.**
   A model trained on one native-English speaker's voice will assign low
   posteriors to L2 phoneme realisations that are acoustically valid but
   different from the training distribution.  Scores may be systematically
   low for non-native speakers regardless of actual pronunciation quality.

6. **Forced alignment cannot be used as a validity gate.**
   The CTC Viterbi aligner produces a per-phone score for any audio against
   any phrase, including audio where the wrong words were spoken.  The API
   refuses to return a pronunciation score when the content check fails
   (`status: "unavailable", reason: "content_mismatch"`), but this means
   pronunciation feedback is only shown on successful attempts — learners
   who say the wrong phrase get no phoneme-level feedback at all.
