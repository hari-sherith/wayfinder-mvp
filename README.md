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
├── scoring.py          Goodness-of-Pronunciation (GOP) scorer (Task 2)
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
uvicorn server:app --reload
```

## Pronunciation scoring

See the **Pronunciation scoring** section below for limitations.

---

*(Tasks 1–5 details to be filled in as work progresses.)*
