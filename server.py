from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import whisper
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Form, Body, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fsm import load_scenario, get_step, get_next_step
import os
import shutil
import time

load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/assets", StaticFiles(directory="content/assets"), name="assets")

whisper_model = whisper.load_model("base")
eleven_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Lazy-import scoring so the server still boots if the model isn't cached yet.
# score_utterance is called per request; first call downloads the model (~376MB).
def _score(audio_path, target_phrase):
    try:
        from scoring import score_utterance
        return score_utterance(audio_path, target_phrase)
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc), "overall": None, "phones": None}


def load_prompt(filename, replacements):
    with open(f"content/prompts/{filename}", "r") as f:
        prompt = f.read()
    for key, value in replacements.items():
        prompt = prompt.replace(f"{{{{{key}}}}}", str(value))
    return prompt

def get_world_data(scenario_id):
    path = f"content/scenarios/{scenario_id}.txt"
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return ""

def _tts(text):
    audio_out = eleven_client.text_to_speech.convert(
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings={"speed": 0.75}
    )
    return audio_out

@app.post("/process_turn")
async def process_turn(
    audio: UploadFile,
    scenario_id: str = Form(...),
    current_step_id: str = Form(...),
    fail_count: int = Form(0)
):
    print(f"📥 Request from frontend | scenario: {scenario_id} | step: {current_step_id} | fail_count: {fail_count}")
    timings = {}

    audio_path = f"temp_{audio.filename}"
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    # --- STT ---
    t0 = time.perf_counter()
    result = whisper_model.transcribe(audio_path, language="en", fp16=False)
    transcript = result["text"]
    timings["stt_ms"] = round((time.perf_counter() - t0) * 1000)
    print(f"📝 Transcript: {transcript}  [{timings['stt_ms']}ms]")

    scenario = load_scenario(scenario_id)
    step = get_step(scenario, current_step_id)
    # fail_count = prior failures at this step (sent by frontend, starts at 0).
    # attempt = this failure number; used for routing and the buddy prompt.
    attempt = fail_count + 1
    next_step_id, passed, next_step, content_score = get_next_step(
        scenario, current_step_id, transcript, attempt
    )
    print(f"{'✅' if passed else '❌'} Passed: {passed} | {current_step_id} → {next_step_id}")

    # --- Pronunciation scoring ---
    # Only score pronunciation when the content check passed.  Forced alignment
    # produces a plausible-looking score for any audio against any phrase; scoring
    # wrong words gives misleading per-phone feedback, so we refuse when the
    # transcript did not match the target phrase.
    target_phrase = step.get("target_phrase")
    t0 = time.perf_counter()
    if not target_phrase:
        pronunciation = {"status": "unavailable", "reason": "step has no target_phrase",
                         "overall": None, "phones": None}
    elif not passed:
        pronunciation = {"status": "unavailable", "reason": "content_mismatch",
                         "overall": None, "phones": None}
    else:
        pronunciation = _score(audio_path, target_phrase)
    timings["scoring_ms"] = round((time.perf_counter() - t0) * 1000)
    print(f"🎯 Pronunciation: overall={pronunciation.get('overall')}  [{timings['scoring_ms']}ms]")

    os.remove(audio_path)

    total = timings["stt_ms"] + timings["scoring_ms"]
    print(f"⏱  Timings: STT={timings['stt_ms']}ms  scoring={timings['scoring_ms']}ms  total={total}ms")

    # On pass, skip LLM+TTS — the frontend calls /start_step for the next step,
    # which handles all audio.  Return scores only so the panel can render.
    if passed:
        return {
            "transcript": transcript,
            "reply": None,
            "agent": None,
            "passed": True,
            "current_step": current_step_id,
            "next_step": next_step_id,
            "background": next_step["background"] if next_step else None,
            "ambient_sound": next_step.get("ambient_sound") if next_step else None,
            "ui_hint": next_step.get("ui_hint") if next_step else None,
            "visual_hints": next_step["visual_hints"] if next_step else [],
            "audio_url": None,
            "content_match": {"score": content_score, "passed": True, "method": "fuzz.ratio"},
            "pronunciation": pronunciation,
            "timings": timings,
        }

    # On fail, run LLM+TTS so Buddy can give corrective feedback.
    world_data = get_world_data(scenario_id)

    agent = step["scene_context"]["agent"]
    situation = step["scene_context"]["situation"]
    goal = step["scene_context"]["goal"]

    prompt_file = scenario["characters"][agent] + ".md"

    if agent == "buddy":
        persona = load_prompt(prompt_file, {
            "SCENARIO_ID": scenario_id,
            "STEP_ID": current_step_id,
            "SITUATION": situation,
            "GOAL": goal,
            "TARGET_PHRASE": step.get("target_phrase") or "none",
            "FAIL_COUNT": attempt
        })
    else:
        persona = load_prompt(prompt_file, {
            "ROUTE_NUMBER": scenario["scene_defaults"]["route"],
            "DESTINATION": scenario["scene_defaults"]["destination"],
            "SITUATION": situation,
            "USER_INPUT_SUMMARY": transcript
        })

    if world_data:
        persona += f"\n\nLocal facts:\n{world_data}"

    # --- LLM ---
    t0 = time.perf_counter()
    response = groq_client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": persona},
            {"role": "user", "content": transcript}
        ]
    )
    reply = response.choices[0].message.content
    timings["llm_ms"] = round((time.perf_counter() - t0) * 1000)
    print(f"🤖 {agent} said: {reply}  [{timings['llm_ms']}ms]")

    # --- TTS ---
    t0 = time.perf_counter()
    audio_out = _tts(reply)
    with open("reply.mp3", "wb") as f:
        for chunk in audio_out:
            f.write(chunk)
    timings["tts_ms"] = round((time.perf_counter() - t0) * 1000)
    print(f"🔊 TTS done  [{timings['tts_ms']}ms]")

    total = timings["stt_ms"] + timings["scoring_ms"] + timings["llm_ms"] + timings["tts_ms"]
    print(f"⏱  Timings (fail): STT={timings['stt_ms']}ms  scoring={timings['scoring_ms']}ms  LLM={timings['llm_ms']}ms  TTS={timings['tts_ms']}ms  total={total}ms")

    return {
        "transcript": transcript,
        "reply": reply,
        "agent": agent,
        "passed": False,
        "current_step": current_step_id,
        "next_step": next_step_id,
        "background": step["background"] if step else None,
        "ambient_sound": step.get("ambient_sound") if step else None,
        "ui_hint": step.get("ui_hint") if step else None,
        "visual_hints": step["visual_hints"] if step else [],
        "audio_url": "/audio",
        "content_match": {"score": content_score, "passed": False, "method": "fuzz.ratio"},
        "pronunciation": pronunciation,
        "timings": timings,
    }


_CACHE_DIR = "cache"
_NO_CACHE = os.getenv("START_STEP_NO_CACHE", "").lower() in ("1", "true", "yes")
os.makedirs(_CACHE_DIR, exist_ok=True)


@app.get("/start_step")
async def start_step(
    scenario_id: str = Query(...),
    step_id: str = Query(...),
    no_cache: bool = Query(False),
):
    """Generate and return the opening line spoken by the agent when a step loads.

    Responses are cached in cache/start_<scenario_id>_<step_id>.mp3 (+ .txt).
    Bypass with ?no_cache=true or env var START_STEP_NO_CACHE=1.
    """
    cache_mp3 = os.path.join(_CACHE_DIR, f"start_{scenario_id}_{step_id}.mp3")
    cache_txt = os.path.join(_CACHE_DIR, f"start_{scenario_id}_{step_id}.txt")
    use_cache = not (no_cache or _NO_CACHE)

    if use_cache and os.path.exists(cache_mp3):
        print(f"🎬 start_step {step_id} [cache hit]")
        return FileResponse(cache_mp3, media_type="audio/mpeg")

    scenario = load_scenario(scenario_id)
    step = get_step(scenario, step_id)
    if not step:
        return JSONResponse({"error": "step not found"}, status_code=404)

    world_data = get_world_data(scenario_id)
    agent = step["scene_context"]["agent"]
    situation = step["scene_context"]["situation"]
    goal = step["scene_context"]["goal"]
    target = step.get("target_phrase") or "none"

    prompt_file = scenario["characters"][agent] + ".md"

    if agent == "buddy":
        persona = load_prompt(prompt_file, {
            "SCENARIO_ID": scenario_id,
            "STEP_ID": step_id,
            "SITUATION": situation,
            "GOAL": goal,
            "TARGET_PHRASE": target,
            "FAIL_COUNT": 0,
        })
    else:
        persona = load_prompt(prompt_file, {
            "ROUTE_NUMBER": scenario["scene_defaults"]["route"],
            "DESTINATION": scenario["scene_defaults"]["destination"],
            "SITUATION": situation,
            "USER_INPUT_SUMMARY": "[Scene begins. You speak first.]",
        })

    if world_data:
        persona += f"\n\nLocal facts:\n{world_data}"

    response = groq_client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": persona},
            {"role": "user", "content": "[Scene begins. You speak first.]"},
        ]
    )
    reply = response.choices[0].message.content
    print(f"🎬 start_step {step_id} | {agent}: {reply}")

    audio_out = _tts(reply)
    with open(cache_mp3, "wb") as f:
        for chunk in audio_out:
            f.write(chunk)
    with open(cache_txt, "w") as f:
        f.write(reply)

    return FileResponse(cache_mp3, media_type="audio/mpeg")


@app.post("/tts")
async def tts(payload: dict = Body(...)):
    text = payload.get("text", "")
    print(f"🔊 TTS request: {text}")
    audio_out = _tts(text)
    with open("tts_output.mp3", "wb") as f:
        for chunk in audio_out:
            f.write(chunk)
    return FileResponse("tts_output.mp3", media_type="audio/mpeg")


@app.get("/tts_target")
async def tts_target(phrase: str = Query(..., description="Target phrase to synthesise")):
    """Return model pronunciation audio for a target phrase."""
    print(f"🎙  tts_target: {phrase}")
    audio_out = _tts(phrase)
    path = "tts_target_output.mp3"
    with open(path, "wb") as f:
        for chunk in audio_out:
            f.write(chunk)
    return FileResponse(path, media_type="audio/mpeg")


@app.get("/audio")
def get_audio():
    return FileResponse("reply.mp3", media_type="audio/mpeg")
