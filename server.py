from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
import whisper
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Form, Body
from fastapi.responses import FileResponse
from fsm import load_scenario, get_step, get_next_step
import os
import shutil

load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
whisper_model = whisper.load_model("base")
eleven_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

@app.post("/process_turn")
async def process_turn(
    audio: UploadFile,
    scenario_id: str = Form(...),
    current_step_id: str = Form(...),
    fail_count: int = Form(0)
):
    print(f"📥 Request from frontend | scenario: {scenario_id} | step: {current_step_id} | fail_count: {fail_count}")

    audio_path = f"temp_{audio.filename}"
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    result = whisper_model.transcribe(audio_path)
    transcript = result["text"]
    os.remove(audio_path)
    print(f"📝 Transcript: {transcript}")

    scenario = load_scenario(scenario_id)
    step = get_step(scenario, current_step_id)
    next_step_id, passed, next_step = get_next_step(scenario, current_step_id, transcript, fail_count)
    print(f"{'✅' if passed else '❌'} Passed: {passed} | {current_step_id} → {next_step_id}")

    world_data = get_world_data(scenario_id)

    agent = step["scene_context"]["agent"]
    situation = step["scene_context"]["situation"]
    goal = step["scene_context"]["goal"]

    if agent == "buddy":
        persona = load_prompt("buddy_agent.md", {
            "SCENARIO_ID": scenario_id,
            "STEP_ID": current_step_id,
            "SITUATION": situation,
            "GOAL": goal,
            "TARGET_PHRASE": step.get("target_phrase") or "none",
            "FAIL_COUNT": fail_count
        })
    else:
        persona = load_prompt("scene_grt_bus_driver.md", {
            "ROUTE_NUMBER": scenario["scene_defaults"]["route"],
            "DESTINATION": scenario["scene_defaults"]["destination"],
            "SITUATION": situation,
            "USER_INPUT_SUMMARY": transcript
        })

    if world_data:
        persona += f"\n\nLocal facts:\n{world_data}"

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": persona},
            {"role": "user", "content": transcript}
        ]
    )
    reply = response.choices[0].message.content
    print(f"🤖 {agent} said: {reply}")

    audio_out = eleven_client.text_to_speech.convert(
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        text=reply,
        model_id="eleven_monolingual_v1",
        voice_settings={"speed": 0.75}
    )
    with open("reply.mp3", "wb") as f:
        for chunk in audio_out:
            f.write(chunk)

    return {
        "transcript": transcript,
        "reply": reply,
        "agent": agent,
        "passed": passed,
        "current_step": current_step_id,
        "next_step": next_step_id,
        "background": next_step["background"] if next_step else None,
        "ambient_sound": next_step.get("ambient_sound") if next_step else None,
        "ui_hint": next_step.get("ui_hint") if next_step else None,
        "visual_hints": next_step["visual_hints"] if next_step else [],
        "audio_url": "/audio"
    }

@app.post("/tts")
async def tts(payload: dict = Body(...)):
    text = payload.get("text", "")
    print(f"🔊 TTS request: {text}")

    audio_out = eleven_client.text_to_speech.convert(
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        text=text,
        model_id="eleven_monolingual_v1",
        voice_settings={"speed": 0.75}
    )
    with open("tts_output.mp3", "wb") as f:
        for chunk in audio_out:
            f.write(chunk)

    return FileResponse("tts_output.mp3", media_type="audio/mpeg")

@app.get("/audio")
def get_audio():
    return FileResponse("reply.mp3", media_type="audio/mpeg")