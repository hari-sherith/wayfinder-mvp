import whisper
from groq import Groq
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import os

load_dotenv()
whisper_model = whisper.load_model("base")
eleven_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

def load_buddy_prompt(scenario_id, step_id, situation, goal, target_phrase, fail_count):
    with open("content/prompts/buddy_agent.md", "r") as f:
        prompt = f.read()
    return prompt.replace("{{SCENARIO_ID}}", scenario_id) \
                 .replace("{{STEP_ID}}", step_id) \
                 .replace("{{SITUATION}}", situation) \
                 .replace("{{GOAL}}", goal) \
                 .replace("{{TARGET_PHRASE}}", target_phrase) \
                 .replace("{{FAIL_COUNT}}", str(fail_count))

def load_driver_prompt(route_number, destination, situation, user_input_summary):
    with open("content/prompts/scene_bus_driver.md", "r") as f:
        prompt = f.read()
    return prompt.replace("{{ROUTE_NUMBER}}", route_number) \
                 .replace("{{DESTINATION}}", destination) \
                 .replace("{{SITUATION}}", situation) \
                 .replace("{{USER_INPUT_SUMMARY}}", user_input_summary)

def run_pipeline(audio_file, persona_prompt, agent_name="agent"):
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Step 1: transcribe
    result = whisper_model.transcribe(audio_file)
    transcript = result["text"]
    print(f"User said: {transcript}")

    # Step 2: respond in character
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": transcript}
        ]
    )
    reply = response.choices[0].message.content
    print(f"{agent_name} said: {reply}")

    # Step 3: speak the reply
    audio = eleven_client.text_to_speech.convert(
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        text=reply,
        model_id="eleven_monolingual_v1",
        voice_settings={"speed": 0.75}
    )
    with open("reply.mp3", "wb") as f:
        for chunk in audio:
            f.write(chunk)

    print("Audio saved to reply.mp3")
    return transcript, reply

if __name__ == "__main__":
    # --- Test --- swap this block to test different scenarios
    persona = load_driver_prompt(
        route_number="7",
        destination="Conestoga Mall",
        situation="user_presenting_ticket",
        user_input_summary="Here you go"
    )
    import sys
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "test.wav"
    run_pipeline(audio_file, persona, agent_name="bus_driver")
