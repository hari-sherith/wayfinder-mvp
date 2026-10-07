import json
import os
from thefuzz import fuzz

def load_scenario(scenario_id):
    path = f"content/scenarios/{scenario_id}.json"
    with open(path, "r") as f:
        return json.load(f)

def get_step(scenario, step_id):
    for step in scenario["steps"]:
        if step["step_id"] == step_id:
            return step
    return None

def evaluate(transcript, target_phrase, threshold=75):
    if target_phrase is None:
        return True
    score = fuzz.ratio(transcript.lower(), target_phrase.lower())
    print(f"Match score: {score}")
    return score >= threshold

def get_next_step(scenario, current_step_id, transcript, fail_count=0):
    step = get_step(scenario, current_step_id)
    passed = evaluate(transcript, step["target_phrase"])

    if passed:
        next_step_id = step["transitions"].get("on_pass")
    elif fail_count == 1:
        next_step_id = step["transitions"].get("on_fail_1x", current_step_id)
    else:
        next_step_id = step["transitions"].get("on_fail_2x", current_step_id)

    next_step = get_step(scenario, next_step_id) if next_step_id else None

    return next_step_id, passed, next_step
