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

# Uncalibrated — no labelled data was used to choose this value.
# It was set heuristically and has not been tuned against labelled data.
CONTENT_MATCH_THRESHOLD = 75

def evaluate(transcript, target_phrase, threshold=CONTENT_MATCH_THRESHOLD):
    if target_phrase is None:
        return True, None
    # partial_ratio checks whether the target phrase appears within the transcript.
    # This passes "I need a bus ticket" against target "bus ticket" correctly,
    # where fuzz.ratio would fail due to length mismatch.
    score = fuzz.partial_ratio(transcript.lower(), target_phrase.lower())
    print(f"Match score (partial_ratio): {score}")
    return score >= threshold, score

def get_next_step(scenario, current_step_id, transcript, attempt=1):
    """attempt = this failure number (1 = first failure, 2+ = repeat failures)."""
    step = get_step(scenario, current_step_id)
    passed, content_score = evaluate(transcript, step["target_phrase"])

    if passed:
        next_step_id = step["transitions"].get("on_pass")
    elif attempt == 1:
        next_step_id = step["transitions"].get("on_fail_1x", current_step_id)
    else:
        next_step_id = step["transitions"].get("on_fail_2x", current_step_id)

    next_step = get_step(scenario, next_step_id) if next_step_id else None

    return next_step_id, passed, next_step, content_score
