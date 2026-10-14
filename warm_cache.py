#!/usr/bin/env python3
"""
warm_cache.py — Pre-fill the /start_step cache for every step in a scenario.

Usage:
    python warm_cache.py                          # warms grt_bus_ticket, skips cached
    python warm_cache.py --no-cache               # regenerates every step
    python warm_cache.py --scenario my_scenario   # different scenario

Run this before a demo so every step opener is deterministic and instant.
The generated lines are printed to stdout and saved alongside the audio as
cache/start_<scenario_id>_<step_id>.txt.
"""

import argparse
import json
import sys
import urllib.request
import urllib.error

DEFAULT_SCENARIO = "grt_bus_ticket"
DEFAULT_BASE     = "http://localhost:8000"


def main():
    parser = argparse.ArgumentParser(description="Warm /start_step cache")
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--base",     default=DEFAULT_BASE)
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass existing cache and regenerate every step")
    args = parser.parse_args()

    # Load scenario to get step list
    scenario_path = f"content/scenarios/{args.scenario}.json"
    try:
        with open(scenario_path) as f:
            scenario = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {scenario_path} not found. Run from the wayfinder-mvp directory.")
        sys.exit(1)

    steps = [s["step_id"] for s in scenario.get("steps", [])]
    print(f"Warming {len(steps)} steps for scenario '{args.scenario}'")
    print(f"Base URL: {args.base}\n")

    for step_id in steps:
        url = (
            f"{args.base}/start_step"
            f"?scenario_id={args.scenario}"
            f"&step_id={step_id}"
            + ("&no_cache=true" if args.no_cache else "")
        )
        print(f"  [{step_id}] ", end="", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                size = len(resp.read())
            # Read the saved text alongside
            txt_path = f"cache/start_{args.scenario}_{step_id}.txt"
            try:
                with open(txt_path) as tf:
                    line = tf.read().strip()
            except FileNotFoundError:
                line = "(text file not found)"
            print(f"ok ({size//1024} KB) — \"{line}\"")
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code}: {e.reason}")
        except Exception as e:
            print(f"ERROR: {e}")

    print("\nCache warmed. Start the server and open demo/index.html.")


if __name__ == "__main__":
    main()
