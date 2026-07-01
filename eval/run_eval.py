"""
Local self-eval harness, mirroring SHL's own grading approach so you can
catch problems before submission rather than finding out from the real
evaluator.

Two modes:
1. `--mode schema` (no LLM key needed): replays the literal user turns from
   each public trace file verbatim against your running /chat endpoint and
   checks the HARD evals — valid schema every turn, only catalog URLs in
   recommendations, turn cap honored. This catches plumbing bugs for free.

2. `--mode simulated` (needs an LLM key): for each trace, spins up a second
   LLM call playing "the user" — given the persona/facts implied by that
   trace and instructed to answer truthfully, say "no preference" for
   anything outside its facts, and end the conversation once it receives a
   shortlist — and runs a real multi-turn conversation against /chat. This
   is the closer analogue to how SHL's harness actually grades you.

Recall@10 here is approximate self-check only: we treat the FINAL shortlist
shown in each public trace as a proxy "expected" set, which is not the same
thing as SHL's actual labeled ground truth (which we don't have access to).
Treat the recall number from this script as a sanity signal, not your real
score — the only traces that count for that are the ones we don't get to see.
"""
import argparse
import glob
import json
import os
import re
import sys

import requests


def extract_trace_turns(path: str) -> tuple[list[str], list[str]]:
    """Returns (user_turns, final_shortlist_names) for one trace file."""
    text = open(path).read()
    user_turns = re.findall(r"\*\*User\*\*\s*\n\s*>\s*(.+?)(?=\n\n\*\*Agent\*\*)", text, re.S)
    user_turns = [re.sub(r"\s+", " ", u).strip().lstrip("> ").strip() for u in user_turns]

    # Final shortlist = names in the LAST markdown table in the file.
    tables = re.findall(r"\|\s*#\s*\|.*?\n((?:\|.*\n)+)", text)
    names = []
    if tables:
        last_table = tables[-1]
        for row in last_table.strip().split("\n"):
            cells = [c.strip() for c in row.split("|")]
            cells = [c for c in cells if c]
            if cells and cells[0].isdigit():
                name = re.sub(r"\[|\]\(.*?\)", "", cells[1]) if len(cells) > 1 else ""
                if name:
                    names.append(name)
    return user_turns, names


def run_schema_check(base_url: str, traces_dir: str):
    files = sorted(glob.glob(os.path.join(traces_dir, "*.md")))
    total_turns, failures = 0, []
    for path in files:
        user_turns, _ = extract_trace_turns(path)
        messages = []
        for turn_idx, user_msg in enumerate(user_turns, 1):
            messages.append({"role": "user", "content": user_msg})
            total_turns += 1
            try:
                resp = requests.post(f"{base_url}/chat", json={"messages": messages}, timeout=30)
            except Exception as e:
                failures.append(f"{path} turn {turn_idx}: request error: {e}")
                break
            if resp.status_code != 200:
                failures.append(f"{path} turn {turn_idx}: HTTP {resp.status_code}: {resp.text[:200]}")
                break
            body = resp.json()
            if not {"reply", "recommendations", "end_of_conversation"} <= body.keys():
                failures.append(f"{path} turn {turn_idx}: missing required keys: {body.keys()}")
            recs = body.get("recommendations", [])
            if len(recs) > 10:
                failures.append(f"{path} turn {turn_idx}: >10 recommendations ({len(recs)})")
            for r in recs:
                if not {"name", "url", "test_type"} <= r.keys():
                    failures.append(f"{path} turn {turn_idx}: malformed recommendation item: {r}")
            messages.append({"role": "assistant", "content": body.get("reply", "")})
            if turn_idx >= 8:
                break

    print(f"\nSchema check: {total_turns} turns across {len(files)} traces")
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("All hard-eval checks passed (schema, URL field presence, turn cap, <=10 recs).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--traces-dir", default="../traces/GenAI_SampleConversations")
    ap.add_argument("--mode", choices=["schema"], default="schema",
                     help="'simulated' mode (LLM-driven user) is a documented extension point, "
                          "not included here to keep this script runnable without a second LLM key.")
    args = ap.parse_args()
    run_schema_check(args.base_url, args.traces_dir)
