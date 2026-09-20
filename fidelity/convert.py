"""tau2 runner `Results` JSON -> the flat JSONL episode schema (+ summary.json).

Lets rollouts produced locally be read by the same measurement scripts as any other traces.

Nodes: assistant turns (sampled, with tool calls), user prose, and assistant-side tool results. User-side
tool calls and their results are skipped, because they are not visible to the agent.

Usage: python fidelity/convert.py --results runs/teacher_k3.json --tasks out/telecom/tasks.jsonl \
           --out runs/teacher_k3/traces.jsonl.gz [--label teacher-k3]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STOP_MAP = {"user_stop": "user_completed", "agent_stop": "agent_completed", "max_steps": "tau2_max_steps",
            "too_many_errors": "tau2_too_many_errors"}
ERROR_REASONS = {"agent_error", "user_error", "infrastructure_error", "unexpected_error", "context_window_exceeded"}


def sim_to_nodes(messages: list[dict]) -> list[dict]:
    nodes = []
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            msg = {"role": "assistant", "content": m.get("content")}
            rc = (m.get("raw_data") or {}).get("reasoning_content") if isinstance(m.get("raw_data"), dict) else None
            if rc:
                msg["reasoning_content"] = rc
            if m.get("tool_calls"):
                msg["tool_calls"] = [{"id": tc.get("id"), "type": "function", "name": tc.get("name"), "arguments": json.dumps(tc.get("arguments") or {})} for tc in m["tool_calls"]]
            nodes.append({"message": msg, "sampled": True})
        elif role == "user":
            if m.get("tool_calls"):
                continue  # user-side tool call, not visible to the agent
            nodes.append({"message": {"role": "user", "content": m.get("content")}, "sampled": False})
        elif role == "tool":
            if m.get("requestor") == "user":
                continue
            nodes.append({"message": {"role": "tool", "tool_call_id": m.get("id"), "content": m.get("content"), "error": bool(m.get("error"))}, "sampled": False})
    return nodes


def convert(results: dict, task_records: dict[str, dict], label: str) -> tuple[list[dict], dict]:
    rows = []; scores = []
    for sim in results.get("simulations", []):
        rec = task_records.get(sim["task_id"])
        if rec is None:
            continue
        nodes = sim_to_nodes(sim.get("messages") or [])
        reward = float((sim.get("reward_info") or {}).get("reward") or 0.0)
        reason = sim.get("termination_reason")
        ok = reason not in ERROR_REASONS
        scores.append(reward)
        trace = {
            "version": 1, "id": sim.get("id") or uuid.uuid4().hex, "nodes": nodes, "calls": sum(1 for n in nodes if n["sampled"]),
            "rewards": {"tau2_reward": {"score": reward, "weight": 1.0}}, "metrics": {},
            "info": {"tau2": {"simulation": {k: sim.get(k) for k in ("id", "task_id", "trial", "seed", "termination_reason", "duration", "agent_cost", "user_cost")},
                              "reward_info": sim.get("reward_info")}},
            "stop_condition": STOP_MAP.get(reason, reason), "ok": ok, "errors": [], "is_completed": ok,
        }
        rows.append({"id": uuid.uuid4().hex, "env": {"id": "tau2-bench"}, "task": rec,
                     "run": {"type": "eval", "id": label, "name": label}, "ok": ok, "errors": [], "traces": [trace]})
    n = len(rows)
    summary = {"n": n, "n_scored": n, "score": round(sum(scores) / max(1, n), 4), "binary": True, "label": label,
               "by_trial": {}}
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="tau2 runner Results JSON (its save_to path)")
    ap.add_argument("--tasks", required=True, help="the generated tasks.jsonl")
    ap.add_argument("--out", required=True, help="output traces.jsonl.gz")
    ap.add_argument("--label", default="teacher")
    a = ap.parse_args()
    results = json.loads(Path(a.results).read_text())
    recs = {}
    for l in open(a.tasks):
        if l.strip():
            r = json.loads(l); recs[r["data"]["id"]] = r
    rows, summary = convert(results, recs, a.label)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.out, "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path(a.out).with_name("summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary)); print(f"wrote {a.out} ({len(rows)} episodes)")


if __name__ == "__main__":
    main()
