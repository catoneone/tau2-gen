"""Mid-conversation entry points: cut a rollout short and make the remainder a new task.

Each cut becomes a task carrying `initial_state.message_history`, so the model is dropped straight into a
decision point instead of always starting from "hello". Cuts are taken after a user message, because tau2
requires the history to end with a user or assistant turn.

  id_given           after the user first states the phone number from `known_info`
  after_lookup       after the lookup chain finished, the agent sent prose, and the user replied
  after_failed_step  after the user first reports that a step did not help
  after_frustration  after the user first expresses frustration
  after_wrong_info   after the first user message following a failed tool call

The new task keeps the original `evaluation_criteria` and initialisation; only the id gains an
`[EP:<kind>@<node>]` suffix.

Traces from a benchmark you want to keep held out can only be inspected with `--dry-run`, which reports
how many entry points *would* be harvested without writing any task.

Usage:
  python common/entrypoints.py --tasks out/telecom/tasks.jsonl --traces runs/teacher_k3/traces.jsonl.gz \
      --out out/telecom/tasks.entrypoints.jsonl
  python common/entrypoints.py --traces some/heldout/traces.jsonl.gz --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import schema  # noqa: E402
from fidelity.taxonomy import load  # noqa: E402

LOOKUPS = {"get_customer_by_phone", "get_customer_by_id", "get_details_by_id", "get_customer_by_name"}
RE_FAIL = re.compile(r"\b(still|not working|didn'?t work|doesn'?t work|no change|nothing changed|same (issue|problem)|hasn'?t (changed|helped)|no luck)\b", re.I)
RE_FRUST = re.compile(r"\b(frustrat|annoy|ridiculous|seriously|come on|this is getting|wasting|fed up|unbelievable|sigh|ugh)\b", re.I)
RE_PHONE = re.compile(r"\d{3}-\d{3}-\d{4}")


def _tool_error(node: dict) -> bool:
    c = (node["message"].get("content") or "").lower()
    return node["message"].get("role") == "tool" and (node["message"].get("error") or "error" in c[:80] or "not found" in c[:120])


def cut_points(nodes: list[dict], phone: str | None) -> dict[str, int]:
    """Return {kind: node index of the cut, inclusive}. nodes[0] is the system message."""
    cuts: dict[str, int] = {}
    seen_lookup = False; agent_text_after_lookup = False; tool_err = False
    for i, n in enumerate(nodes):
        m = n["message"]; role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            if any(tc.get("name") in LOOKUPS for tc in m["tool_calls"]):
                seen_lookup = True
        elif role == "assistant" and seen_lookup and (m.get("content") or "").strip():
            agent_text_after_lookup = True
        elif role == "tool" and _tool_error(n):
            tool_err = True
        elif role == "user":
            c = m.get("content") or ""
            if "id_given" not in cuts and phone and phone in c:
                cuts["id_given"] = i
            if "after_lookup" not in cuts and agent_text_after_lookup:
                cuts["after_lookup"] = i
            if "after_failed_step" not in cuts and RE_FAIL.search(c) and i > 4:
                cuts["after_failed_step"] = i
            if "after_frustration" not in cuts and RE_FRUST.search(c):
                cuts["after_frustration"] = i
            if "after_wrong_info" not in cuts and tool_err:
                cuts["after_wrong_info"] = i
    return cuts


def to_history(nodes: list[dict], end: int) -> list[dict]:
    """nodes[1..end] -> list of tau2 Message dicts (system skipped)."""
    hist = []
    for n in nodes[1:end + 1]:
        m = n["message"]; role = m.get("role")
        if role == "assistant":
            tcs = None
            if m.get("tool_calls"):
                tcs = []
                for tc in m["tool_calls"]:
                    args = tc.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"_raw": args}
                    tcs.append({"id": tc.get("id") or "", "name": tc.get("name"), "arguments": args or {}, "requestor": "assistant"})
            hist.append({"role": "assistant", "content": m.get("content"), "tool_calls": tcs})
        elif role == "user":
            hist.append({"role": "user", "content": m.get("content")})
        elif role == "tool":
            hist.append({"role": "tool", "id": m.get("tool_call_id") or "", "content": m.get("content"), "requestor": "assistant", "error": bool(m.get("error"))})
    return hist


def derive(records_by_key: dict[str, dict], rows: list[dict], max_per_task: int, kinds: set[str] | None) -> tuple[list[dict], Counter]:
    out, stats = [], Counter()
    per_task = Counter()
    for r in rows:
        key = r["task"].get("key") or r["task"]["data"]["id"]
        rec = records_by_key.get(key)
        if rec is None:
            stats["trace_task_not_in_set"] += 1; continue
        data = rec["data"]; nodes = r["traces"][0]["nodes"]
        phone_m = RE_PHONE.search(((data.get("user_scenario") or {}).get("instructions") or {}).get("known_info") or "")
        cuts = cut_points(nodes, phone_m.group(0) if phone_m else None)
        for kind, i in sorted(cuts.items(), key=lambda kv: kv[1]):
            if kinds and kind not in kinds:
                continue
            if per_task[key] >= max_per_task:
                stats["capped"] += 1; continue
            d = json.loads(json.dumps(data))
            d["id"] = d["name"] = f"{data['id']}[EP:{kind}@{i}]"
            d["initial_state"]["message_history"] = to_history(nodes, i)
            out.append(schema.episode_record(d)); per_task[key] += 1; stats[kind] += 1
    stats["tasks_with_entrypoints"] = len(per_task)
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default=None, help="generated tasks.jsonl (omit with --dry-run)")
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-per-task", type=int, default=3)
    ap.add_argument("--kinds", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true", help="count cut points only, write nothing (required for held-out traces)")
    a = ap.parse_args()
    rows = [r for f in a.traces for r in load(f)]
    if not a.dry_run:
        if not a.tasks or not a.out:
            sys.exit("--tasks and --out are required (or use --dry-run)")
        if any("[GEN:" not in r["task"]["data"]["id"] for r in rows):
            sys.exit("traces contain tasks that are not from this generator (no [GEN:] in the id); use --dry-run")
        records = {r["key"]: r for r in (json.loads(l) for l in open(a.tasks) if l.strip())}
    else:
        records = {(r["task"].get("key") or r["task"]["data"]["id"]): {"data": r["task"]["data"]} for r in rows}
    out, stats = derive(records, rows, a.max_per_task, set(a.kinds) if a.kinds else None)
    print(json.dumps(stats, ensure_ascii=False))
    if a.dry_run:
        print(f"dry-run: {len(out)} entrypoint tasks would be derived from {len(rows)} rollouts (not written)")
        return
    with open(a.out, "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {a.out} ({len(out)} tasks)")


if __name__ == "__main__":
    main()
