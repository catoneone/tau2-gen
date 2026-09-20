"""Solvability filter: aggregate per-task pass/fail over k reference-model rollouts.

Keeps a task when at least one reference rollout (or one optional frontier-model rollout) solved it, and
drops tasks nobody solved, which are usually unsolvable or a stuck user simulator.

`--drop-always-pass` additionally removes tasks the reference model solved k out of k times; only
meaningful at k >= 3, and only wanted if you are deliberately shaping difficulty.

Usage:
  python common/solvability.py --tasks out/telecom/tasks.jsonl \
      --teacher-traces runs/teacher_k3/traces.jsonl.gz [--frontier-traces runs/frontier.jsonl.gz] \
      [--drop-always-pass] --out out/telecom/tasks.solvable.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fidelity.taxonomy import load, reward, task_category  # noqa: E402


def pass_rates(rows: list[dict]) -> dict[str, tuple[int, int]]:
    by: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        k = r["task"].get("key") or r["task"]["data"]["id"]
        rw = reward(r["traces"][0])
        by[k].append(bool(rw is not None and rw >= 1))
    return {k: (sum(v), len(v)) for k, v in by.items()}


def filter_tasks(records: list[dict], teacher: dict[str, tuple[int, int]], frontier: dict[str, tuple[int, int]] | None,
                 drop_always_pass: bool) -> tuple[list[dict], dict]:
    kept, stats = [], Counter()
    by_intent: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        key = r["key"]; intent = task_category(r["data"])
        tp, tk = teacher.get(key, (0, 0)); fp, fk = (frontier or {}).get(key, (0, 0))
        if tk == 0 and fk == 0:
            stats["no_rollouts"] += 1; by_intent[intent]["no_rollouts"] += 1; continue
        if tp == 0 and fp == 0:
            stats["dropped_unsolvable"] += 1; by_intent[intent]["dropped_unsolvable"] += 1; continue
        if drop_always_pass and tk >= 3 and tp == tk:
            stats["dropped_always_pass"] += 1; by_intent[intent]["dropped_always_pass"] += 1; continue
        if tk >= 3 and tp == tk:
            stats["kept_always_pass"] += 1
        if tp == 0 and fp > 0:
            stats["kept_frontier_only"] += 1
        kept.append(r); stats["kept"] += 1; by_intent[intent]["kept"] += 1
    stats["by_intent"] = {k: dict(v) for k, v in by_intent.items()}
    return kept, dict(stats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--teacher-traces", nargs="+", required=True, help="reference-model rollouts, k files or one file with k trials")
    ap.add_argument("--frontier-traces", nargs="*", default=None)
    ap.add_argument("--drop-always-pass", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    records = [json.loads(l) for l in open(a.tasks) if l.strip()]
    teacher = pass_rates([r for f in a.teacher_traces for r in load(f)])
    frontier = pass_rates([r for f in a.frontier_traces for r in load(f)]) if a.frontier_traces else None
    kept, stats = filter_tasks(records, teacher, frontier, a.drop_always_pass)
    with open(a.out, "w") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path(a.out).with_suffix(".summary.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(json.dumps(stats, ensure_ascii=False)); print(f"wrote {a.out} ({len(kept)} tasks)")


if __name__ == "__main__":
    main()
