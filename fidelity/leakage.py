"""Leakage check: generated ids, phone numbers, emails, names and IMEIs must not intersect the reference set,
and no generated task id (minus its `[GEN:...]` suffix) may equal a held-out task id.

Reference identifiers come from the tau2-bench clone (`db.toml`, `user_db.toml`, `tasks.json`) plus any
trace files you pass. Held-out task ids are the split named in `--split` (default `base`) plus every task
appearing in those traces.

Usage: python fidelity/leakage.py --gen out/telecom/tasks.jsonl [--bench-traces ...] [--split base]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.db import identifiers  # noqa: E402
from common.tau2_compat import TAU2_TELECOM_DATA  # noqa: E402

RE_GEN_SUFFIX = re.compile(r"\[GEN:[^\]]*\]$")


def load_jsonl(p: str | Path) -> list[dict]:
    import gzip

    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def benchmark_identifiers(bench_traces: list[str] | None = None) -> tuple[dict[str, set], set[str]]:
    import toml

    objs = []
    for fn in ("db.toml", "user_db.toml"):
        p = TAU2_TELECOM_DATA / fn
        if p.exists():
            objs.append(toml.load(p))
    task_ids: set[str] = set()  # held out: the named split plus every task seen in the traces
    full_ids: set[str] = set()  # informational: tau2's full programmatic enumeration
    tj = TAU2_TELECOM_DATA / "tasks.json"
    if tj.exists():
        tasks = json.loads(tj.read_text())
        objs.append(tasks)
        full_ids.update(t["id"] for t in tasks)
    sp = TAU2_TELECOM_DATA / "split_tasks.json"
    if sp.exists():
        task_ids.update(json.loads(sp.read_text()).get("base", []))
    for f in bench_traces or []:
        for r in load_jsonl(f):
            objs.append(r["task"]); task_ids.add(r["task"]["data"]["id"])
    ids = identifiers(objs)
    ids["_full_ids"] = full_ids
    return ids, task_ids


def check(gen_records: list[dict], bench_traces: list[str] | None = None, extra: dict | None = None) -> dict:
    """extra: the shared db.toml contents in --db-mode shared, where tasks carry no DB of their own."""
    gen_ids = identifiers([r["data"] for r in gen_records] + ([extra] if extra else []))
    bench_ids, bench_task_ids = benchmark_identifiers(bench_traces)
    full_ids = bench_ids.pop("_full_ids")
    overlap = {k: sorted(gen_ids[k] & bench_ids[k]) for k in gen_ids}
    gen_task_ids = {RE_GEN_SUFFIX.sub("", r["data"]["id"]) for r in gen_records}
    id_overlap = sorted(gen_task_ids & bench_task_ids)
    ok = not any(overlap.values()) and not id_overlap
    return {
        "ok": ok,
        "overlap": overlap,
        "task_id_overlap": id_overlap,  # same id as a held-out task -> hard failure
        "full_enumeration_id_overlap": len(gen_task_ids & full_ids),  # same (composition, persona) as some task in the full enumeration; informational only
        "gen_counts": {k: len(v) for k, v in gen_ids.items()},
        "bench_counts": {k: len(v) for k, v in bench_ids.items()},
        "heldout_task_ids": len(bench_task_ids),
        "full_enumeration_ids": len(full_ids),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", required=True, help="generated tasks.jsonl (one {type,data,key,hash} per line)")
    ap.add_argument("--bench-traces", nargs="*", default=None)
    a = ap.parse_args()
    R = check(load_jsonl(a.gen), a.bench_traces)
    print(json.dumps(R, indent=1, ensure_ascii=False))
    sys.exit(0 if R["ok"] else 1)


if __name__ == "__main__":
    main()
