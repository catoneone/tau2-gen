"""Export a generated task set into Affine's `affine_tau2_v1` task-set format.

Affine (Bittensor SN120, https://github.com/AffineFoundation/affine) runs τ² through
`rollouts/envs/affine_tau2_v1`: a verifiers taskset whose records are τ²'s native `Task` fields plus a
few of its own (`idx`, `name`, `prompt`, `domain`, `tau_description`, `system_prompt`). Its harness hands
the record back to `tau2.run.run_task`, so per-task databases travel exactly as this generator emits
them: in `initial_state.initialization_data` (telecom needs `patches/0001` on the τ² side; see README).

This script turns `tasks_tau2.json` into that shape, validates every record against τ²'s `Task` model on
the way, and refuses to write anything whose id collides with a held-out τ² split (the benchmark tasks
Affine keeps out of its corpus), so the file can be dropped into the fold's intake as is.

Usage:
  upstream/tau2-bench/.venv/bin/python scripts/export_affine.py \
      --tasks out/telecom/tasks_tau2.json --domain telecom --out out/telecom/affine_tasks.json \
      [--manifest out/telecom/manifest.json] [--exclude-split base] [--name-prefix tau2g-]

Output: one JSON object
  {"schema": "affine_tau2_v1/Tau2Data@1", "domain": ..., "source": {...}, "n": N, "tasks": [record, ...]}
where each record is
  {"idx", "name", "prompt", "domain", "system_prompt", "tau_description", + every τ² Task field
   (id, description (str), user_scenario, initial_state, evaluation_criteria, annotations, ...)}.

`name` is `<prefix><τ² id>` (default prefix `tau2g-`; Affine addresses tasks by name because τ² ids start
with `[`, which its eval CLI would parse as JSON). `prompt` is τ²'s fixed first agent message.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.tau2_compat import TAU2_DOMAIN_DATA, repo_commit, require_tau2, tau2_commit  # noqa: E402

require_tau2()

SCHEMA = "affine_tau2_v1/Tau2Data@1"
FIRST_AGENT_MESSAGE = "Hi! How can I help you today?"


def load_task_list(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw["tasks"] if isinstance(raw, dict) and "tasks" in raw else raw


def held_out_ids(domain: str, split: str | None) -> set[str]:
    """Task ids of the τ² split Affine keeps out of its corpus (telecom `base` = the 114 benchmark tasks;
    airline / retail `base` = the whole card). Empty set when the domain has no split file."""
    if not split:
        return set()
    from tau2.run import load_tasks

    try:
        return {t.id for t in load_tasks(task_set_name=domain, task_split_name=split)}
    except Exception:
        # domains without split files (or an unknown split): fall back to the full task file
        tasks_file = TAU2_DOMAIN_DATA / domain / "tasks.json"
        if not tasks_file.exists():
            return set()
        return {t["id"] for t in load_task_list(tasks_file)}


def to_affine_record(task: dict, index: int, domain: str, prefix: str, system_prompt: str | None) -> dict:
    from tau2.data_model.tasks import Task

    model = Task.model_validate(task)
    rec = model.model_dump(mode="json", exclude={"description"})
    rec.update({
        "idx": index,
        "name": f"{prefix}{model.id}",
        "prompt": FIRST_AGENT_MESSAGE,
        "domain": domain,
        "description": str(model.description) if model.description else None,
        "tau_description": model.description.model_dump(mode="json") if model.description else None,
    })
    if system_prompt is not None:
        rec["system_prompt"] = system_prompt
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True, help="tasks_tau2.json from a generator run")
    ap.add_argument("--domain", default="telecom", choices=["telecom", "airline", "retail"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None, help="the run's manifest.json (seed, commits, leakage) — recorded in the export")
    ap.add_argument("--exclude-split", default="base",
                    help="τ² split whose ids must not appear in the export (default base = the benchmark tasks); '' to skip")
    ap.add_argument("--name-prefix", default="tau2g-")
    ap.add_argument("--system-prompt", default=None, help="optional system message stored on every record")
    a = ap.parse_args()

    tasks = load_task_list(Path(a.tasks))
    held = held_out_ids(a.domain, a.exclude_split or None)
    collisions = [t["id"] for t in tasks if t["id"] in held]
    if collisions:
        raise SystemExit(f"refusing to export: {len(collisions)} task id(s) collide with τ² {a.domain}/{a.exclude_split}: "
                         f"{collisions[:3]}...")
    without_db = sum(1 for t in tasks if not ((t.get("initial_state") or {}).get("initialization_data")))
    records = [to_affine_record(t, i, a.domain, a.name_prefix, a.system_prompt) for i, t in enumerate(tasks)]
    manifest = json.loads(Path(a.manifest).read_text()) if a.manifest else None
    out = {
        "schema": SCHEMA,
        "domain": a.domain,
        "source": {
            "generator": "catoneone/tau2-gen", "tau2_gen_commit": repo_commit(), "tau2_bench_commit": tau2_commit(),
            "tasks_file": str(a.tasks), "manifest": manifest,
            "held_out_split": a.exclude_split or None, "held_out_ids_checked": len(held),
        },
        "n": len(records),
        "n_without_initialization_data": without_db,
        "note": ("telecom records with initialization_data need tau2-bench patched with patches/0001 "
                 "(set_state must keep the separate user DB) on the consumer side"),
        "tasks": records,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=None), encoding="utf-8")
    print(f"wrote {a.out}: {len(records)} {a.domain} records, {without_db} without initialization_data, "
          f"0 collisions against {len(held)} held-out ids ({a.exclude_split or 'no split'})")


if __name__ == "__main__":
    main()
