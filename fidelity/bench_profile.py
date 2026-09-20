"""Extract the shape a generated set should match from a reference task set.

Produces intent weights, the faults-per-task histogram per intent, the persona mix, and the list of
(intent, fault composition, persona) triples to exclude so generated tasks never collide with the
reference set.

Two sources:
  --from-tau2   read tau2-bench's own telecom data (`tasks.json` + `split_tasks.json`). Self-contained:
                needs nothing but the clone, and excludes the split you name (default `base`).
  --traces      read rollout traces instead, which additionally records the reference model's pass rate
                per category so `fidelity/report.py` can compare against it.

Usage:
  python fidelity/bench_profile.py --from-tau2 --split base --out domains/telecom/bench_profile.json
  python fidelity/bench_profile.py --traces reference/traces.jsonl.gz --out domains/telecom/bench_profile.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.tau2_compat import TAU2_TELECOM_DATA  # noqa: E402
from fidelity.taxonomy import analyze, load, n_faults, persona_of, task_category  # noqa: E402

RE_ID = re.compile(r"^\[([a-z_]+)\](.*?)\[PERSONA:([A-Za-z_]+)\]")


def _shape(triples: list[tuple[str, list[str], str]]) -> dict:
    intents = Counter(); k_by_intent: dict[str, Counter] = defaultdict(Counter); personas = Counter()
    for intent, faults, persona in triples:
        intents[intent] += 1; k_by_intent[intent][len(faults)] += 1; personas[persona] += 1
    return {
        "intent_weights": dict(intents),
        "n_faults_by_intent": {i: {str(k): v for k, v in sorted(c.items())} for i, c in k_by_intent.items()},
        "persona_mix": dict(personas),
        "exclude_compositions": [[i, sorted(f), p] for i, f, p in triples],
    }


def _triple(task_id: str) -> tuple[str, list[str], str] | None:
    m = RE_ID.match(task_id)
    return (m.group(1), m.group(2).split("|"), m.group(3)) if m else None


def profile_from_tau2(split: str = "base", domain: str = "telecom") -> dict:
    """Read tau2-bench's own telecom task set and the named split. Needs nothing but the clone."""
    tasks = json.loads((TAU2_TELECOM_DATA / "tasks.json").read_text())
    split_file = TAU2_TELECOM_DATA / "split_tasks.json"
    ids = set(json.loads(split_file.read_text())[split]) if split_file.exists() else {t["id"] for t in tasks}
    triples = [t for t in (_triple(x["id"]) for x in tasks if x["id"] in ids) if t]
    nact = [len((x.get("evaluation_criteria") or {}).get("actions") or []) for x in tasks if x["id"] in ids]
    unfix = sum(1 for x in tasks if x["id"] in ids
                and any(a.get("name") == "transfer_to_human_agents" for a in (x.get("evaluation_criteria") or {}).get("actions") or []))
    R = _shape(triples)
    R.update({
        "source": f"tau2-bench {domain} tasks.json, split={split}", "domain": domain, "n_tasks": len(triples),
        "tasks": {"n_unique_tasks": len(triples), "unfixable_tasks": unfix,
                  "expected_actions": {"min": min(nact), "median": sorted(nact)[len(nact) // 2], "max": max(nact)} if nact else None},
    })
    return R


def profile(rows: list[dict], domain: str = "telecom") -> dict:
    """Read rollout traces: same shape, plus the reference model's pass rate per category."""
    triples = [t for t in (_triple(r["task"]["data"]["name"]) for r in rows) if t]
    A = analyze(rows)
    R = _shape(triples)
    R.update({
        "source": "reference rollout traces", "domain": domain, "n_tasks": len(rows),
        "teacher": {"overall": A["success"]["overall"], "by_category": A["success"]["by_category"]},
        "rollouts": A["rollouts"], "tasks": A["tasks"],
    })
    return R


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-tau2", action="store_true", help="read tau2-bench's own task set (self-contained)")
    src.add_argument("--traces", nargs="+", help="read rollout traces instead")
    ap.add_argument("--split", default="base", help="task split to treat as held out (default: base)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.from_tau2:
        P = profile_from_tau2(a.split)
    else:
        P = profile([r for f in a.traces for r in load(f)])
        P["traces"] = [str(Path(f)) for f in a.traces]
    Path(a.out).write_text(json.dumps(P, indent=1, ensure_ascii=False))
    print(f"source={P['source']} intents={P['intent_weights']} personas={P['persona_mix']} exclude={len(P['exclude_compositions'])}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
