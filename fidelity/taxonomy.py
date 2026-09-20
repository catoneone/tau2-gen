"""Profile any trace file: pass rate, message-class mix, msg#0/msg#1 uniqueness and Jaccard, tool-sequence variety.

Input is the flat JSONL episode schema (one episode per line, `traces[0]` the main rollout). Several files
are concatenated, which is how you pass k rollouts of the same task set.

Usage: python fidelity/taxonomy.py --traces a.jsonl.gz [b.jsonl.gz ...] [--label X] [--json out.json]
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load(f: str | Path) -> list[dict]:
    op = gzip.open if str(f).endswith(".gz") else open
    with op(f, "rt") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def reward(tr: dict):
    rw = tr.get("rewards") or {}
    sc = [v["score"] for v in rw.values() if isinstance(v, dict) and "score" in v]
    return sc[0] if sc else None


def sampled(tr: dict) -> list[dict]:
    return [n["message"] for n in tr["nodes"] if n.get("sampled")]


def toks(t: str) -> set[str]:
    return set(re.findall(r"[a-z']+", t.lower()))


def jac(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def ctype(t: str) -> str:
    t = t.lower()
    if re.search(r"(user id|phone number|customer id|email|full name|date of birth|order id|reservation id|confirmation number|zip code).*\?", t):
        return "ask_identity"
    if "transferred to a human" in t or "transfer you" in t:
        return "transfer"
    if re.search(r"(would you like|do you want|shall i|confirm|proceed|go ahead)", t) and "?" in t:
        return "confirm_ask"
    if re.search(r"(i'm unable|i cannot|not able to|can't|not allowed|unfortunately|not eligible|policy)", t):
        return "refuse_or_limit"
    if re.search(r"(can you|could you|please)\b.*(check|turn|toggle|go to|open|restart|tap|enable|disable|settings|try)", t):
        return "instruct_step"
    if "?" in t:
        return "clarify_other"
    return "inform_or_close"


def task_category(d: dict) -> str:
    """telecom: the intent in the task name. airline/retail: the dominant expected action."""
    a = (d.get("evaluation_criteria") or {}).get("actions") or []
    m = re.match(r"\[(.*?)\]", d.get("name") or "")
    if m:
        return m.group(1)
    return a[0].get("name") if a and isinstance(a[0], dict) else "no_action"


def n_faults(name: str) -> int | None:
    m = re.match(r"\[[^\]]*\](.*?)\[PERSONA", name or "")
    return len(m.group(1).split("|")) if m else None


def persona_of(d: dict) -> str:
    m = re.search(r"\[PERSONA:([A-Za-z_]+)\]", d.get("name") or "")
    if m:
        return m.group(1)
    p = (d.get("user_scenario") or {}).get("persona")
    return "None" if not p else p.strip()[:20]


def diversity(L: list[str], cap: int = 60) -> dict:
    Tk = [toks(x) for x in L]
    pairs = [jac(a, b) for a, b in itertools.combinations(Tk[:cap], 2)]
    return {
        "n": len(L),
        "unique": round(len(set(L)) / max(1, len(L)), 3),
        "jaccard_mean": round(st.mean(pairs), 3) if pairs else None,
    }


def analyze(rows: list[dict]) -> dict:
    """rows = episode records, one per line, `traces[0]` the main rollout."""
    R: dict = {"n_episodes": len(rows)}
    cat = Counter(); acts = Counter(); nact = []; succ = defaultdict(list); nl = 0; comm = 0
    personas = Counter(); kf = Counter(); zero = 0; unfix = 0; basis = Counter()
    by_task: dict[str, list] = defaultdict(list)
    for r in rows:
        d = r["task"]["data"]; ev = d.get("evaluation_criteria") or {}; a = ev.get("actions") or []
        nact.append(len(a)); zero += (len(a) == 0)
        names = [x.get("name") if isinstance(x, dict) else str(x)[:30] for x in a]
        acts.update(names); unfix += ("transfer_to_human_agents" in names)
        nl += bool(ev.get("nl_assertions")); comm += bool(ev.get("communicate_info"))
        basis[tuple(ev.get("reward_basis") or [])] += 1
        personas[persona_of(d)] += 1
        k = n_faults(d.get("name")); kf[k] += 1
        c = task_category(d); cat[c] += 1
        rw = reward(r["traces"][0]); succ[c].append(rw)
        by_task[r["task"].get("key") or d.get("id")].append(bool(rw and rw >= 1))
    allr = [x for v in succ.values() for x in v]
    R["tasks"] = {
        "n_unique_tasks": len(by_task),
        "expected_actions": {"min": min(nact), "median": sorted(nact)[len(nact) // 2], "max": max(nact)} if nact else None,
        "zero_action_tasks": zero, "unfixable_tasks": unfix, "nl_assertions": nl, "communicate_info": comm,
        "reward_basis": {"|".join(k): v for k, v in basis.items()},
        "expected_action_names": acts.most_common(15),
        "personas": dict(personas.most_common()),
        "n_faults_hist": {str(k): v for k, v in sorted(kf.items(), key=lambda kv: (kv[0] is None, kv[0]))},
        "categories": dict(cat.most_common()),
    }
    R["success"] = {
        "overall": round(sum(1 for x in allr if x and x >= 1) / max(1, len(allr)), 3),
        "by_category": {k: {"n": len(v), "rate": round(sum(1 for x in v if x and x >= 1) / len(v), 3)} for k, v in sorted(succ.items(), key=lambda kv: -len(kv[1]))},
        "per_task_pass_rate_hist": dict(Counter(round(sum(v) / len(v), 2) for v in by_task.values())),
    }
    cls = Counter(); pos = defaultdict(Counter); first = []; second = []; opens = []; seqs = Counter(); nmsg = []; ncall = []; stops = Counter()
    for r in rows:
        tr = r["traces"][0]; stops[tr.get("stop_condition")] += 1
        s = sampled(tr); msgs = [m.get("content") or "" for m in s if not m.get("tool_calls")]
        calls = [m["tool_calls"][0]["name"] for m in s if m.get("tool_calls")]
        nmsg.append(len(msgs)); ncall.append(len(calls)); seqs[tuple(calls[:3])] += 1
        for i, t in enumerate(msgs):
            c = ctype(t); cls[c] += 1; pos[min(i, 4)][c] += 1
        if msgs: first.append(msgs[0])
        if len(msgs) > 1: second.append(msgs[1])
        u = [n["message"].get("content") or "" for n in tr["nodes"] if n["message"].get("role") == "user"]
        if u: opens.append(u[0])
    R["rollouts"] = {
        "messages_median": st.median(nmsg) if nmsg else None,
        "tool_calls_median": st.median(ncall) if ncall else None,
        "stop_conditions": dict(stops),
        "message_classes": dict(cls.most_common()),
        "by_position": {f"msg#{p}{'+' if p == 4 else ''}": dict(pos[p].most_common(5)) for p in sorted(pos)},
        "diversity": {"agent_msg0": diversity(first), "agent_msg1": diversity(second), "user_opening": diversity(opens)},
        "distinct_first3_tool_seqs": len(seqs),
        "top_tool_seqs": [(list(k), v) for k, v in seqs.most_common(3)],
        "sample_msg0": first[0][:200].replace("\n", " ") if first else None,
    }
    return R


def fmt(R: dict, label: str = "") -> str:
    L = [f"################ {label}: {R['n_episodes']} episodes / {R['tasks']['n_unique_tasks']} tasks"]
    t = R["tasks"]
    L.append(f"expected actions/task min/med/max: {t['expected_actions']} | zero-action: {t['zero_action_tasks']} | unfixable(transfer): {t['unfixable_tasks']} | nl_assertions: {t['nl_assertions']} | communicate_info: {t['communicate_info']}")
    L.append(f"reward_basis: {t['reward_basis']}")
    L.append(f"expected action names: {t['expected_action_names'][:12]}")
    L.append(f"personas: {t['personas']}")
    L.append(f"n_faults hist: {t['n_faults_hist']}")
    L.append(f"categories: {t['categories']}")
    L.append("success by category:")
    for k, v in R["success"]["by_category"].items():
        if v["n"] >= 3:
            L.append(f"   {k:34s} n={v['n']:3d} success={v['rate']:.2f}")
    L.append(f"   overall success={R['success']['overall']:.3f} | per-task pass-rate hist={R['success']['per_task_pass_rate_hist']}")
    ro = R["rollouts"]
    L.append(f"per rollout messages median {ro['messages_median']} tool calls median {ro['tool_calls_median']} stops={ro['stop_conditions']}")
    L.append(f"message classes: {ro['message_classes']}")
    for p, v in ro["by_position"].items():
        L.append(f"   {p}: {v}")
    for k, v in ro["diversity"].items():
        L.append(f"   {k:22s} n={v['n']:3d} unique={v['unique']:.2f} Jaccard mean={v['jaccard_mean']}")
    L.append(f"distinct first-3 tool sequences: {ro['distinct_first3_tool_seqs']} top: {ro['top_tool_seqs'][:2]}")
    L.append(f"sample msg#0: {ro['sample_msg0']}")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = [r for f in a.traces for r in load(f)]
    R = analyze(rows)
    print(fmt(R, a.label or Path(a.traces[0]).parent.name))
    if a.json:
        Path(a.json).write_text(json.dumps(R, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
