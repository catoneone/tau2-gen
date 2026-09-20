"""Fidelity report: a generated set against a reference set.

Task level (needs only tasks.jsonl): intent / persona / faults-per-task mix, expected-action counts,
share of tasks whose correct answer is to escalate rather than act, reward basis, assertion kinds.

Rollout level (needs reference-model traces over the generated set; pass k files for k rollouts):
  1. pass rate within [0.6, 0.85], and no category of 5+ tasks sitting at exactly 0 or 1
  2. msg#1 uniqueness >= 0.78 and pairwise Jaccard <= 0.25, message-class top-3 matching the reference
  3. reference usability at k >= 2: >= 0.9 of tasks with two usable rollouts, >= 0.8 with non-identical ones
  4. leakage, via fidelity/leakage.py

Reference columns are filled from --bench-traces when given; without it they read "-" and only the
thresholds are applied.

Usage:
  python fidelity/report.py --gen-tasks out/telecom/tasks.jsonl [--gen-traces runs/teacher_k3/traces.jsonl.gz] \
      [--bench-traces reference/traces.jsonl.gz] [--out out/telecom/fidelity_report.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fidelity import leakage  # noqa: E402
from fidelity.taxonomy import analyze, load, n_faults, persona_of, sampled, task_category  # noqa: E402

# Acceptance thresholds. Reference values themselves come from --bench-traces; nothing is hardcoded.
DASH = "-"
THRESH = {"teacher_lo": 0.6, "teacher_hi": 0.85, "msg1_unique": 0.78, "msg1_jaccard": 0.25, "ref_valid2": 0.9, "ref_not_identical": 0.8}


def pct(c: Counter, n: int) -> dict:
    return {k: f"{v} ({100 * v / max(1, n):.0f}%)" for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))}


def task_stats(datas: list[dict]) -> dict:
    n = len(datas)
    intents = Counter(task_category(d) for d in datas)
    personas = Counter(persona_of(d) for d in datas)
    kf = Counter(n_faults(d.get("name")) for d in datas)
    nact = [len((d.get("evaluation_criteria") or {}).get("actions") or []) for d in datas]
    unfix = sum(1 for d in datas if any(a.get("name") == "transfer_to_human_agents" for a in (d.get("evaluation_criteria") or {}).get("actions") or []))
    basis = Counter("|".join((d.get("evaluation_criteria") or {}).get("reward_basis") or []) for d in datas)
    afn = Counter(a["func_name"] for d in datas for a in (d.get("evaluation_criteria") or {}).get("env_assertions") or [])
    ifn = Counter(a["func_name"] for d in datas for a in (d.get("initial_state") or {}).get("initialization_actions") or [])
    with_db = sum(1 for d in datas if ((d.get("initial_state") or {}).get("initialization_data") or {}).get("agent_data"))
    return {"n": n, "intents": pct(intents, n), "personas": pct(personas, n), "n_faults": pct(kf, n),
            "expected_actions_min_med_max": (min(nact), sorted(nact)[len(nact) // 2], max(nact)) if nact else None,
            "unfixable": f"{unfix} ({100 * unfix / max(1, n):.0f}%)", "reward_basis": dict(basis),
            "assertion_funcs": dict(afn.most_common()), "init_funcs": dict(ifn.most_common()), "with_initialization_data": with_db}


def ref_usability(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["task"].get("key") or r["task"]["data"]["id"]].append(r)
    valid2 = 0; not_ident = 0; n2 = 0
    for k, rs in by.items():
        ok = [r for r in rs if r.get("ok", True) and any(n.get("sampled") for n in r["traces"][0]["nodes"])]
        if len(ok) >= 2:
            valid2 += 1; n2 += 1
            firsts = set()
            for r in ok:
                s = [m.get("content") or "" for m in sampled(r["traces"][0]) if not m.get("tool_calls")]
                firsts.add(s[0] if s else "")
            not_ident += len(firsts) > 1
    return {"tasks": len(by), "k_median": sorted(len(v) for v in by.values())[len(by) // 2] if by else 0,
            "valid_ge2": round(valid2 / max(1, len(by)), 3), "not_identical": round(not_ident / max(1, n2), 3) if n2 else None}


def row(name, gen, bench, thr, ok):
    mark = "n.a." if ok is None else ("PASS" if ok else "FAIL")
    return f"| {name} | {gen} | {bench} | {thr} | {mark} |"


def build_report(gen_records: list[dict], bench_rows: list[dict] | None, gen_rows: list[dict] | None, bench_traces: list[str] | None) -> tuple[str, bool]:
    L = ["# tau2-gen telecom - fidelity report", ""]
    gen_datas = [r["data"] for r in gen_records]
    gs = task_stats(gen_datas)
    bs = task_stats([r["task"]["data"] for r in bench_rows]) if bench_rows else None
    L += ["## Task level", "", "| metric | generated | reference |", "|---|---|---|"]
    for k in ("n", "intents", "personas", "n_faults", "expected_actions_min_med_max", "unfixable", "reward_basis", "assertion_funcs", "init_funcs", "with_initialization_data"):
        L.append(f"| {k} | {gs[k]} | {bs[k] if bs else DASH} |")
    L.append("")
    all_ok = True
    L += ["## Leakage", ""]
    lk = leakage.check(gen_records, bench_traces)
    all_ok &= lk["ok"]
    counts = {k: len(v) for k, v in lk["overlap"].items()}
    L.append(f"- Result: **{'PASS' if lk['ok'] else 'FAIL'}**. Identifier overlap {counts}. Held-out task-id overlap {len(lk['task_id_overlap'])} (out of {lk['heldout_task_ids']} held-out tasks).")
    L.append(f"- Informational: {lk['full_enumeration_id_overlap']} tasks share a (composition, persona) with tau2's full enumeration of {lk['full_enumeration_ids']}. Those are not held out, so they are allowed.")
    L.append("")
    L += ["## Rollout level", ""]
    if not gen_rows:
        L.append("_No --gen-traces given: run a reference model over the set first (see README, \"Reference rollouts\"). The table below only shows the reference side._")
        L.append("")
        B = analyze(bench_rows) if bench_rows else None
        L += ["| metric | generated | reference | threshold | verdict |", "|---|---|---|---|---|"]
        b_over = B["success"]["overall"] if B else DASH
        L.append(row("reference pass rate", DASH, b_over, f"[{THRESH['teacher_lo']}, {THRESH['teacher_hi']}]", None))
        if B:
            for c, v in B["success"]["by_category"].items():
                L.append(row(f"  category {c}", DASH, f"{v['rate']} (n={v['n']})", "not 0 or 1 (n>=5)", None))
            dv = B["rollouts"]["diversity"]
            L.append(row("msg#1 uniqueness", DASH, dv["agent_msg1"]["unique"], f">= {THRESH['msg1_unique']}", None))
            L.append(row("msg#1 Jaccard", DASH, dv["agent_msg1"]["jaccard_mean"], f"<= {THRESH['msg1_jaccard']}", None))
            L.append(row("message-class top-3", DASH, list(B["rollouts"]["message_classes"])[:3], "same set", None))
            L.append(row("messages per rollout, median", DASH, B["rollouts"]["messages_median"], "same order", None))
        L.append(row("tasks with >=2 usable rollouts", DASH, DASH, f">= {THRESH['ref_valid2']}", None))
        L.append(row("tasks with non-identical rollouts", DASH, DASH, f">= {THRESH['ref_not_identical']}", None))
        return "\n".join(L) + "\n", all_ok
    G = analyze(gen_rows); B = analyze(bench_rows) if bench_rows else None
    L += ["| metric | generated | reference | threshold | verdict |", "|---|---|---|---|---|"]
    g_over = G["success"]["overall"]; b_over = B["success"]["overall"] if B else DASH
    ok = THRESH["teacher_lo"] <= g_over <= THRESH["teacher_hi"]; all_ok &= ok
    L.append(row("reference pass rate", g_over, b_over, f"[{THRESH['teacher_lo']}, {THRESH['teacher_hi']}]", ok))
    for c, v in G["success"]["by_category"].items():
        bv = (B["success"]["by_category"].get(c) or {}).get("rate") if B else DASH
        ok_c = (v["n"] < 5) or (0 < v["rate"] < 1)
        all_ok &= ok_c
        L.append(row(f"  category {c}", f"{v['rate']} (n={v['n']})", bv, "not 0 or 1 (n>=5)", ok_c if v["n"] >= 5 else None))
    dv = G["rollouts"]["diversity"]; bdv = B["rollouts"]["diversity"] if B else None
    u1 = dv["agent_msg1"]["unique"]; j1 = dv["agent_msg1"]["jaccard_mean"]
    ok = u1 >= THRESH["msg1_unique"]; all_ok &= ok
    L.append(row("msg#1 uniqueness", u1, bdv["agent_msg1"]["unique"] if bdv else DASH, f">= {THRESH['msg1_unique']}", ok))
    ok = j1 is not None and j1 <= THRESH["msg1_jaccard"]; all_ok &= ok
    L.append(row("msg#1 Jaccard", j1, bdv["agent_msg1"]["jaccard_mean"] if bdv else DASH, f"<= {THRESH['msg1_jaccard']}", ok))
    L.append(row("msg#0 uniqueness (info)", dv["agent_msg0"]["unique"], bdv["agent_msg0"]["unique"] if bdv else DASH, DASH, None))
    L.append(row("user opening uniqueness (info)", dv["user_opening"]["unique"], bdv["user_opening"]["unique"] if bdv else DASH, DASH, None))
    gtop = list(G["rollouts"]["message_classes"])[:3]; btop = list(B["rollouts"]["message_classes"])[:3] if B else None
    ok = (btop is None) or set(gtop) == set(btop); all_ok &= ok
    L.append(row("message-class top-3", gtop, btop or DASH, "same set", ok if btop else None))
    L.append(row("message-class mix", G["rollouts"]["message_classes"], B["rollouts"]["message_classes"] if B else DASH, DASH, None))
    L.append(row("messages per rollout, median", G["rollouts"]["messages_median"], B["rollouts"]["messages_median"] if B else DASH, "same order", None))
    L.append(row("distinct first-3 tool sequences", G["rollouts"]["distinct_first3_tool_seqs"], B["rollouts"]["distinct_first3_tool_seqs"] if B else DASH, DASH, None))
    L.append(row("stop conditions", G["rollouts"]["stop_conditions"], B["rollouts"]["stop_conditions"] if B else DASH, DASH, None))
    ru = ref_usability(gen_rows)
    if ru["k_median"] >= 2:
        ok = ru["valid_ge2"] >= THRESH["ref_valid2"]; all_ok &= ok
        L.append(row("tasks with >=2 usable rollouts", ru["valid_ge2"], DASH, f">= {THRESH['ref_valid2']}", ok))
        ok = ru["not_identical"] is not None and ru["not_identical"] >= THRESH["ref_not_identical"]; all_ok &= ok
        L.append(row("tasks with non-identical rollouts", ru["not_identical"], DASH, f">= {THRESH['ref_not_identical']}", ok))
    else:
        L.append(row("reference usability", f"median k {ru['k_median']} < 2", DASH, "needs k >= 2", None))
    # Share of the generated set sitting in categories the reference model is weak on.
    if B:
        weak = {c for c, v in B["success"]["by_category"].items() if v["n"] >= 3 and v["rate"] < 0.5}
        share = sum(v["n"] for c, v in G["success"]["by_category"].items() if c in weak) / max(1, G["n_episodes"])
        L.append(row("share in reference-weak categories (<0.5)", round(share, 3), sorted(weak), "informational", None))
    L.append("")
    L.append(f"**Overall: {'PASS' if all_ok else 'FAIL'}**")
    return "\n".join(L) + "\n", all_ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-tasks", required=True)
    ap.add_argument("--gen-traces", nargs="*", default=None)
    ap.add_argument("--bench-traces", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    gen_records = leakage.load_jsonl(a.gen_tasks)
    bench_rows = [r for f in a.bench_traces for r in load(f)] if a.bench_traces else None
    gen_rows = [r for f in a.gen_traces for r in load(f)] if a.gen_traces else None
    md, ok = build_report(gen_records, bench_rows, gen_rows, a.bench_traces)
    print(md)
    if a.out:
        Path(a.out).write_text(md)
        print(f"wrote {a.out}")
    if gen_rows is not None:
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
