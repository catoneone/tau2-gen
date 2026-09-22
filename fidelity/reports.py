"""Two reports emitted beside `fidelity_report.md`, so a task set carries its own provenance.

`guards_report.md` says what the build-time invariants refused and what the anchor check saw.
`composition_report.md` says how many distinct decisions the set actually contains, and how that grows
with n.

Both are written from the manifest and `meta.jsonl` that generation already produces, so they can be
regenerated for an existing set without rebuilding it.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


def kind(m: dict) -> str:
    """What a task is, in whichever field the domain records it.

    Airline and retail name a case; telecom composes atomic faults and names an intent instead."""
    return m.get("case") or m.get("intent") or "task"


def _table(rows: list[tuple], head: tuple) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def write_guards_report(out_dir: Path, manifest: dict, metas: list[dict]) -> Path:
    """What the build-time guards refused, per guard and per case.

    A guard that never fires is either redundant or broken; a guard that fires constantly is a case
    whose sampling is wrong. Neither is visible from a single kept/attempted ratio, which is why this
    is reported per guard rather than as one number."""
    stats = manifest.get("stats") or {}
    kept = stats.get("kept", len(metas))
    guard_rows, case_rows = [], []
    by_guard: Counter = Counter()
    for k, v in stats.items():
        if k.startswith("guard_reject:"):
            _, guard, case = k.split(":", 2)
            guard_rows.append((guard, case, v))
            by_guard[guard] += v
        elif k.startswith("build_fail:"):
            case_rows.append((k.split(":", 1)[1], v))
    verify_rows = [(k.split(":", 1)[1], v) for k, v in stats.items() if k.startswith("verify_fail:")]
    rejected = sum(v for _, v in case_rows)

    L = [f"# Build-time guards — {manifest.get('generator', out_dir.name)}",
         "",
         f"seed {manifest.get('seed')}, kept {kept}, rejected during sampling {rejected} "
         f"({rejected / max(1, kept + rejected):.1%} of attempts)",
         "",
         "A rejected candidate is discarded and the slot resampled, so rejections do not change the",
         "mix; they are reported because an unsolvable or ambiguous task that reaches a training set",
         "is worse than a missing one — it scores a correct agent zero.",
         ""]
    if guard_rows:
        L += ["## By guard", ""]
        L += _table(sorted(guard_rows, key=lambda r: (-r[2], r[0])), ("guard", "case", "rejected"))
        L += ["", "totals: " + ", ".join(f"`{g}` {n}" for g, n in by_guard.most_common()), ""]
    if case_rows:
        L += ["## By case", ""]
        L += _table(sorted(case_rows, key=lambda r: -r[1]), ("case", "rejected"))
        L += [""]
    if verify_rows:
        L += ["## Rejected by executing the task against the real environment", ""]
        L += _table(sorted(verify_rows, key=lambda r: -r[1]), ("case", "rejected"))
        L += [""]

    # The anchor check runs on every kept task, so its outcome is a property of the set, not of the
    # rejections: every kept task's must-mention strings appear in the request the customer makes.
    anchored = [m for m in metas if m.get("communicate_info")]
    missing = [m for m in metas if not m.get("communicate_info")]
    per_case: dict[str, Counter] = {}
    for m in metas:
        per_case.setdefault(kind(m), Counter())["with" if m.get("communicate_info") else "without"] += 1
    L += ["## Topic-anchor check", "",
          f"{len(anchored)}/{len(metas)} kept tasks carry must-mention strings; "
          f"{len(missing)} carry none.", "",
          "An empty `communicate_info` scores 1.0 without asking the judge anything, so a refusal with",
          "no anchor passes on silence. Every anchor is checked at build time against",
          "`reason_for_call` only — a word the customer never says is a phrasing trap, not a test.", ""]
    L += _table(sorted(((c, v.get("with", 0), v.get("without", 0)) for c, v in per_case.items()),
                       key=lambda r: (r[2] == 0, -r[1])),
                ("case", "with anchors", "without"))
    if missing:
        L += ["", f"**{len(missing)} task(s) carry no anchors.** Cases: "
              + ", ".join(sorted({kind(m) for m in missing}))]
    L += [""]
    p = out_dir / "guards_report.md"
    p.write_text("\n".join(L))
    return p


def fingerprint(m: dict) -> tuple:
    """What decision a task actually asks for: the rules that fired and the writes they imply.

    Two tasks with the same fingerprint differ only in which customer and which record they used, which
    is the difference that does not teach anything new."""
    if m.get("units"):
        shape = tuple(sorted(m["units"]))
    elif m.get("families"):
        # telecom's decision is the set of faults it composed, which is where its variety comes from
        shape = tuple(sorted(m["families"]))
    else:
        shape = (kind(m),)
    return (shape, tuple(m.get("rule_reasons") or ()), tuple(sorted(m.get("write_names") or ())))


def rarefaction(metas: list[dict], points: Iterable[int], trials: int = 40,
                seed: int = 0) -> list[tuple[int, float]]:
    """Mean distinct fingerprints in a random subset of size n, over `trials` draws."""
    rng = random.Random(seed)
    fps = [fingerprint(m) for m in metas]
    out = []
    for n in points:
        if n > len(fps):
            continue
        out.append((n, sum(len({*rng.sample(fps, n)}) for _ in range(trials)) / trials))
    return out


def write_composition_report(out_dir: Path, manifest: dict, metas: list[dict],
                             ceiling: Optional[int] = None,
                             ceiling_note: str = "") -> Path:
    """How many distinct decisions the set contains, and how that grows with n.

    One case per rule gives one decision per task and the set saturates: past the ceiling every extra
    task is the same decision with a different customer. The curve is what says whether a set of this
    size is still buying new decisions."""
    fps = Counter(fingerprint(m) for m in metas)
    n = len(metas)
    pts = [p for p in (10, 25, 50, 100, 200, 400, 800) if p <= n] + [n]
    curve = rarefaction(metas, sorted(set(pts)))
    L = [f"# Composition — {manifest.get('generator', out_dir.name)}", "",
         f"seed {manifest.get('seed')}, {n} tasks, **{len(fps)} distinct decisions**",
         "",
         "A decision is (what the task is made of, which rule outcomes fired, which writes follow).",
         "Two tasks with the same one differ only in the customer and the record.",
         ""]
    if ceiling:
        L += [f"ceiling: **{ceiling}** distinct decisions the rules can express"
              + (f" — {ceiling_note}" if ceiling_note else ""),
              f"coverage at n={n}: {len(fps)}/{ceiling} = {len(fps) / ceiling:.0%}", ""]
    L += ["## Distinct decisions by set size", ""]
    L += _table([(nn, f"{d:.1f}", f"{d / nn:.2f}") for nn, d in curve],
                ("n", "distinct (mean of 40 draws)", "per task"))
    L += ["", "A flat tail means more tasks of the same kind; a rising one means the set is still",
          "buying new decisions.", "", "## Most repeated decisions", ""]
    L += _table([(" + ".join(f[0]), ", ".join(f[1]) or "—", ", ".join(f[2]) or "no write", c)
                 for f, c in fps.most_common(12)],
                ("made of", "rule outcomes", "writes", "tasks"))
    L += ["", f"seen once: {sum(1 for c in fps.values() if c == 1)} of {len(fps)}", ""]
    p = out_dir / "composition_report.md"
    p.write_text("\n".join(L))
    return p


def write_all(out_dir: Path, ceiling: Optional[int] = None, ceiling_note: str = "") -> list[Path]:
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    metas = [json.loads(l) for l in (out_dir / "meta.jsonl").read_text().splitlines() if l.strip()]
    return [write_guards_report(out_dir, manifest, metas),
            write_composition_report(out_dir, manifest, metas, ceiling, ceiling_note)]


def summary(out_dir: Path) -> dict:
    """The numbers the two reports state, in a form a consumer can read without parsing markdown.

    The markdown is for a person; this is what the exporter carries, so a set's guard rejections and
    its composition travel with it instead of being scraped back out of `manifest.json.stats`."""
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    metas = [json.loads(l) for l in (out_dir / "meta.jsonl").read_text().splitlines() if l.strip()]
    stats = manifest.get("stats") or {}
    by_guard: Counter = Counter()
    by_guard_case: dict[str, int] = {}
    for k, v in stats.items():
        if k.startswith("guard_reject:"):
            _, guard, case = k.split(":", 2)
            by_guard[guard] += v
            by_guard_case[f"{guard}:{case}"] = v
    fps = Counter(fingerprint(m) for m in metas)
    n = len(metas)
    return {
        "guards": {
            "kept": stats.get("kept", n),
            "rejected": sum(v for k, v in stats.items() if k.startswith("build_fail:")),
            "by_guard": dict(by_guard.most_common()),
            "by_guard_and_case": by_guard_case,
            "verify_rejected": {k.split(":", 1)[1]: v for k, v in stats.items()
                                if k.startswith("verify_fail:")},
        },
        "anchors": {
            "tasks_with_communicate_info": sum(1 for m in metas if m.get("communicate_info")),
            "tasks_without": sum(1 for m in metas if not m.get("communicate_info")),
            "cases_without": sorted({kind(m) for m in metas if not m.get("communicate_info")}),
        },
        "composition": {
            "n": n,
            "distinct_decisions": len(fps),
            "seen_once": sum(1 for c in fps.values() if c == 1),
            "distinct_by_n": {str(nn): round(d, 1) for nn, d in
                              rarefaction(metas, [p for p in (10, 25, 50, 100, 200, 400, 800) if p <= n] + [n])},
        },
        "files": ["guards_report.md", "composition_report.md"],
    }
