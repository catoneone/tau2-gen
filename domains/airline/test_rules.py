"""Hold rules.yaml to tau2-bench's own 50 airline tasks.

The benchmark tasks are fixtures only. They are read here and never written into a generated set.

What can and cannot be asserted:

- Every reservation a benchmark task **cancels** must be eligible under the rules. The converse does
  not hold: a task may leave an eligible reservation alone because the user only asked about another
  one (task 42 cancels two of seven eligible business reservations).
- Every reservation a benchmark task **changes flights on** must be flight-changeable, so not basic
  economy and not already flown.
- Reservations with a flown segment must be refused by the engine, which is the one direction the
  policy states unconditionally.
- Any `send_certificate` amount must equal the engine's compensation amount.
- Baggage: where a task sets `total_baggages`, the engine's free allowance must not exceed it.

Usage: upstream/tau2-bench/.venv/bin/python domains/airline/test_rules.py [-v]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.tau2_compat import TAU2_ROOT  # noqa: E402
from domains.airline.rules import AirlineRules  # noqa: E402

AIRLINE_DATA = TAU2_ROOT / "data" / "tau2" / "domains" / "airline"
COVERED = {"health", "weather"}


def load():
    db = json.loads((AIRLINE_DATA / "db.json").read_text())
    tasks = json.loads((AIRLINE_DATA / "tasks.json").read_text())
    return db, tasks


def actions(task) -> list[dict]:
    return (task.get("evaluation_criteria") or {}).get("actions") or []


def scenario_text(task) -> str:
    ins = (task.get("user_scenario") or {}).get("instructions") or {}
    return " ".join(str(ins.get(k) or "") for k in ("reason_for_call", "task_instructions", "known_info")).lower()


def infer_reason(task) -> str:
    """The benchmark does not label the cancellation reason, so read it off the scenario text."""
    t = scenario_text(task)
    if any(w in t for w in ("sick", "unwell", "ill ", "health", "injur", "covid", "fever")):
        return "health"
    if any(w in t for w in ("weather", "storm", "hurricane", "snow")):
        return "weather"
    return "other"


def run(verbose: bool = False) -> int:
    db, tasks = load()
    R = AirlineRules()
    checks = Counter()
    failures: list[str] = []

    def check(ok: bool, label: str, msg: str):
        checks[label + (":pass" if ok else ":FAIL")] += 1
        if not ok:
            failures.append(msg)
        elif verbose:
            print(f"  ok  {msg}")

    for task in tasks:
        tid = task["id"]
        reason = infer_reason(task)
        for a in actions(task):
            name, args = a["name"], a["arguments"]

            if name == "cancel_reservation":
                rid = args["reservation_id"]
                res = db["reservations"].get(rid)
                if res is None:
                    continue
                # Task 7 upgrades XEHM4B to business before cancelling it, so evaluate the
                # post-upgrade cabin when the same task also changes that reservation's cabin.
                upgraded = next((x for x in actions(task)
                                 if x["name"] == "update_reservation_flights"
                                 and x["arguments"].get("reservation_id") == rid
                                 and x["arguments"].get("cabin")), None)
                if upgraded:
                    res = {**res, "cabin": upgraded["arguments"]["cabin"]}
                d = R.can_cancel(db, res, reason)
                check(d.allowed, "cancel_eligible",
                      f"task {tid}: cancel {rid} ({res['cabin']}, ins={res.get('insurance')}, reason={reason}) -> {d.reason}")

            elif name == "update_reservation_flights":
                rid = args["reservation_id"]
                res = db["reservations"].get(rid)
                if res is None:
                    continue
                if args.get("cabin") and args["cabin"] != res["cabin"]:
                    d = R.can_change_cabin(db, res)   # cabin change is allowed for basic economy too
                    check(d.allowed, "cabin_change_eligible",
                          f"task {tid}: change cabin {rid} {res['cabin']} -> {args['cabin']} -> {d.reason}")
                else:
                    d = R.can_change_flights(db, res)
                    check(d.allowed, "flight_change_eligible",
                          f"task {tid}: change flights {rid} ({res['cabin']}) -> {d.reason}")

            elif name == "send_certificate":
                rid = next((x["arguments"]["reservation_id"] for x in actions(task)
                            if x["name"] == "get_reservation_details"), None)
                res = db["reservations"].get(rid) if rid else None
                if res is None:
                    continue
                for complaint in ("cancelled_flight", "delayed_flight"):
                    d = R.compensation(db, res, complaint, user_asked=True, changed_or_cancelled=True)
                    if d.allowed and d.extra.get("amount") == args["amount"]:
                        check(True, "compensation_amount",
                              f"task {tid}: certificate ${args['amount']} matches {complaint}")
                        break
                else:
                    check(False, "compensation_amount",
                          f"task {tid}: certificate ${args['amount']} matches no compensation rule for {rid}")

            elif name == "update_reservation_baggages":
                rid = args["reservation_id"]
                res = db["reservations"].get(rid)
                if res is None:
                    continue
                total = args.get("total_baggages")
                nonfree = args.get("nonfree_baggages")
                if total is None or nonfree is None:
                    continue
                # The free allowance depends on the cabin at the time of the change, and a single
                # conversation may upgrade the cabin first (tasks 17 and 22 upgrade, then add bags).
                upgraded = next((x for x in actions(task)
                                 if x["name"] == "update_reservation_flights"
                                 and x["arguments"].get("reservation_id") == rid
                                 and x["arguments"].get("cabin")), None)
                if upgraded:
                    res = {**res, "cabin": upgraded["arguments"]["cabin"]}
                d = R.baggage_charge(db, res, total)
                expected_nonfree = d.extra.get("nonfree_baggages")
                check(d.allowed and expected_nonfree == nonfree, "baggage_nonfree",
                      f"task {tid}: {rid} total={total} nonfree expected {expected_nonfree}, task says {nonfree}")

    # The one unconditional direction: a flown segment must always block cancel and flight change.
    flown = [r for r in db["reservations"].values() if R.has_flown(db, r)][:200]
    for res in flown:
        d1, d2 = R.can_cancel(db, res, "health"), R.can_change_flights(db, res)
        check(not d1.allowed and d1.reason == "flown_segment", "flown_blocks_cancel",
              f"flown {res['reservation_id']}: cancel -> {d1.reason}")
        check(not d2.allowed, "flown_blocks_change", f"flown {res['reservation_id']}: change -> {d2.reason}")

    print(f"airline rules vs {len(tasks)} benchmark tasks and {len(flown)} flown reservations")
    for k in sorted(checks):
        print(f"  {k:34s} {checks[k]}")
    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f in failures[:25]:
            print("  -", f)
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(run("-v" in sys.argv))
