"""Generator for tau2-shaped airline tasks.

tau2-bench's 50 airline tasks are hand-written, so unlike telecom there is no upstream pipeline to
adapt. Tasks here are built from the policy instead: `rules.yaml` encodes each clause, `rules.py`
applies it to a concrete reservation, and every task is a case constructed to land on one side of one
rule. The expected actions are then derived from the engine, not written by hand.

Each task carries a small database delta (its own user, reservations and flights) in
`initial_state.initialization_data.agent_data`. tau2's `update_db` deep-merges dict-keyed sections, so
the delta lands in the default database without touching anything else and each task stays a couple of
kilobytes. Every generated id is disjoint from tau2-bench's own.

Scoring matches the benchmark exactly: `reward_basis` is `[DB, COMMUNICATE]`. For a task whose correct
answer is to refuse, the expected action list contains no write, so the database check alone enforces
the refusal. `nl_assertions` carry the reason in words; they are diagnostic under this basis, which is
how all 50 benchmark tasks are scored too.

Usage:
  upstream/tau2-bench/.venv/bin/python domains/airline/gen.py --n 150 --seed 0 --out out/airline
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.tau2_compat import GEN_ROOT, repo_commit, require_tau2, tau2_commit  # noqa: E402

require_tau2()
from tau2.data_model.tasks import InitializationData  # noqa: E402

from common import schema, user_sim  # noqa: E402
from domains.airline import db as adb  # noqa: E402
from domains.airline.rules import AirlineRules  # noqa: E402

DOMAIN = "airline"
R = AirlineRules()
# Tools that never move the database. `transfer_to_human_agents` only returns a message, so a task may
# require it while still expecting the database to stay untouched.
READ_TOOLS = {"get_user_details", "get_reservation_details", "search_direct_flight", "search_onestop_flight",
              "list_all_airports", "get_flight_status", "calculate", "transfer_to_human_agents"}


class BuildError(Exception):
    pass


# --------------------------------------------------------------------------------------
# A case builds a database delta and returns what the agent is expected to do with it.
# --------------------------------------------------------------------------------------
class Case:
    def __init__(self, name: str, fn: Callable, weight: float, group: str):
        self.name, self.fn, self.weight, self.group = name, fn, weight, group


def _read_actions(uid: str, rid: Optional[str], with_user: bool = True) -> list[dict]:
    """The lookups a correct trajectory performs. Reads do not move the database; they are included
    because the benchmark includes them and because they document the intended path."""
    out = []
    if with_user:
        out.append({"name": "get_user_details", "arguments": {"user_id": uid}})
    if rid:
        out.append({"name": "get_reservation_details", "arguments": {"reservation_id": rid}})
    return out


def _status_reads(d, rid: str) -> list[dict]:
    """Checking each segment's status is what "always confirm the facts" means in practice, and it is
    what the benchmark's trajectories do before cancelling or compensating."""
    return [{"name": "get_flight_status", "arguments": {"flight_number": f["flight_number"], "date": f["date"]}}
            for f in d.reservations[rid]["flights"]]


def _acts(raw: list[dict]) -> list[schema.Action]:
    return [schema.Action(action_id=f"{a['name']}_{i}", requestor="assistant", name=a["name"],
                          arguments=a["arguments"]) for i, a in enumerate(raw)]


# ---- cancellation ----
def _cancel_case(rng, cabin, insurance, created, statuses, reason_word, eligible_expected):
    d = adb.AirlineDB(rng)
    membership = rng.choice(adb.MEMBERSHIPS)
    uid = d.add_user(membership, ["credit_card", "gift_card"])
    o, dst = adb.city_pair(rng)
    segs = []
    for st in statuses:
        date = adb.past_date(rng) if st in ("landed", "flying") else adb.future_date(rng)
        segs.append((o, dst, date, st))
        o, dst = dst, o
    rid = d.add_reservation(uid, cabin, insurance, created, segs, rng.randint(1, 3))
    res = d.reservations[rid]
    reason = {"sick": "health", "storm": "weather"}.get(reason_word, "other")
    dec = R.can_cancel(d.delta(), res, reason)
    if dec.allowed != eligible_expected:
        raise BuildError(f"case built the wrong way: expected eligible={eligible_expected}, got {dec.reason}")
    writes = [{"name": "cancel_reservation", "arguments": {"reservation_id": rid}}] if dec.allowed else []
    status_reads = _status_reads(d, rid)
    why = {"sick": " You have come down with a bad flu and cannot travel.",
           "storm": " A storm is forecast and you do not want to risk the trip."}.get(reason_word, "")
    scenario = {
        "reason_for_call": f"You want to cancel reservation {rid}.{why}",
        "known_info": f"You are {d.users[uid]['name']['first_name']} {d.users[uid]['name']['last_name']}. Your user id is {uid}.",
        "task_instructions": ("You are polite but persistent. If the agent tells you the reservation cannot be "
                              "cancelled, ask once whether anything can be done, then accept the answer."
                              if not dec.allowed else
                              "You are brief and cooperative. Confirm once the agent lists the cancellation details."),
    }
    return d, uid, rid, _read_actions(uid, rid) + status_reads + writes, [dec.detail], scenario


def case_cancel_within_24h(rng):
    return _cancel_case(rng, rng.choice(["basic_economy", "economy"]), "no",
                        adb.created_within_24h(rng), ["available"], "plan", True)


def case_cancel_business(rng):
    return _cancel_case(rng, "business", "no", adb.created_long_ago(rng), ["available", "available"], "plan", True)


def case_cancel_insurance_health(rng):
    return _cancel_case(rng, rng.choice(["basic_economy", "economy"]), "yes",
                        adb.created_long_ago(rng), ["available"], "sick", True)


def case_cancel_airline_cancelled(rng):
    return _cancel_case(rng, rng.choice(["basic_economy", "economy"]), "no",
                        adb.created_long_ago(rng), ["cancelled", "available"], "plan", True)


def case_cancel_denied(rng):
    return _cancel_case(rng, rng.choice(["basic_economy", "economy"]), "no",
                        adb.created_long_ago(rng), ["available", "available"], "plan", False)


def case_cancel_denied_flown(rng):
    d, uid, rid, acts, nl, scen = _cancel_case(rng, "business", "yes", adb.created_long_ago(rng),
                                               ["landed", "available"], "plan", False)
    # The policy says to escalate here. The call does not move the database, so including it keeps the
    # shape of the benchmark's own transfer task without weakening the database check.
    acts = acts + [{"name": "transfer_to_human_agents",
                    "arguments": {"summary": f"Part of reservation {rid} has already been flown; cancellation needs a human agent."}}]
    nl = [nl[0], "Agent should transfer the user to a human agent."]
    return d, uid, rid, acts, nl, scen


# ---- modification ----
def _modify_reservation(rng, cabin, statuses=("available", "available"), insurance="no", n_pax=None):
    d = adb.AirlineDB(rng)
    uid = d.add_user(rng.choice(adb.MEMBERSHIPS), ["credit_card", "gift_card"])
    o, dst = adb.city_pair(rng)
    segs, oo, dd = [], o, dst
    for st in statuses:
        date = adb.past_date(rng) if st in ("landed", "flying") else adb.future_date(rng)
        segs.append((oo, dd, date, st))
        oo, dd = dd, oo
    rid = d.add_reservation(uid, cabin, insurance, adb.created_long_ago(rng), segs,
                            n_pax or rng.randint(1, 3))
    return d, uid, rid


def case_change_flights(rng):
    d, uid, rid = _modify_reservation(rng, rng.choice(["economy", "business"]))
    res = d.reservations[rid]
    dec = R.can_change_flights(d.delta(), res)
    if not dec.allowed:
        raise BuildError(dec.reason)
    seg = res["flights"][0]
    new_date = adb.future_date(rng, 3, 20)
    alts = d.add_alternative_flights(seg["origin"], seg["destination"], new_date, 2, len(res["passengers"]))
    chosen = alts[0]
    new_flights = [{"flight_number": chosen, "date": new_date}] + \
                  [{"flight_number": f["flight_number"], "date": f["date"]} for f in res["flights"][1:]]
    pay = next(p for p, m in d.users[uid]["payment_methods"].items() if m["source"] in ("credit_card", "gift_card"))
    writes = [{"name": "update_reservation_flights",
               "arguments": {"reservation_id": rid, "cabin": res["cabin"],
                             "flights": new_flights, "payment_id": pay}}]
    reads = _read_actions(uid, rid) + [
        {"name": "search_direct_flight", "arguments": {"origin": seg["origin"], "destination": seg["destination"], "date": new_date}}]
    scenario = {
        "reason_for_call": f"You want to move the outbound flight on reservation {rid} to {new_date}.",
        "known_info": f"You are {d.users[uid]['name']['first_name']} {d.users[uid]['name']['last_name']}. Your user id is {uid}.",
        "task_instructions": "You want to keep the return flight as it is. Pay any difference with the card on file.",
    }
    return d, uid, rid, reads + writes, [dec.detail], scenario


def case_change_flights_denied_basic_economy(rng):
    d, uid, rid = _modify_reservation(rng, "basic_economy")
    res = d.reservations[rid]
    dec = R.can_change_flights(d.delta(), res)
    if dec.allowed:
        raise BuildError("basic economy should not be flight-changeable")
    seg = res["flights"][0]
    d.add_alternative_flights(seg["origin"], seg["destination"], adb.future_date(rng, 3, 20), 2, len(res["passengers"]))
    scenario = {
        "reason_for_call": f"You want to move your flight on reservation {rid} to a later date.",
        "known_info": f"You are {d.users[uid]['name']['first_name']} {d.users[uid]['name']['last_name']}. Your user id is {uid}.",
        "task_instructions": ("If the agent says the reservation cannot be changed, ask whether upgrading the cabin "
                              "would help, and accept the answer you are given. Do not ask to cancel."),
    }
    return d, uid, rid, _read_actions(uid, rid), [dec.detail], scenario


def case_change_cabin(rng):
    d, uid, rid = _modify_reservation(rng, rng.choice(["basic_economy", "economy"]))
    res = d.reservations[rid]
    dec = R.can_change_cabin(d.delta(), res)
    if not dec.allowed:
        raise BuildError(dec.reason)
    new_cabin = "business" if res["cabin"] == "economy" else "economy"
    pay = next(p for p, m in d.users[uid]["payment_methods"].items() if m["source"] in ("credit_card", "gift_card"))
    writes = [{"name": "update_reservation_flights",
               "arguments": {"reservation_id": rid, "cabin": new_cabin,
                             "flights": [{"flight_number": f["flight_number"], "date": f["date"]} for f in res["flights"]],
                             "payment_id": pay}}]
    scenario = {
        "reason_for_call": f"You want to upgrade reservation {rid} to {new_cabin.replace('_', ' ')}.",
        "known_info": f"You are {d.users[uid]['name']['first_name']} {d.users[uid]['name']['last_name']}. Your user id is {uid}.",
        "task_instructions": "Keep the same flights and dates. You are willing to pay the difference with the card on file.",
    }
    return d, uid, rid, _read_actions(uid, rid) + writes, [dec.detail], scenario


def case_baggage_add(rng):
    d, uid, rid = _modify_reservation(rng, rng.choice(["economy", "business"]))
    res = d.reservations[rid]
    free = R.free_baggage(d.delta(), res)
    total = free + rng.randint(1, 2)
    dec = R.baggage_charge(d.delta(), res, total)
    if not dec.allowed:
        raise BuildError(dec.reason)
    pay = next(p for p, m in d.users[uid]["payment_methods"].items() if m["source"] in ("credit_card", "gift_card"))
    writes = [{"name": "update_reservation_baggages",
               "arguments": {"reservation_id": rid, "total_baggages": total,
                             "nonfree_baggages": dec.extra["nonfree_baggages"], "payment_id": pay}}]
    scenario = {
        "reason_for_call": f"You want to add checked bags to reservation {rid}, {total} in total.",
        "known_info": f"You are {d.users[uid]['name']['first_name']} {d.users[uid]['name']['last_name']}. Your user id is {uid}.",
        "task_instructions": "Ask what it will cost before you agree. Pay with the card on file.",
    }
    return d, uid, rid, _read_actions(uid, rid) + writes, [dec.detail], scenario


def case_baggage_remove_denied(rng):
    d, uid, rid = _modify_reservation(rng, rng.choice(["economy", "business"]))
    res = d.reservations[rid]
    res["total_baggages"] = 2
    dec = R.baggage_charge(d.delta(), res, 0)
    if dec.allowed:
        raise BuildError("removing bags should be refused")
    scenario = {
        "reason_for_call": f"You want to remove the checked bags from reservation {rid} and get the money back.",
        "known_info": f"You are {d.users[uid]['name']['first_name']} {d.users[uid]['name']['last_name']}. Your user id is {uid}.",
        "task_instructions": "If the agent says bags cannot be removed, accept it and end the conversation politely.",
    }
    return d, uid, rid, _read_actions(uid, rid), [dec.detail], scenario


def case_insurance_add_denied(rng):
    d, uid, rid = _modify_reservation(rng, rng.choice(["economy", "business"]), insurance="no")
    dec = R.can_add_insurance()
    scenario = {
        "reason_for_call": f"You want to add travel insurance to reservation {rid}.",
        "known_info": f"You are {d.users[uid]['name']['first_name']} {d.users[uid]['name']['last_name']}. Your user id is {uid}.",
        "task_instructions": "You are willing to pay for it. If the agent refuses, ask once why, then accept.",
    }
    return d, uid, rid, _read_actions(uid, rid), [dec.detail], scenario


def case_passenger_count_denied(rng):
    d, uid, rid = _modify_reservation(rng, rng.choice(["economy", "business"]), n_pax=2)
    dec = R.can_change_passenger_count()
    scenario = {
        "reason_for_call": f"One of the two travellers on reservation {rid} can no longer come, so you want to drop them.",
        "known_info": f"You are {d.users[uid]['name']['first_name']} {d.users[uid]['name']['last_name']}. Your user id is {uid}.",
        "task_instructions": "If the agent says the number of passengers cannot change, ask whether a human agent could, then accept the answer.",
    }
    return d, uid, rid, _read_actions(uid, rid), [dec.detail], scenario


def case_passenger_change(rng):
    d, uid, rid = _modify_reservation(rng, rng.choice(["economy", "business"]), n_pax=2)
    res = d.reservations[rid]
    new = d.passengers(uid, 2)[1]
    new = {"first_name": rng.choice(["Dana", "Noor", "Rafael", "Ingrid"]), "last_name": new["last_name"], "dob": new["dob"]}
    writes = [{"name": "update_reservation_passengers",
               "arguments": {"reservation_id": rid, "passengers": [res["passengers"][0], new]}}]
    scenario = {
        "reason_for_call": f"The second traveller on reservation {rid} changed, you want to put {new['first_name']} {new['last_name']} on it instead.",
        "known_info": (f"You are {d.users[uid]['name']['first_name']} {d.users[uid]['name']['last_name']}. Your user id is {uid}. "
                       f"The new passenger is {new['first_name']} {new['last_name']}, born {new['dob']}."),
        "task_instructions": "The number of travellers stays the same. Confirm when the agent lists the change.",
    }
    return d, uid, rid, _read_actions(uid, rid) + writes, ["Agent should replace the passenger without changing the number of passengers."], scenario


# ---- compensation ----
def case_compensation_cancelled_flight(rng):
    d = adb.AirlineDB(rng)
    uid = d.add_user(rng.choice(["silver", "gold"]), ["credit_card", "gift_card"])
    o, dst = adb.city_pair(rng)
    rid = d.add_reservation(uid, rng.choice(["economy", "business"]), "yes", adb.created_long_ago(rng),
                            [(o, dst, adb.future_date(rng), "cancelled"), (dst, o, adb.future_date(rng), "available")],
                            rng.randint(1, 3))
    res = d.reservations[rid]
    dec = R.compensation(d.delta(), res, "cancelled_flight", user_asked=True, changed_or_cancelled=False)
    if not dec.allowed:
        raise BuildError(dec.reason)
    writes = _status_reads(d, rid) + [{"name": "send_certificate", "arguments": {"user_id": uid, "amount": dec.extra["amount"]}}]
    scenario = {
        "reason_for_call": f"The airline cancelled a flight on reservation {rid} and you want to be compensated for the trouble.",
        "known_info": f"You are {d.users[uid]['name']['first_name']} {d.users[uid]['name']['last_name']}. Your user id is {uid}.",
        "task_instructions": "You explicitly ask for compensation. You do not want to cancel the rest of the trip.",
    }
    return d, uid, rid, _read_actions(uid, rid) + writes, [dec.detail], scenario


def case_compensation_denied(rng):
    d = adb.AirlineDB(rng)
    uid = d.add_user("regular", ["credit_card"])
    o, dst = adb.city_pair(rng)
    rid = d.add_reservation(uid, rng.choice(["basic_economy", "economy"]), "no", adb.created_long_ago(rng),
                            [(o, dst, adb.future_date(rng), "delayed")], rng.randint(1, 2))
    res = d.reservations[rid]
    dec = R.compensation(d.delta(), res, "delayed_flight", user_asked=True, changed_or_cancelled=False)
    if dec.allowed:
        raise BuildError("regular member without insurance in economy should not be compensated")
    scenario = {
        "reason_for_call": f"Your flight on reservation {rid} is delayed and you want a travel certificate for it.",
        "known_info": f"You are {d.users[uid]['name']['first_name']} {d.users[uid]['name']['last_name']}. Your user id is {uid}.",
        "task_instructions": "You ask for compensation directly. If the agent says you are not eligible, ask once more, then accept.",
    }
    return d, uid, rid, _read_actions(uid, rid), [dec.detail], scenario


# ---- booking ----
def case_book(rng):
    d = adb.AirlineDB(rng)
    uid = d.add_user(rng.choice(adb.MEMBERSHIPS), ["credit_card", "gift_card"])
    o, dst = adb.city_pair(rng)
    date = adb.future_date(rng, 3, 20)
    cabin = rng.choice(["economy", "business"])
    n_pax = rng.randint(1, 2)
    fn, entry = d.add_flight(o, dst, date, "available", min_seats=n_pax)
    pay = next(p for p, m in d.users[uid]["payment_methods"].items() if m["source"] == "credit_card")
    dec = R.can_book(d.delta(), d.users[uid], cabin, d.passengers(uid, n_pax), [pay], ["available"])
    if not dec.allowed:
        raise BuildError(dec.reason)
    free_per_pax = R.r["baggage"]["free_allowance"][d.users[uid]["membership"]][cabin]
    writes = [{"name": "book_reservation", "arguments": {
        "user_id": uid, "origin": o, "destination": dst, "flight_type": "one_way", "cabin": cabin,
        "flights": [{"flight_number": fn, "date": date}],
        "passengers": d.passengers(uid, n_pax),
        "payment_methods": [{"payment_id": pay, "amount": int(entry["prices"][cabin] * n_pax)}],
        "total_baggages": free_per_pax * n_pax, "nonfree_baggages": 0, "insurance": "no"}}]
    scenario = {
        "reason_for_call": f"You want to book a one way {cabin.replace('_', ' ')} flight from {o} to {dst} on {date} for {n_pax} traveller(s).",
        "known_info": f"You are {d.users[uid]['name']['first_name']} {d.users[uid]['name']['last_name']}. Your user id is {uid}.",
        "task_instructions": ("Pay with the credit card on file. You only want the free checked bags you are entitled to, "
                              "and you do not want travel insurance."),
    }
    reads = [{"name": "get_user_details", "arguments": {"user_id": uid}},
             {"name": "search_direct_flight", "arguments": {"origin": o, "destination": dst, "date": date}}]
    return d, uid, None, reads + writes, [dec.detail], scenario


def case_cancel_two_reservations(rng):
    """Two eligible reservations in one conversation, so the task needs two writes. The benchmark has
    tasks with up to five."""
    d = adb.AirlineDB(rng)
    uid = d.add_user(rng.choice(adb.MEMBERSHIPS), ["credit_card", "gift_card"])
    rids = []
    for _ in range(2):
        o, dst = adb.city_pair(rng)
        rids.append(d.add_reservation(uid, "business", "no", adb.created_long_ago(rng),
                                      [(o, dst, adb.future_date(rng), "available")], rng.randint(1, 2)))
    reads = [{"name": "get_user_details", "arguments": {"user_id": uid}}]
    writes, nl = [], []
    for rid in rids:
        dec = R.can_cancel(d.delta(), d.reservations[rid], "other")
        if not dec.allowed:
            raise BuildError(dec.reason)
        reads.append({"name": "get_reservation_details", "arguments": {"reservation_id": rid}})
        writes.append({"name": "cancel_reservation", "arguments": {"reservation_id": rid}})
        nl.append(f"Agent should cancel reservation {rid}.")
    scenario = {
        "reason_for_call": f"You want to cancel both of your upcoming trips, reservations {rids[0]} and {rids[1]}.",
        "known_info": f"You are {d.users[uid]['name']['first_name']} {d.users[uid]['name']['last_name']}. Your user id is {uid}.",
        "task_instructions": "Both should go. Confirm each one when the agent lists it.",
    }
    return d, uid, rids[0], reads + writes, nl, scenario


CASES = [
    Case("cancel_two_reservations", case_cancel_two_reservations, 5, "cancel"),
    Case("cancel_within_24h", case_cancel_within_24h, 6, "cancel"),
    Case("cancel_business", case_cancel_business, 6, "cancel"),
    Case("cancel_insurance_health", case_cancel_insurance_health, 6, "cancel"),
    Case("cancel_airline_cancelled", case_cancel_airline_cancelled, 5, "cancel"),
    Case("cancel_denied", case_cancel_denied, 8, "refuse"),
    Case("cancel_denied_flown", case_cancel_denied_flown, 5, "refuse"),
    Case("change_flights", case_change_flights, 10, "modify"),
    Case("change_flights_denied_basic_economy", case_change_flights_denied_basic_economy, 8, "refuse"),
    Case("change_cabin", case_change_cabin, 8, "modify"),
    Case("baggage_add", case_baggage_add, 7, "modify"),
    Case("baggage_remove_denied", case_baggage_remove_denied, 4, "refuse"),
    Case("insurance_add_denied", case_insurance_add_denied, 4, "refuse"),
    Case("passenger_count_denied", case_passenger_count_denied, 4, "refuse"),
    Case("passenger_change", case_passenger_change, 4, "modify"),
    Case("compensation_cancelled_flight", case_compensation_cancelled_flight, 5, "compensation"),
    Case("compensation_denied", case_compensation_denied, 5, "refuse"),
    Case("book", case_book, 8, "book"),
]


def build_task(idx: int, tag: str, case: Case, persona_name: str, rng: random.Random) -> tuple[dict, dict]:
    d, uid, rid, raw_actions, nl, scen = case.fn(rng)
    persona_text = user_sim.render_persona(persona_name, rng, "555-555-0100") if persona_name != "None" else None
    name = f"[{case.name}][PERSONA:{persona_name}][GEN:{tag}]"
    writes = [a for a in raw_actions if a["name"] not in
              READ_TOOLS]
    data = schema.Tau2TaskData(
        idx=idx, name=name, description=f"Purpose: {case.name.replace('_', ' ')}", id=name,
        user_scenario=schema.UserScenario(
            persona=persona_text,
            instructions=schema.UserInstructions(domain=DOMAIN, reason_for_call=scen["reason_for_call"],
                                                 known_info=scen["known_info"], unknown_info=None,
                                                 task_instructions=scen["task_instructions"]),
        ),
        ticket=None,
        initial_state=schema.InitialState(
            initialization_data=schema.InitializationData(agent_data=d.delta(), user_data=None),
            initialization_actions=None, message_history=None),
        evaluation_criteria=schema.EvaluationCriteria(
            actions=_acts(raw_actions), env_assertions=[], communicate_info=None,
            nl_assertions=nl, reward_basis=["DB", "COMMUNICATE"]),
        domain=DOMAIN,
        tau_description=schema.TauDescription(purpose=case.name.replace("_", " "), relevant_policies=None, notes=None),
    ).to_dict()
    data["evaluation_criteria"]["env_assertions"] = None
    meta = {"idx": idx, "id": name, "case": case.name, "group": case.group, "persona": persona_name,
            "n_writes": len(writes), "write_names": sorted({a["name"] for a in writes}),
            "user_id": uid, "reservation_id": rid, "delta_bytes": len(json.dumps(d.delta()))}
    return data, meta


def verify_task(data: dict) -> Optional[str]:
    """Replay the expected actions in a real environment. They must all execute, and the database must
    move exactly when the task expects a write."""
    from tau2.domains.airline.environment import get_environment

    env = get_environment()
    env.set_state(
        initialization_data=InitializationData.model_validate(data["initial_state"]["initialization_data"]),
        initialization_actions=None, message_history=[])
    before = env.get_db_hash()
    actions = data["evaluation_criteria"]["actions"]
    writes = [a for a in actions if a["name"] not in
              READ_TOOLS]
    for a in actions:
        try:
            env.make_tool_call(tool_name=a["name"], requestor=a["requestor"], **a["arguments"])
        except Exception as e:  # noqa: BLE001
            return f"action {a['name']} failed: {type(e).__name__}: {str(e)[:160]}"
    after = env.get_db_hash()
    if writes and before == after:
        return f"expected {len(writes)} write(s) but the database did not change"
    if not writes and before != after:
        return "task expects no write but the database changed"
    return None


def rebalance(cases: list, refuse_share: float | None) -> list[float]:
    """Weights, optionally renormalised so the `refuse` group takes a target share of the set.

    The default matches the reference set's own share of tasks whose correct answer is to do nothing.
    Raise it when the point is training data for refusal rather than a distribution match."""
    w = [c.weight for c in cases]
    if refuse_share is None:
        return w
    idx_r = [i for i, c in enumerate(cases) if c.group == "refuse"]
    idx_o = [i for i, c in enumerate(cases) if c.group != "refuse"]
    tot_r, tot_o = sum(w[i] for i in idx_r), sum(w[i] for i in idx_o)
    for i in idx_r:
        w[i] = w[i] / tot_r * refuse_share
    for i in idx_o:
        w[i] = w[i] / tot_o * (1 - refuse_share)
    return w


def generate(n: int, seed: int, persona_mix: dict, refuse_share: float | None = None,
             verbose: bool = False) -> tuple[list[dict], list[dict], dict]:
    rng = random.Random(seed)
    weights = rebalance(CASES, refuse_share)
    records, metas, stats = [], [], Counter()
    idx = 0
    attempts = 0
    while len(records) < n and attempts < n * 60:
        attempts += 1
        case = rng.choices(CASES, weights=weights, k=1)[0]
        persona = user_sim.sample_persona(rng, persona_mix)
        try:
            data, meta = build_task(idx, f"{seed}-{idx:05d}", case, persona, rng)
        except BuildError as e:
            stats[f"build_fail:{case.name}"] += 1
            if verbose:
                print(f"build fail {case.name}: {e}", file=sys.stderr)
            continue
        err = verify_task(data)
        if err:
            stats[f"verify_fail:{case.name}"] += 1
            if verbose:
                print(f"verify fail {case.name}: {err}", file=sys.stderr)
            continue
        records.append(schema.episode_record(data))
        metas.append(meta)
        idx += 1
        stats["kept"] += 1
    return records, metas, dict(stats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(GEN_ROOT / "out" / "airline"))
    ap.add_argument("--persona-mix", default=None)
    ap.add_argument("--refuse-share", type=float, default=0.48,
                    help="share of tasks whose correct answer is to do nothing (tau2-bench airline: 0.48)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    mix = json.loads(a.persona_mix) if a.persona_mix else {"None": 0.5, "Easy": 0.08, "Hard": 0.08, "Verbose": 0.06,
                                                           "Terse": 0.06, "NonNative": 0.06, "Impatient": 0.06,
                                                           "WrongNumberOnce": 0.04, "SideRequest": 0.03, "TechSavvy": 0.03}
    t0 = time.time()
    records, metas, stats = generate(a.n, a.seed, mix, a.refuse_share, a.verbose)
    print(f"generated {len(records)} airline tasks in {time.time() - t0:.1f}s; stats={dict(stats)}")

    from fidelity.leakage import check as leakage_check

    leak = leakage_check(records, None, domain=DOMAIN)
    if not leak["ok"]:
        print("LEAKAGE CHECK FAILED — not writing output:", json.dumps(leak, indent=1)[:1500], file=sys.stderr)
        sys.exit(2)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "tasks.jsonl", "w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "sample.jsonl", "w") as fh:
        for r in records[:2]:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "meta.jsonl", "w") as fh:
        for m in metas:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    (out / "tasks_tau2.json").write_text(json.dumps(
        [schema.to_tau2_task(r["data"]).model_dump(mode="json") for r in records], indent=1, ensure_ascii=False))
    (out / "taskset.toml").write_text('[env.taskset]\ndomain = "airline"\ntasks = "tasks_tau2.json"\n')
    manifest = {
        "generator": "domains/airline/gen.py", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_commit": repo_commit(), "tau2_commit": tau2_commit(), "seed": a.seed, "n": len(records),
        "args": vars(a), "persona_mix": mix, "stats": stats,
        "case_hist": dict(Counter(m["case"] for m in metas)),
        "group_hist": dict(Counter(m["group"] for m in metas)),
        "persona_hist": dict(Counter(m["persona"] for m in metas)),
        "no_write_tasks": sum(1 for m in metas if m["n_writes"] == 0),
        "write_names": dict(Counter(w for m in metas for w in m["write_names"])),
        "median_delta_bytes": sorted(m["delta_bytes"] for m in metas)[len(metas) // 2] if metas else 0,
        "leakage": leak,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(json.dumps({k: manifest[k] for k in ("n", "group_hist", "no_write_tasks", "write_names", "median_delta_bytes")}, ensure_ascii=False))
    print(f"leakage ok={leak['ok']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
