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
from domains.airline import units as U  # noqa: E402
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
    def __init__(self, name: str, fn: Callable, weight: float, group: str, single_record: bool = True,
                 history: bool = True, validate: Optional[Callable] = None):
        self.name, self.fn, self.weight, self.group = name, fn, weight, group
        # Whether the unknown-id axis applies: it needs exactly one target record that the scenario
        # names. Booking has no record yet, and the multi-record cases already span several.
        self.single_record = single_record
        # Whether the shared history filler may add reservations. A case whose request ranges over
        # *every* record the user holds has to own its own distractors, because a filler reservation
        # that happens to satisfy the rule would make the expected action list wrong.
        self.history = history
        # An optional check run on the finished task. A case whose request ranges over every record
        # the user holds cannot state its expected actions until the database is final.
        self.validate = validate


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


# ---------------------------------------------------------------------------
# Account history and the unknown-id axis
# ---------------------------------------------------------------------------
# Two structural properties that cut across every case, rather than belonging to any one of them:
#
#   1. Real customers have more than one booking. Giving each user a few extra reservations costs
#      nothing in the reference trajectory (the agent still acts on one) but stops every profile from
#      looking identical, and gives a careless agent something to get wrong.
#   2. Customers often cannot quote the record number. When they cannot, the correct trajectory really
#      does read several reservations to find the right one, which is what lifts the number of records
#      a task touches. Turning that into an axis applied to any single-record case, instead of two
#      bespoke cases, is what makes the spread reach the reference set's.
#
# The DB check ignores reads, so neither property changes how a task is scored.

def add_history(d, uid: str, rng: random.Random, n: tuple[int, int] = (1, 3)) -> list[str]:
    """Extra reservations for this user, none of which is the task's target."""
    out = []
    for _ in range(rng.randint(*n)):
        o, dst = adb.city_pair(rng)
        out.append(d.add_reservation(uid, rng.choice(adb.CABINS), rng.choice(["yes", "no"]),
                                     adb.created_long_ago(rng),
                                     [(o, dst, adb.future_date(rng), "available")], rng.randint(1, 2)))
    return out


def describe_reservation(res: dict) -> str:
    f = res["flights"][0]
    return f"the trip from {f['origin']} to {f['destination']} on {f['date']}"


def hide_id(scen: dict, d, uid: str, rid: str, reads: list[dict], rng: random.Random) -> tuple[dict, list[dict]]:
    """Replace the reservation id in the scenario with a description, and make the reference trajectory
    look through the profile instead of jumping straight to the record.

    Returns the scenario and reads unchanged if the id cannot be removed cleanly, so a task never
    claims the user does not know something that the text still spells out."""
    desc = describe_reservation(d.reservations[rid])
    reason = scen["reason_for_call"]
    if rid not in reason:
        return scen, reads
    reason = reason.replace(f"reservation {rid}", desc).replace(rid, desc)
    if rid in reason:
        return scen, reads
    scen = dict(scen)
    scen["reason_for_call"] = reason + " You do not remember the reservation number."
    all_rids = list(d.users[uid]["reservations"])
    rng.shuffle(all_rids)
    lookups = [{"name": "get_user_details", "arguments": {"user_id": uid}}]
    lookups += [{"name": "get_reservation_details", "arguments": {"reservation_id": r}} for r in all_rids]
    tail = [a for a in reads if a["name"] not in ("get_user_details", "get_reservation_details")]
    return scen, lookups + tail


# ---------------------------------------------------------------------------
# Contingencies derived from the rule engine
# ---------------------------------------------------------------------------
# tau2-bench's airline instructions run to fifty-eight words at the median, and what fills them is not
# varied phrasing of generic behaviour but task-specific contingencies: "if the agent tells you
# cancellation is not possible, mention that you were told you didn't need insurance". That is pressure
# aimed at the exact rule the task turns on.
#
# The rule engine already names that rule: every Decision carries a reason. So the contingency can be
# looked up rather than written per case, and it stays correct when the rules change.
#
# Each line ends by closing the loop, with the user declining the alternative they just floated. The
# pressure has to leave the correct end state where it was: a user who floats "could we cancel and
# rebook instead" and then agrees to it has authorised a different outcome, and the expected actions
# would no longer describe a correct trajectory.

CONTINGENCY = {
    "basic_economy": [
        ("If the agent says the reservation is basic economy and cannot be changed, ask whether upgrading "
         "the cabin first would make it possible, and say you do not want to pay for an upgrade when they explain it."),
        ("If you are told basic economy cannot be modified, say you were not aware of that when you booked, "
         "and accept the answer once the agent has explained it."),
    ],
    "flown_segment": [
        "If the agent says part of the trip has already been flown, ask whether a human agent could still do it, and accept what you are told.",
        "If you are told the trip has already started, ask what happens to the remaining leg, then let it go.",
    ],
    "not_eligible": [
        ("If the agent says the booking cannot be cancelled, ask whether adding travel insurance now would "
         "change that, and drop it once the agent explains."),
        ("If you are told there is no refund, ask whether a credit for a future trip is possible instead, "
         "and accept the answer."),
    ],
    "insurance_after_booking": [
        "If the agent refuses, ask whether it can be added for just one passenger, and accept the answer.",
        "If you are told insurance cannot be added now, ask whether it could be bought separately, then let it go.",
    ],
    "passenger_count": [
        ("If the agent says the number of travellers cannot change, ask whether cancelling and rebooking would "
         "work, and say you do not want to do that once they explain."),
        ("If you are told the passenger count is fixed, ask whether the seat can simply go unused, and accept "
         "what you are told."),
    ],
    "not_eligible_for_compensation": [
        ("If the agent declines, mention that someone else on the same flight was given a certificate, and let "
         "it go when the agent explains the policy."),
        ("If you are told you are not eligible, ask what would have made you eligible, and accept the answer."),
    ],
    "remove_not_allowed": [
        "If the agent says bags cannot be removed, ask whether the fee can be refunded instead, and accept the answer.",
        "If you are told bags cannot be taken off, ask whether you can simply not bring them, then move on.",
    ],
    "within_24h": [
        "If the agent asks why you are cancelling, say your plans changed.",
        "If you are asked for a reason, say something came up and you no longer need the trip.",
    ],
    "cabin_business": [
        "If the agent asks why you are cancelling, say your plans changed.",
        "If you are asked for a reason, say the trip is no longer going ahead.",
    ],
    "airline_cancelled": [
        "If the agent asks why you are cancelling, say it is because the airline cancelled one of the flights.",
        "If you are asked for a reason, point to the flight the airline cancelled.",
    ],
    "insurance_covered": [
        "If the agent asks why you are cancelling, say it is for health reasons and that you have travel insurance.",
        "If you are asked for a reason, explain that you are unwell and that the booking has insurance.",
    ],
}


def core_with_contingency(core: str, dec, rng: random.Random, p: float = 0.75) -> str:
    """Case constraints plus, usually, the contingency that belongs to the rule this task turns on."""
    variants = CONTINGENCY.get(getattr(dec, "reason", None))
    if variants and rng.random() < p:
        return f"{core} {rng.choice(variants)}".strip()
    return core

# ---------------------------------------------------------------------------
# What the agent must say
# ---------------------------------------------------------------------------
# With basis [DB, COMMUNICATE] and an empty communicate_info, the COMMUNICATE leg is vacuous, so a task
# whose correct answer is to change nothing passes for any agent that changes nothing — including one
# that says nothing at all, or invents a reason. For training data that rewards freezing as much as a
# correct refusal, and 39% of this set has no correct write.
#
# The fix is one short must-mention string, matched case-insensitively as a substring. Picking it is the
# whole problem: too specific and a correct agent fails on phrasing. Measured on tau2-bench's own
# airline traces, over the 36 tasks its teacher passes:
#
#   a topic word taken from the user's own request   35/36   97%
#   "basic economy", where that rule is the blocker  11/12   92%
#   "human agent", where a segment has been flown     3/5    60%
#   "insurance", where cancellation is not eligible  13/18   72%
#   the reservation id the agent acted on            35/48   73%
#
# So the requirement is anchored on the subject the user raised, not on the reason the agent must give.
# It cannot be satisfied by silence, and it does not punish a correct refusal for its wording. The
# reason-anchored strings were measured and rejected: at 60-73% recall they would fail a correct agent
# between a quarter and two-fifths of the time, which is worse for training data than vacuity.

TOPIC_ANCHOR = {
    "cancel_within_24h": "cancel", "cancel_business": "cancel", "cancel_insurance_health": "cancel",
    "cancel_airline_cancelled": "cancel", "cancel_denied": "cancel", "cancel_denied_flown": "cancel",
    "cancel_two_reservations": "cancel", "cancel_unknown_reservation": "cancel",
    "cancel_then_compensation": "cancel",
    "change_flights": "flight", "change_flights_denied_basic_economy": "flight",
    "change_flights_then_baggage": "flight",
    "change_cabin": "cabin", "upgrade_then_baggage": "cabin",
    "baggage_add": "bag", "baggage_remove_denied": "bag",
    "insurance_add_denied": "insurance",
    "passenger_change": "passenger", "passenger_count_denied": "passenger",
    "compensation_cancelled_flight": "compensation", "compensation_denied": "certificate",
    "compensation_facts_denied": "compensation", "cancel_mixed_eligibility": "cancel",
    "book": "book",
}
# Added on top of the topic anchor only where the rule reason itself is well attested.
REASON_ANCHOR = {"basic_economy": "basic economy"}


def communicate_info(case_name: str, dec, scen: dict | None = None) -> list[str] | None:
    """The must-mention strings for a task, with the topic anchor checked against the request.

    The topic anchor only has its measured 97% recall because it is a word the user themselves used; an
    anchor the scenario never says would be a phrasing trap. That property is checked here rather than
    trusted, so drifting scenario wording fails loudly instead of quietly producing tasks a correct
    agent cannot pass. The reason anchor is exempt: the user does not know the reason, which is the
    point of the task, and it carries its own measurement."""
    out = []
    given = (scen or {}).get("_anchors")
    if given:
        # Topic anchors must be words the user used; reason anchors are exempt for the same reason as
        # in the single-rule path — the user does not know the reason, which is the point of the task.
    # Only `reason_for_call` counts. `task_instructions` is guidance to the simulator about how to
    # behave, not words the user necessarily says aloud, and the agent echoes what is said. A probe
    # run found a task whose database check passed and whose certificate amount was exact, scored
    # zero because the request said "be compensated" while the anchor was "compensation": the word
    # appeared only in the instructions, so this check had waved it through.
        request = (scen.get("reason_for_call") or "").lower()
        exempt = set(REASON_ANCHOR.values())
        for a in given:
            if a not in exempt and a not in request:
                raise BuildError(f"topic anchor {a!r} is absent from the composed request")
            if a not in out:
                out.append(a)
        return out or None
    a = TOPIC_ANCHOR.get(case_name)
    if a:
        if scen is not None:
        # Only `reason_for_call` counts. `task_instructions` is guidance to the simulator about how to
        # behave, not words the user necessarily says aloud, and the agent echoes what is said. A probe
        # run found a task whose database check passed and whose certificate amount was exact, scored
        # zero because the request said "be compensated" while the anchor was "compensation": the word
        # appeared only in the instructions, so this check had waved it through.
            request = (scen.get("reason_for_call") or "").lower()
            if a not in request:
                raise BuildError(f"topic anchor {a!r} for {case_name} is absent from the user's request")
        out.append(a)
    r = REASON_ANCHOR.get(getattr(dec, "reason", None))
    if r and r not in out:
        out.append(r)
    return out or None

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
        "task_instructions": user_sim.instructions(rng, n=2, core=core_with_contingency("", dec, rng), refusable=True),
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


# Naming the flight is not decoration. The request "move it to <date>" is satisfied by any flight on
# that date, so the one the generator picked is not the only correct answer, and the default database
# still holds tau2-bench's own inventory on the same routes, which the agent can and does book. A probe
# run scored change_flights 1 of 7 and caught a booking placed on an upstream HAT flight for exactly
# these two reasons.
def case_change_flights(rng):
    d, uid, rid = _modify_reservation(rng, rng.choice(["economy", "business"]))
    res = d.reservations[rid]
    dec = R.can_change_flights(d.delta(), res)
    if not dec.allowed:
        raise BuildError(dec.reason)
    seg = res["flights"][0]
    new_date = adb.date_before(rng, res["flights"][1]["date"] if len(res["flights"]) > 1 else None, 3, 20)
    if new_date is None:
        raise BuildError("no date leaves the outbound before the return")
    alts = d.add_alternative_flights(seg["origin"], seg["destination"], new_date, 2, len(res["passengers"]))
    chosen = alts[0]
    dep = d.flights[chosen]["scheduled_departure_time_est"][:5]
    new_flights = [{"flight_number": chosen, "date": new_date}] + \
                  [{"flight_number": f["flight_number"], "date": f["date"]} for f in res["flights"][1:]]
    pay = next(p for p, m in d.users[uid]["payment_methods"].items() if m["source"] in ("credit_card", "gift_card"))
    writes = [{"name": "update_reservation_flights",
               "arguments": {"reservation_id": rid, "cabin": res["cabin"],
                             "flights": new_flights, "payment_id": pay}}]
    reads = _read_actions(uid, rid) + [
        {"name": "search_direct_flight", "arguments": {"origin": seg["origin"], "destination": seg["destination"], "date": new_date}}]
    scenario = {
        "reason_for_call": (f"You want to move the outbound flight on reservation {rid} to flight "
                            f"{chosen} on {new_date}, the one departing {dep}."),
        "known_info": f"You are {d.users[uid]['name']['first_name']} {d.users[uid]['name']['last_name']}. Your user id is {uid}.",
        "task_instructions": user_sim.instructions(rng, n=2, core="You want to keep the return flight as it is."),
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
        "task_instructions": user_sim.instructions(rng, n=2, core=core_with_contingency("Do not ask to cancel the reservation.", dec, rng)),
        "_decision": dec,
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
        "reason_for_call": f"You want to move reservation {rid} up to the {new_cabin.replace('_', ' ')} cabin.",
        "known_info": f"You are {d.users[uid]['name']['first_name']} {d.users[uid]['name']['last_name']}. Your user id is {uid}.",
        "task_instructions": user_sim.instructions(rng, n=2, core="Keep the same flights and dates. You are willing to pay the difference."),
    }
    return d, uid, rid, _read_actions(uid, rid) + writes, [dec.detail], scenario


def case_baggage_add(rng):
    # The reservation usually already carries bags, and the user asks for N *more*. The agent then has
    # to add to what is there and work the free allowance out per passenger, rather than read one number
    # off the allowance table: the version of this case where the reservation started empty and the user
    # named a total passed 5 out of 5 in a reference run.
    d, uid, rid = _modify_reservation(rng, rng.choice(adb.CABINS), n_pax=rng.randint(1, 3))
    res = d.reservations[rid]
    free = R.free_baggage(d.delta(), res)
    # The request has to land near the free allowance or the allowance table never gets consulted: the
    # earlier version drew the number of bags independently, so a gold member in business (eight free)
    # was asked for two and "no charge" was right without doing the arithmetic. Both generated tasks
    # came out that way. The regime is picked first and the request derived from it.
    #
    # "under" and "at" are kept in the mix on purpose. If every baggage task ended in a charge, "always
    # charge" would score as well as reading the table, and that is the shortcut this case exists to
    # deny. They are only drawable when the allowance leaves room for them.
    regimes, weights = ["over"], [6]
    if free >= 1:
        regimes.append("at"); weights.append(2)
    if free >= 2:
        regimes.append("under"); weights.append(2)
    regime = rng.choices(regimes, weights=weights)[0]
    if regime == "over":
        existing = rng.randint(0, free)
        total = free + rng.randint(1, 3)
    elif regime == "at":
        existing = rng.randint(0, free - 1)
        total = free
    else:
        existing = rng.randint(0, free - 2)
        total = rng.randint(existing + 1, free - 1)
    res["total_baggages"] = existing
    more = total - existing
    dec = R.baggage_charge(d.delta(), res, total)
    if not dec.allowed:
        raise BuildError(dec.reason)
    pay = next(p for p, m in d.users[uid]["payment_methods"].items() if m["source"] in ("credit_card", "gift_card"))
    writes = [{"name": "update_reservation_baggages",
               "arguments": {"reservation_id": rid, "total_baggages": total,
                             "nonfree_baggages": dec.extra["nonfree_baggages"], "payment_id": pay}}]
    scenario = {
        "reason_for_call": (f"You want to add {more} more checked bag(s) to reservation {rid}"
                            + (f", which already has {existing}." if existing else ".")),
        "known_info": f"You are {d.users[uid]['name']['first_name']} {d.users[uid]['name']['last_name']}. Your user id is {uid}.",
        "task_instructions": user_sim.instructions(rng, n=2, core="You are willing to pay for any bags over your allowance."),
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
        "task_instructions": user_sim.instructions(rng, n=2, core=core_with_contingency("", dec, rng), refusable=True),
    }
    return d, uid, rid, _read_actions(uid, rid), [dec.detail], scenario


def case_insurance_add_denied(rng):
    d, uid, rid = _modify_reservation(rng, rng.choice(["economy", "business"]), insurance="no")
    dec = R.can_add_insurance()
    scenario = {
        "reason_for_call": f"You want to add travel insurance to reservation {rid}.",
        "known_info": f"You are {d.users[uid]['name']['first_name']} {d.users[uid]['name']['last_name']}. Your user id is {uid}.",
        "task_instructions": user_sim.instructions(rng, n=2, core=core_with_contingency("You are willing to pay for it.", dec, rng), refusable=True),
    }
    return d, uid, rid, _read_actions(uid, rid), [dec.detail], scenario


def case_passenger_count_denied(rng):
    d, uid, rid = _modify_reservation(rng, rng.choice(["economy", "business"]), n_pax=2)
    dec = R.can_change_passenger_count()
    scenario = {
        "reason_for_call": f"One of the two passengers on reservation {rid} can no longer come, so you want to drop them.",
        "known_info": f"You are {d.users[uid]['name']['first_name']} {d.users[uid]['name']['last_name']}. Your user id is {uid}.",
        "task_instructions": user_sim.instructions(rng, n=2, core=core_with_contingency("", dec, rng), refusable=True),
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
        "reason_for_call": f"The second passenger on reservation {rid} changed, you want to put {new['first_name']} {new['last_name']} on it instead.",
        "known_info": (f"You are {d.users[uid]['name']['first_name']} {d.users[uid]['name']['last_name']}. Your user id is {uid}. "
                       f"The new passenger is {new['first_name']} {new['last_name']}, born {new['dob']}."),
        "task_instructions": user_sim.instructions(rng, n=2, core="The number of passengers stays the same."),
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
        "reason_for_call": (f"The airline cancelled a flight on reservation {rid} and you want "
                            f"compensation for the trouble."),
        "known_info": f"You are {d.users[uid]['name']['first_name']} {d.users[uid]['name']['last_name']}. Your user id is {uid}.",
        "task_instructions": user_sim.instructions(rng, n=2, core="You explicitly ask for compensation. You do not want to cancel the rest of the trip."),
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
        "task_instructions": user_sim.instructions(rng, n=2, core=core_with_contingency("You ask for compensation directly.", dec, rng), refusable=True),
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
    dep = d.flights[fn]["scheduled_departure_time_est"][:5]
    pax = d.passengers(uid, n_pax)
    # Every passenger's name and date of birth has to be something the user can actually say: the
    # policy makes the agent collect all three per passenger, and a probe run failed bookings where the
    # second passenger's date of birth existed only in the expected action, so the simulator invented one.
    others = "; ".join(f"{p['first_name']} {p['last_name']}, born {p['dob']}" for p in pax[1:])
    pay = next(p for p, m in d.users[uid]["payment_methods"].items() if m["source"] == "credit_card")
    dec = R.can_book(d.delta(), d.users[uid], cabin, pax, [pay], ["available"])
    if not dec.allowed:
        raise BuildError(dec.reason)
    free_per_pax = R.r["baggage"]["free_allowance"][d.users[uid]["membership"]][cabin]
    writes = [{"name": "book_reservation", "arguments": {
        "user_id": uid, "origin": o, "destination": dst, "flight_type": "one_way", "cabin": cabin,
        "flights": [{"flight_number": fn, "date": date}],
        "passengers": pax,
        "payment_methods": [{"payment_id": pay, "amount": int(entry["prices"][cabin] * n_pax)}],
        "total_baggages": free_per_pax * n_pax, "nonfree_baggages": 0, "insurance": "no"}}]
    scenario = {
        "reason_for_call": (f"You want to book flight {fn} from {o} to {dst} on {date}, the one departing "
                            f"{dep}, in {cabin.replace('_', ' ')}, for {n_pax} passenger(s)."),
        "known_info": (f"You are {d.users[uid]['name']['first_name']} {d.users[uid]['name']['last_name']}. "
                       f"Your user id is {uid}." + (f" The other passenger is {others}." if others else "")),
        "task_instructions": user_sim.instructions(rng, n=2, core="You only want the free checked bags you are entitled to, and you do not want travel insurance."),
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
        "task_instructions": user_sim.instructions(rng, n=2, core="Both reservations should be cancelled."),
    }
    return d, uid, rids[0], reads + writes, nl, scenario


# ---- composite cases: two or three rules inside one conversation ----
def case_upgrade_then_baggage(rng):
    """Upgrade the cabin, then add bags whose free allowance follows the *new* cabin. This is the shape
    of benchmark tasks 17 and 22, and the reason the rule engine evaluates baggage post-upgrade."""
    d, uid, rid = _modify_reservation(rng, rng.choice(["basic_economy", "economy"]))
    res = d.reservations[rid]
    dec_cabin = R.can_change_cabin(d.delta(), res)
    if not dec_cabin.allowed:
        raise BuildError(dec_cabin.reason)
    new_cabin = "business" if res["cabin"] == "economy" else "economy"
    after = {**res, "cabin": new_cabin}
    free = R.free_baggage(d.delta(), after)
    total = free + rng.randint(1, 2)
    dec_bag = R.baggage_charge(d.delta(), after, total)
    if not dec_bag.allowed:
        raise BuildError(dec_bag.reason)
    pay = next(p for p, m in d.users[uid]["payment_methods"].items() if m["source"] in ("credit_card", "gift_card"))
    writes = [
        {"name": "update_reservation_flights",
         "arguments": {"reservation_id": rid, "cabin": new_cabin,
                       "flights": [{"flight_number": f["flight_number"], "date": f["date"]} for f in res["flights"]],
                       "payment_id": pay}},
        {"name": "update_reservation_baggages",
         "arguments": {"reservation_id": rid, "total_baggages": total,
                       "nonfree_baggages": dec_bag.extra["nonfree_baggages"], "payment_id": pay}},
    ]
    scenario = {
        "reason_for_call": (f"For reservation {rid} you want two things: move it up to the "
                            f"{new_cabin.replace('_', ' ')} cabin, and have {total} checked bags in total."),
        "known_info": f"You are {d.users[uid]['name']['first_name']} {d.users[uid]['name']['last_name']}. Your user id is {uid}.",
        "task_instructions": user_sim.instructions(
            rng, "Mention both things. Keep the same flights and dates."),
    }
    return d, uid, rid, _read_actions(uid, rid) + writes, [dec_cabin.detail, dec_bag.detail], scenario


def case_change_flights_then_baggage(rng):
    """Move a flight and add bags in the same conversation."""
    d, uid, rid = _modify_reservation(rng, rng.choice(["economy", "business"]))
    res = d.reservations[rid]
    dec = R.can_change_flights(d.delta(), res)
    if not dec.allowed:
        raise BuildError(dec.reason)
    seg = res["flights"][0]
    new_date = adb.date_before(rng, res["flights"][1]["date"] if len(res["flights"]) > 1 else None, 3, 20)
    if new_date is None:
        raise BuildError("no date leaves the outbound before the return")
    alts = d.add_alternative_flights(seg["origin"], seg["destination"], new_date, 2, len(res["passengers"]))
    chosen = alts[0]
    dep = d.flights[chosen]["scheduled_departure_time_est"][:5]
    pay = next(p for p, m in d.users[uid]["payment_methods"].items() if m["source"] in ("credit_card", "gift_card"))
    new_flights = [{"flight_number": chosen, "date": new_date}] + \
                  [{"flight_number": f["flight_number"], "date": f["date"]} for f in res["flights"][1:]]
    free = R.free_baggage(d.delta(), res)
    total = free + rng.randint(1, 2)
    dec_bag = R.baggage_charge(d.delta(), res, total)
    writes = [
        {"name": "update_reservation_flights",
         "arguments": {"reservation_id": rid, "cabin": res["cabin"], "flights": new_flights, "payment_id": pay}},
        {"name": "update_reservation_baggages",
         "arguments": {"reservation_id": rid, "total_baggages": total,
                       "nonfree_baggages": dec_bag.extra["nonfree_baggages"], "payment_id": pay}},
    ]
    reads = _read_actions(uid, rid) + [
        {"name": "search_direct_flight", "arguments": {"origin": seg["origin"], "destination": seg["destination"], "date": new_date}}]
    scenario = {
        "reason_for_call": (f"On reservation {rid} you want to move the outbound flight to flight {chosen} "
                            f"on {new_date}, the one departing {dep}, and while the agent is at it, have "
                            f"{total} checked bags in total."),
        "known_info": f"You are {d.users[uid]['name']['first_name']} {d.users[uid]['name']['last_name']}. Your user id is {uid}.",
        "task_instructions": user_sim.instructions(
            rng, "The return flight stays as it is."),
    }
    return d, uid, rid, reads + writes, [dec.detail, dec_bag.detail], scenario


def case_cancel_then_compensation(rng):
    """The airline cancelled a flight: the reservation may be cancelled, and the user who asks is owed a
    certificate. Two rules, one conversation."""
    d = adb.AirlineDB(rng)
    uid = d.add_user(rng.choice(["silver", "gold"]), ["credit_card", "gift_card"])
    o, dst = adb.city_pair(rng)
    rid = d.add_reservation(uid, rng.choice(["economy", "business"]), "yes", adb.created_long_ago(rng),
                            [(o, dst, adb.future_date(rng), "cancelled"), (dst, o, adb.future_date(rng), "available")],
                            rng.randint(1, 3))
    res = d.reservations[rid]
    dec_c = R.can_cancel(d.delta(), res, "other")
    dec_k = R.compensation(d.delta(), res, "cancelled_flight", user_asked=True, changed_or_cancelled=True)
    if not (dec_c.allowed and dec_k.allowed):
        raise BuildError(f"{dec_c.reason}/{dec_k.reason}")
    writes = [{"name": "cancel_reservation", "arguments": {"reservation_id": rid}},
              {"name": "send_certificate", "arguments": {"user_id": uid, "amount": dec_k.extra["amount"]}}]
    scenario = {
        "reason_for_call": (f"The airline cancelled a flight on reservation {rid}. You want the whole "
                            f"reservation cancelled, and you want compensation for the trouble."),
        "known_info": f"You are {d.users[uid]['name']['first_name']} {d.users[uid]['name']['last_name']}. Your user id is {uid}.",
        "task_instructions": user_sim.instructions(rng, n=2, core="You ask for compensation explicitly, after the cancellation is done."),
    }
    return d, uid, rid, _read_actions(uid, rid) + _status_reads(d, rid) + writes, [dec_c.detail, dec_k.detail], scenario


def case_cancel_unknown_reservation(rng):
    """The user knows the trip but not the reservation id, so the agent has to go through the profile.
    Benchmark task 42 reads seven reservations before acting on two."""
    d = adb.AirlineDB(rng)
    uid = d.add_user(rng.choice(adb.MEMBERSHIPS), ["credit_card", "gift_card"])
    rids = []
    for i in range(3):
        o, dst = adb.city_pair(rng)
        cabin = "business" if i == 0 else rng.choice(["basic_economy", "economy"])
        rids.append(d.add_reservation(uid, cabin, "no", adb.created_long_ago(rng),
                                      [(o, dst, adb.future_date(rng), "available")], rng.randint(1, 2)))
    target = rids[0]
    res = d.reservations[target]
    dec = R.can_cancel(d.delta(), res, "other")
    if not dec.allowed:
        raise BuildError(dec.reason)
    seg = res["flights"][0]
    reads = [{"name": "get_user_details", "arguments": {"user_id": uid}}]
    order = rids[:]
    rng.shuffle(order)
    reads += [{"name": "get_reservation_details", "arguments": {"reservation_id": r}} for r in order]
    writes = [{"name": "cancel_reservation", "arguments": {"reservation_id": target}}]
    scenario = {
        "reason_for_call": (f"You want to cancel the trip from {seg['origin']} to {seg['destination']} on "
                            f"{seg['date']}. You do not remember the reservation number."),
        "known_info": f"You are {d.users[uid]['name']['first_name']} {d.users[uid]['name']['last_name']}. Your user id is {uid}.",
        "task_instructions": user_sim.instructions(
            rng, "Only that trip should be cancelled. Your other bookings stay as they are."),
    }
    return d, uid, target, reads + writes, [dec.detail], scenario


def case_composed(rng):
    """Two or three rules in one conversation, drawn from the composable units.

    Units act on their own reservation unless a pair is declared to share one, which is where the
    combination is more than the sum of its parts: upgrading the cabin changes the free baggage
    allowance, so an agent that computes the allowance from the original cabin gets it wrong."""
    import itertools

    d = adb.AirlineDB(rng)
    uid = d.add_user(rng.choice(adb.MEMBERSHIPS), ["credit_card", "gift_card"])
    k = rng.choices([2, 3], weights=[0.65, 0.35], k=1)[0]
    names = None
    for _ in range(40):
        cand = tuple(sorted(rng.sample(sorted(U.UNITS), k)))
        # At least one unit must produce a write, so a composed task is "do this, and refuse that"
        # rather than a second way of writing a pure refusal.
        if U.compatible(cand) and set(cand) & U.ACTION_UNITS:
            names = cand
            break
    if names is None:
        raise BuildError("no compatible unit combination")
    # Settled before any unit runs, because more than one unit reads it and a unit that set it itself
    # invalidated whatever an earlier unit had already computed from it.
    d.users[uid]["membership"] = rng.choice(U.required_memberships(names))

    # If a declared shared-record pair is present, build it in its declared order on one reservation.
    order = list(names)
    shared = next((p for p in U.SHARED_RECORD if set(p) <= set(names)), None)
    if shared:
        order = [n for n in order if n not in shared]
        order = list(shared) + order

    results, shared_rid = [], None
    for name in order:
        rid = shared_rid if (shared and name == shared[1]) else None
        try:
            res = U.UNITS[name](d, uid, rng, rid)
        except U.UnitError as e:
            raise BuildError(f"{name}: {e}") from e
        if shared and name == shared[0]:
            shared_rid = res.rid
        results.append((name, res))

    reads = [{"name": "get_user_details", "arguments": {"user_id": uid}}]
    seen = set()
    for _, r in results:
        if r.rid and r.rid not in seen:
            seen.add(r.rid)
            reads.append({"name": "get_reservation_details", "arguments": {"reservation_id": r.rid}})
    writes, nl, anchors, constraints = [], [], [], []
    for _, r in results:
        reads.extend(x for x in r.extra_reads if x["name"] != "transfer_to_human_agents")
        writes.extend(x for x in r.extra_reads if x["name"] == "transfer_to_human_agents")
        writes.extend(r.writes)
        nl.extend(r.nl)
        if r.anchor and r.anchor not in anchors:
            anchors.append(r.anchor)
        if r.constraint:
            constraints.append(r.constraint)
    for _, r in results:
        extra = REASON_ANCHOR.get(getattr(r.dec, "reason", None))
        if extra and extra not in anchors:
            anchors.append(extra)

    asks = [r.request for _, r in results]
    joined = "; ".join(asks[:-1]) + "; and " + asks[-1] if len(asks) > 2 else " and ".join(asks)
    name = d.users[uid]["name"]
    scenario = {
        "reason_for_call": f"You have {len(asks)} things to sort out: {joined}.",
        "known_info": f"You are {name['first_name']} {name['last_name']}. Your user id is {uid}.",
        "task_instructions": user_sim.instructions(rng, n=2, core=" ".join(constraints)),
        "_anchors": anchors,
        "_units": list(names),
    }
    return d, uid, results[0][1].rid, reads + writes, nl, scenario


# ---------------------------------------------------------------------------
# Cases that require a decision, not an execution
# ---------------------------------------------------------------------------
# A reference run put nine airline case types at a pass rate of exactly 1.00. They were not hard: the
# account held one record, it was the record the user named, and the rule applied to it unconditionally.
# The agent had to execute, not decide.
#
# What the reference set's own hardest category looks like is the opposite: its teacher scores 0.20 on
# cancellation, and those tasks hand the agent several bookings and ask for all of them, so the rule has
# to be applied one reservation at a time and the ineligible ones left alone.

def case_cancel_mixed_eligibility(rng):
    """Several upcoming bookings, only some cancellable, and the user asks for all of them.

    The expected actions are still unique: the rules decide exactly which reservations may go. What the
    agent cannot do is act on the request as a whole."""
    d = adb.AirlineDB(rng)
    uid = d.add_user(rng.choice(adb.MEMBERSHIPS), ["credit_card", "gift_card"])
    # The filler is off for this case, so the distractors are built here, under the same rule check.
    plan = ["eligible"] * rng.randint(1, 2) + ["ineligible"] * rng.randint(2, 4)
    if rng.random() < 0.5:
        plan.append(rng.choice(["eligible", "ineligible"]))
    rng.shuffle(plan)
    rids, eligible, nl, told = [], [], [], []
    for kind in plan:
        o, dst = adb.city_pair(rng)
        if kind == "eligible":
            style = rng.choice(["business", "recent", "airline_cancelled"])
            cabin = "business" if style == "business" else rng.choice(["basic_economy", "economy"])
            created = adb.created_within_24h(rng) if style == "recent" else adb.created_long_ago(rng)
            statuses = ["cancelled", "available"] if style == "airline_cancelled" else ["available"]
        else:
            cabin, created, statuses = rng.choice(["basic_economy", "economy"]), adb.created_long_ago(rng), ["available"]
        segs = []
        oo, dd = o, dst
        for st in statuses:
            segs.append((oo, dd, adb.future_date(rng), st))
            oo, dd = dd, oo
        rid = d.add_reservation(uid, cabin, "no", created, segs, rng.randint(1, 2))
        dec = R.can_cancel(d.delta(), d.reservations[rid], "other")
        if kind == "eligible" and style == "airline_cancelled":
            # A cancelled flight lives in the flights table, and no read tool shows it to the agent:
            # policy.md has the agent *ask* for the reason. So the caller has to know this one.
            told.append(f"The airline cancelled one of the flights on reservation {rid}.")
        if dec.allowed != (kind == "eligible"):
            raise BuildError(f"reservation came out {dec.reason}, wanted {kind}")
        rids.append(rid)
        if dec.allowed:
            eligible.append(rid)
            nl.append(f"Agent should cancel reservation {rid} because {dec.reason.replace('_', ' ')}.")
        else:
            nl.append(f"Agent should not cancel reservation {rid}: it meets no cancellation condition.")
    if not eligible or len(eligible) == len(rids):
        raise BuildError("this case needs both kinds")
    reads = [{"name": "get_user_details", "arguments": {"user_id": uid}}]
    reads += [{"name": "get_reservation_details", "arguments": {"reservation_id": r}} for r in rids]
    writes = [{"name": "cancel_reservation", "arguments": {"reservation_id": r}} for r in eligible]
    name = d.users[uid]["name"]
    scenario = {
        "reason_for_call": f"You want to cancel all {len(rids)} of your upcoming trips.",
        "known_info": " ".join([f"You are {name['first_name']} {name['last_name']}. Your user id is {uid}."] + told),
        "task_instructions": user_sim.instructions(
            rng, n=2, core=("Even if the agent says some of them cannot be refunded, you still want everything "
                            "cancelled that can be. You do not know which of them qualify."), refusable=True),
        "_decision": None,
    }
    return d, uid, rids[0], reads + writes, nl, scenario


def case_compensation_facts_denied(rng):
    """The user is eligible for compensation and asks for it, but the facts do not support the claim:
    no flight in the reservation was cancelled. Denial on the evidence rather than on membership, which
    is the branch the single-condition denial case never reached."""
    d = adb.AirlineDB(rng)
    uid = d.add_user(rng.choice(["silver", "gold"]), ["credit_card", "gift_card"])
    o, dst = adb.city_pair(rng)
    rid = d.add_reservation(uid, rng.choice(["economy", "business"]), "yes", adb.created_long_ago(rng),
                            [(o, dst, adb.future_date(rng), "available"),
                             (dst, o, adb.future_date(rng), "available")], rng.randint(1, 3))
    dec = R.compensation(d.delta(), d.reservations[rid], "cancelled_flight", user_asked=True, changed_or_cancelled=True)
    if dec.allowed or dec.reason != "facts_not_confirmed":
        raise BuildError(f"wanted facts_not_confirmed, got {dec.reason}")
    name = d.users[uid]["name"]
    scenario = {
        "reason_for_call": (f"You believe the airline cancelled a flight on reservation {rid} and you want "
                            f"compensation for it."),
        "known_info": f"You are {name['first_name']} {name['last_name']}. Your user id is {uid}.",
        "task_instructions": user_sim.instructions(
            rng, n=2, core=("You are sure you remember a cancellation. If the agent checks and tells you none of "
                            "your flights was cancelled, accept it."), refusable=True),
        "_decision": None,
    }
    return d, uid, rid, _read_actions(uid, rid) + _status_reads(d, rid), [dec.detail], scenario


def check_itinerary_stays_in_order(raw_actions: list[dict]) -> None:
    """An expected flight change must leave the trip in date order.

    Drawing the new outbound date freely produced a reservation whose outbound left three days after
    the return, and the agent refused to book it, which was the right reading of the itinerary."""
    for a in raw_actions:
        if a["name"] != "update_reservation_flights":
            continue
        dates = [f["date"] for f in a["arguments"].get("flights", [])]
        if dates != sorted(dates):
            raise BuildError(f"expected itinerary is out of order: {dates}")


def check_new_flights_are_named(d, raw_actions: list[dict], scen: dict) -> None:
    """A flight the agent has to book must be named in the request.

    "Move it to <date>" is satisfied by any flight that day, including tau2-bench's own inventory,
    which the default database still holds: an agent that booked HAT115 instead of the NVA flight the
    generator picked was answering the question as asked. Two cases already named the flight; this
    makes it a property of the set rather than of whoever wrote the case."""
    asked = scen["reason_for_call"] + " " + scen.get("task_instructions", "")
    for a in raw_actions:
        if a["name"] != "update_reservation_flights":
            continue
        rid = a["arguments"].get("reservation_id")
        res = d.reservations.get(rid, {})
        # A segment the reservation already holds is being kept, and keeping it needs no naming; a
        # segment the agent has to pick out of the timetable does.
        held = {f["flight_number"] for f in res.get("flights", [])}
        for f in a["arguments"].get("flights", []):
            fn = f.get("flight_number")
            if fn and fn not in held and fn not in asked:
                raise BuildError(f"expected action books {fn}, which the request never names")


def name_payment_method(d, uid: str, raw_actions: list[dict], scen: dict) -> None:
    """Say which card to use whenever the account holds more than one.

    Every write that moves money takes a payment_id, the expected action names one, and the database
    check compares payment history, so a customer holding both a credit card and a gift card gives the
    task two correct answers. The agent that picked the other one was right and scored zero. policy.md
    only requires the method to be in the profile, so the request is what has to pin it down."""
    pids: list[str] = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "payment_id" and isinstance(v, str):
                    pids.append(v)
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(raw_actions)
    methods = d.users.get(uid, {}).get("payment_methods", {})
    if not pids or len(methods) < 2:
        return
    phrases = []
    for pid in dict.fromkeys(pids):
        m = methods.get(pid)
        if not m:
            continue
        if m["source"] == "credit_card":
            phrases.append(f"your {m['brand']} card ending {m['last_four']}")
        else:
            phrases.append(f"your {m['source'].replace('_', ' ')} ending {pid[-4:]}")
    if phrases:
        # First, not last: it follows the request the customer came with, and a payment clause parked
        # after the persona's own quirks competes with them for the simulator's attention.
        scen["task_instructions"] = ("Pay with " + " and ".join(phrases) + ". "
                                     + scen["task_instructions"].lstrip())


def check_baggage_matches_rule(d, raw_actions: list[dict]) -> None:
    """Every expected baggage write must agree with the allowance as it stands at that point.

    The allowance depends on the booking user's membership, the cabin and the passenger count, and a
    composed task can change the first two before the baggage write happens. Working the number out
    when the unit was built rather than where the write lands left five of seventeen expectations
    stale, and a stale one is worse than a missing task: the agent that reads the table correctly is
    the one that scores zero."""
    db = d.delta()
    cabins = {rid: r["cabin"] for rid, r in d.reservations.items()}
    for a in raw_actions:
        arg = a.get("arguments", {})
        rid = arg.get("reservation_id")
        if a["name"] == "update_reservation_flights" and rid in cabins and arg.get("cabin"):
            cabins[rid] = arg["cabin"]                      # a cabin change moves the allowance
        elif a["name"] == "update_reservation_baggages" and rid in cabins:
            res = dict(d.reservations[rid], cabin=cabins[rid])
            free = R.free_baggage(db, res)
            want = max(0, arg["total_baggages"] - free)
            if want != arg["nonfree_baggages"]:
                raise BuildError(f"{rid}: {arg['total_baggages']} bags with {free} free means "
                                 f"{want} paid, expected action says {arg['nonfree_baggages']}")


def check_all_cancellable_are_expected(d, uid: str, raw_actions: list[dict]) -> None:
    """Every reservation the rules would let go must be in the expected actions, and no other.

    A request phrased over the whole account ("cancel all my trips") is only well posed if the
    expected action list is closed under the rule. It was not: the history filler could add a
    business-cabin reservation, which the rule allows and the list did not name, so an agent that
    got it right scored zero."""
    allowed = {rid for rid in d.users[uid]["reservations"]
               if R.can_cancel(d.delta(), d.reservations[rid], "other").allowed}
    expected = {a["arguments"]["reservation_id"] for a in raw_actions if a["name"] == "cancel_reservation"}
    if allowed != expected:
        raise BuildError(f"cancellable set {sorted(allowed)} != expected {sorted(expected)}")


CASES = [
    Case("composed", case_composed, 130, "composite", single_record=False),
    Case("cancel_two_reservations", case_cancel_two_reservations, 5, "cancel", single_record=False),
    Case("cancel_mixed_eligibility", case_cancel_mixed_eligibility, 22, "cancel", single_record=False,
         history=False, validate=check_all_cancellable_are_expected),
    Case("compensation_facts_denied", case_compensation_facts_denied, 8, "refuse"),
    Case("cancel_unknown_reservation", case_cancel_unknown_reservation, 10, "composite", single_record=False),
    Case("upgrade_then_baggage", case_upgrade_then_baggage, 7, "composite"),
    Case("change_flights_then_baggage", case_change_flights_then_baggage, 6, "composite"),
    Case("cancel_then_compensation", case_cancel_then_compensation, 6, "composite"),
    Case("cancel_within_24h", case_cancel_within_24h, 6, "cancel"),
    Case("cancel_business", case_cancel_business, 6, "cancel"),
    Case("cancel_insurance_health", case_cancel_insurance_health, 6, "cancel"),
    Case("cancel_airline_cancelled", case_cancel_airline_cancelled, 5, "cancel"),
    Case("cancel_denied", case_cancel_denied, 8, "refuse"),
    Case("cancel_denied_flown", case_cancel_denied_flown, 5, "refuse"),
    Case("change_flights", case_change_flights, 10, "modify"),
    Case("change_flights_denied_basic_economy", case_change_flights_denied_basic_economy, 8, "refuse"),
    Case("change_cabin", case_change_cabin, 8, "modify"),
    Case("baggage_add", case_baggage_add, 16, "modify"),
    Case("baggage_remove_denied", case_baggage_remove_denied, 4, "refuse"),
    Case("insurance_add_denied", case_insurance_add_denied, 4, "refuse"),
    Case("passenger_count_denied", case_passenger_count_denied, 4, "refuse"),
    Case("passenger_change", case_passenger_change, 4, "modify"),
    Case("compensation_cancelled_flight", case_compensation_cancelled_flight, 5, "compensation"),
    Case("compensation_denied", case_compensation_denied, 5, "refuse"),
    Case("book", case_book, 8, "book", single_record=False),
]


def build_task(idx: int, tag: str, case: Case, persona_name: str, rng: random.Random,
               history: bool = True, unknown_id: bool = False) -> tuple[dict, dict]:
    d, uid, rid, raw_actions, nl, scen = case.fn(rng)
    # The rule reason, when the case recorded one, so a well-attested reason string can be added.
    dec = scen.pop("_decision", None)
    hid = False
    if history and case.history and uid in d.users:
        add_history(d, uid, rng)
    check_itinerary_stays_in_order(raw_actions)
    check_new_flights_are_named(d, raw_actions, scen)
    name_payment_method(d, uid, raw_actions, scen)
    check_baggage_matches_rule(d, raw_actions)
    if case.validate:
        case.validate(d, uid, raw_actions)
    if unknown_id and case.single_record and rid:
        scen, raw_actions = hide_id(scen, d, uid, rid, raw_actions, rng)
        hid = rid not in scen["reason_for_call"]
    persona_text = user_sim.render_persona(persona_name, rng, "555-555-0100") if persona_name != "None" else None
    units = scen.pop("_units", None)
    label = f"composed:{'+'.join(units)}" if units else case.name
    name = f"[{label}][PERSONA:{persona_name}][GEN:{tag}]"
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
            actions=_acts(raw_actions), env_assertions=[],
            communicate_info=communicate_info(case.name, dec, scen),
            nl_assertions=nl, reward_basis=["DB", "COMMUNICATE"]),
        domain=DOMAIN,
        tau_description=schema.TauDescription(purpose=case.name.replace("_", " "), relevant_policies=None, notes=None),
    ).to_dict()
    data["evaluation_criteria"]["env_assertions"] = None
    meta = {"idx": idx, "id": name, "case": case.name, "group": case.group, "persona": persona_name,
            "communicate_info": communicate_info(case.name, dec, scen), "units": units, "unknown_id": hid, "n_user_records": len(d.users.get(uid, {}).get("reservations", [])),
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


def quotas(n: int, weights: list[float]) -> list[int]:
    """Largest-remainder allocation, so the realised case mix matches the intended one exactly.

    Drawing each task independently leaves the mix to chance: at n=150 the share of any one group
    moves by about eight points across runs, which makes `--refuse-share` a suggestion rather than a
    setting. Quotas make it a setting."""
    tot = sum(weights)
    raw = [n * w / tot for w in weights]
    out = [int(x) for x in raw]
    rem = n - sum(out)
    for i in sorted(range(len(raw)), key=lambda i: -(raw[i] - out[i]))[:rem]:
        out[i] += 1
    return out


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
             unknown_id_share: float = 0.10, verbose: bool = False) -> tuple[list[dict], list[dict], dict]:
    rng = random.Random(seed)
    weights = rebalance(CASES, refuse_share)
    left = quotas(n, weights)
    pressure_plan = user_sim.plan_pressure(n, 0.40, rng)
    records, metas, stats = [], [], Counter()
    idx = 0
    attempts = 0
    while len(records) < n and attempts < n * 60:
        attempts += 1
        avail = [i for i, q in enumerate(left) if q > 0]
        if not avail:
            left = quotas(n - len(records), weights) if len(records) < n else []
            avail = [i for i, q in enumerate(left) if q > 0]
            if not avail:
                break
        ci = rng.choices(avail, weights=[left[i] for i in avail], k=1)[0]
        case = CASES[ci]
        persona = user_sim.sample_persona(rng, persona_mix)
        user_sim.PRESSURE_SHARE = 1.0 if pressure_plan[len(records)] else 0.0
        try:
            data, meta = build_task(idx, f"{seed}-{idx:05d}", case, persona, rng,
                                    history=True, unknown_id=rng.random() < unknown_id_share)
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
        left[ci] -= 1
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
    ap.add_argument("--unknown-id-share", type=float, default=0.10,
                    help="share of single-record tasks where the user cannot quote the record number")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    mix = json.loads(a.persona_mix) if a.persona_mix else {"None": 0.5, "Easy": 0.08, "Hard": 0.08, "Verbose": 0.06,
                                                           "Terse": 0.06, "NonNative": 0.06, "Impatient": 0.06,
                                                           "WrongNumberOnce": 0.04, "SideRequest": 0.03, "TechSavvy": 0.03}
    t0 = time.time()
    records, metas, stats = generate(a.n, a.seed, mix, a.refuse_share, a.unknown_id_share, a.verbose)
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
        "unknown_id_tasks": sum(1 for m in metas if m.get("unknown_id")),
        "records_per_user_median": sorted(m.get("n_user_records", 0) for m in metas)[len(metas) // 2] if metas else 0,
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
