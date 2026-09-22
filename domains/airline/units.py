"""Airline rules as composable units, so tasks can be built from combinations rather than one per rule.

One case per rule gives one decision per task, and the set saturates: 22 distinct decisions no matter
how many tasks are generated, because every extra task is the same decision with a different customer.
tau2-bench's telecom domain avoids this by composing atomic faults; its airline tasks avoid it by being
hand-written, several of them spanning many reservations at once.

A unit is one rule applied to one record: what to set up, what the engine decides, what a correct agent
then does, and how the user asks for it. Units on different records compose freely, which is the common
case and is what the reference set's multi-reservation tasks look like. A few pairs are declared to
share a record, and those are the interesting ones, because the first rule changes the second one's
inputs: upgrading the cabin changes the free baggage allowance, so the two cannot be reasoned about
independently.

Each unit reports the topic word the user used, so the composed task's must-mention strings stay
anchored the way the single-rule ones are.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from domains.airline import db as adb
from domains.airline.rules import AirlineRules, Decision

R = AirlineRules()


class UnitError(Exception):
    """This unit cannot be built on the account it was handed."""


@dataclass
class UnitResult:
    rid: Optional[str]                  # record the unit acts on, if any
    dec: Optional[Decision]
    writes: list[dict] = field(default_factory=list)
    extra_reads: list[dict] = field(default_factory=list)
    request: str = ""                   # one clause, joined into the user's opening
    anchor: str = ""                    # topic word, which the clause must contain
    nl: list[str] = field(default_factory=list)
    constraint: str = ""                # goes into task_instructions when it bears on correctness


def _pay(d, uid: str) -> str:
    return next(p for p, m in d.users[uid]["payment_methods"].items()
                if m["source"] in ("credit_card", "gift_card"))


def _reservation(d, uid, rng, cabin, insurance="no", statuses=("available", "available"),
                 created=None, n_pax=None) -> str:
    o, dst = adb.city_pair(rng)
    segs, oo, dd = [], o, dst
    for st in statuses:
        date = adb.past_date(rng) if st in ("landed", "flying") else adb.future_date(rng)
        segs.append((oo, dd, date, st))
        oo, dd = dd, oo
    return d.add_reservation(uid, cabin, insurance, created or adb.created_long_ago(rng), segs,
                             n_pax or rng.randint(1, 3))


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------
def u_cancel_ok(d, uid, rng, rid=None) -> UnitResult:
    rid = rid or _reservation(d, uid, rng, "business")
    dec = R.can_cancel(d.delta(), d.reservations[rid], "other")
    if not dec.allowed:
        raise UnitError(dec.reason)
    return UnitResult(rid, dec, [{"name": "cancel_reservation", "arguments": {"reservation_id": rid}}],
                      request=f"cancel reservation {rid}", anchor="cancel", nl=[dec.detail])


def u_cancel_denied(d, uid, rng, rid=None) -> UnitResult:
    rid = rid or _reservation(d, uid, rng, rng.choice(["basic_economy", "economy"]))
    dec = R.can_cancel(d.delta(), d.reservations[rid], "other")
    if dec.allowed:
        raise UnitError("expected this one to be refused")
    return UnitResult(rid, dec, [], request=f"cancel reservation {rid}", anchor="cancel", nl=[dec.detail])


def u_cancel_denied_flown(d, uid, rng, rid=None) -> UnitResult:
    rid = rid or _reservation(d, uid, rng, "business", statuses=("landed", "available"))
    dec = R.can_cancel(d.delta(), d.reservations[rid], "other")
    if dec.allowed or dec.reason != "flown_segment":
        raise UnitError("expected a flown segment to block this")
    return UnitResult(rid, dec,
                      extra_reads=[{"name": "transfer_to_human_agents",
                                    "arguments": {"summary": f"Part of reservation {rid} has already been flown."}}],
                      request=f"cancel reservation {rid}", anchor="cancel",
                      nl=[dec.detail, "Agent should transfer the user to a human agent."])


def u_change_flights_ok(d, uid, rng, rid=None) -> UnitResult:
    rid = rid or _reservation(d, uid, rng, rng.choice(["economy", "business"]))
    res = d.reservations[rid]
    dec = R.can_change_flights(d.delta(), res)
    if not dec.allowed:
        raise UnitError(dec.reason)
    seg = res["flights"][0]
    new_date = adb.future_date(rng, 3, 20)
    alts = d.add_alternative_flights(seg["origin"], seg["destination"], new_date, 2, len(res["passengers"]))
    dep = d.flights[alts[0]]["scheduled_departure_time_est"][:5]
    flights = [{"flight_number": alts[0], "date": new_date}] + \
              [{"flight_number": f["flight_number"], "date": f["date"]} for f in res["flights"][1:]]
    return UnitResult(rid, dec,
                      [{"name": "update_reservation_flights",
                        "arguments": {"reservation_id": rid, "cabin": res["cabin"], "flights": flights,
                                      "payment_id": _pay(d, uid)}}],
                      extra_reads=[{"name": "search_direct_flight",
                                    "arguments": {"origin": seg["origin"], "destination": seg["destination"],
                                                  "date": new_date}}],
                      # The flight is named: "to <date>" alone is satisfied by any flight that day,
                      # including tau2-bench's own inventory, which the default database still holds.
                      request=(f"move the outbound flight on reservation {rid} to flight {alts[0]} on "
                               f"{new_date}, the one departing {dep}"),
                      anchor="flight", nl=[dec.detail],
                      constraint="The return flight stays as it is.")


def u_change_flights_denied(d, uid, rng, rid=None) -> UnitResult:
    rid = rid or _reservation(d, uid, rng, "basic_economy")
    res = d.reservations[rid]
    dec = R.can_change_flights(d.delta(), res)
    if dec.allowed:
        raise UnitError("basic economy should not be changeable")
    seg = res["flights"][0]
    d.add_alternative_flights(seg["origin"], seg["destination"], adb.future_date(rng, 3, 20), 2,
                              len(res["passengers"]))
    return UnitResult(rid, dec, [], request=f"move your flight on reservation {rid} to a later date",
                      anchor="flight", nl=[dec.detail])


def u_change_cabin_ok(d, uid, rng, rid=None) -> UnitResult:
    rid = rid or _reservation(d, uid, rng, rng.choice(["basic_economy", "economy"]))
    res = d.reservations[rid]
    dec = R.can_change_cabin(d.delta(), res)
    if not dec.allowed:
        raise UnitError(dec.reason)
    new_cabin = "business" if res["cabin"] == "economy" else "economy"
    return UnitResult(rid, dec,
                      [{"name": "update_reservation_flights",
                        "arguments": {"reservation_id": rid, "cabin": new_cabin,
                                      "flights": [{"flight_number": f["flight_number"], "date": f["date"]}
                                                  for f in res["flights"]],
                                      "payment_id": _pay(d, uid)}}],
                      request=f"move reservation {rid} up to the {new_cabin.replace('_', ' ')} cabin",
                      anchor="cabin", nl=[dec.detail],
                      constraint="Keep the same flights and dates.")


def u_baggage_add(d, uid, rng, rid=None) -> UnitResult:
    """After a cabin change on the same record the allowance follows the new cabin, which is why this
    unit reads the reservation as it stands rather than as it was created."""
    rid = rid or _reservation(d, uid, rng, rng.choice(["economy", "business"]))
    res = d.reservations[rid]
    free = R.free_baggage(d.delta(), res)
    total = free + rng.randint(1, 2)
    dec = R.baggage_charge(d.delta(), res, total)
    if not dec.allowed:
        raise UnitError(dec.reason)
    return UnitResult(rid, dec,
                      [{"name": "update_reservation_baggages",
                        "arguments": {"reservation_id": rid, "total_baggages": total,
                                      "nonfree_baggages": dec.extra["nonfree_baggages"],
                                      "payment_id": _pay(d, uid)}}],
                      request=f"have {total} checked bags in total on reservation {rid}",
                      anchor="bag", nl=[dec.detail])


def u_baggage_remove_denied(d, uid, rng, rid=None) -> UnitResult:
    rid = rid or _reservation(d, uid, rng, rng.choice(["economy", "business"]))
    d.reservations[rid]["total_baggages"] = 2
    dec = R.baggage_charge(d.delta(), d.reservations[rid], 0)
    if dec.allowed:
        raise UnitError("removing bags should be refused")
    return UnitResult(rid, dec, [], request=f"take the checked bags off reservation {rid} and get the money back",
                      anchor="bag", nl=[dec.detail])


def u_insurance_denied(d, uid, rng, rid=None) -> UnitResult:
    rid = rid or _reservation(d, uid, rng, rng.choice(["economy", "business"]))
    dec = R.can_add_insurance()
    return UnitResult(rid, dec, [], request=f"add travel insurance to reservation {rid}",
                      anchor="insurance", nl=[dec.detail])


def u_passenger_change_ok(d, uid, rng, rid=None) -> UnitResult:
    rid = rid or _reservation(d, uid, rng, rng.choice(["economy", "business"]), n_pax=2)
    res = d.reservations[rid]
    new = dict(d.passengers(uid, 2)[1])
    new["first_name"] = rng.choice(["Dana", "Noor", "Rafael", "Ingrid", "Mila"])
    return UnitResult(rid, None,
                      [{"name": "update_reservation_passengers",
                        "arguments": {"reservation_id": rid, "passengers": [res["passengers"][0], new]}}],
                      request=(f"put {new['first_name']} {new['last_name']} on reservation {rid} as the "
                               f"second passenger instead"),
                      anchor="passenger",
                      nl=["Agent should replace the passenger without changing the number of passengers."],
                      constraint=(f"The new passenger is {new['first_name']} {new['last_name']}, born "
                                  f"{new['dob']}. The number of passengers stays the same."))


def u_passenger_count_denied(d, uid, rng, rid=None) -> UnitResult:
    rid = rid or _reservation(d, uid, rng, rng.choice(["economy", "business"]), n_pax=2)
    dec = R.can_change_passenger_count()
    return UnitResult(rid, dec, [],
                      request=f"drop one of the two passengers from reservation {rid}",
                      anchor="passenger", nl=[dec.detail])


def u_compensation_ok(d, uid, rng, rid=None) -> UnitResult:
    if rid is None:
        o, dst = adb.city_pair(rng)
        rid = d.add_reservation(uid, rng.choice(["economy", "business"]), "yes", adb.created_long_ago(rng),
                                [(o, dst, adb.future_date(rng), "cancelled"),
                                 (dst, o, adb.future_date(rng), "available")], rng.randint(1, 3))
    dec = R.compensation(d.delta(), d.reservations[rid], "cancelled_flight", True, True)
    if not dec.allowed:
        raise UnitError(dec.reason)
    return UnitResult(rid, dec,
                      [{"name": "send_certificate", "arguments": {"user_id": uid, "amount": dec.extra["amount"]}}],
                      request=f"get compensation for the flight the airline cancelled on reservation {rid}",
                      anchor="compensation", nl=[dec.detail])


def u_compensation_denied(d, uid, rng, rid=None) -> UnitResult:
    if rid is None:
        o, dst = adb.city_pair(rng)
        rid = d.add_reservation(uid, rng.choice(["basic_economy", "economy"]), "no",
                                adb.created_long_ago(rng), [(o, dst, adb.future_date(rng), "delayed")],
                                rng.randint(1, 2))
    dec = R.compensation(d.delta(), d.reservations[rid], "delayed_flight", True, False)
    if dec.allowed:
        raise UnitError("this one should not be eligible")
    return UnitResult(rid, dec, [],
                      request=f"get a travel certificate for the delay on reservation {rid}",
                      anchor="certificate", nl=[dec.detail])


UNITS: dict[str, Callable] = {
    "cancel_ok": u_cancel_ok, "cancel_denied": u_cancel_denied, "cancel_denied_flown": u_cancel_denied_flown,
    "change_flights_ok": u_change_flights_ok, "change_flights_denied": u_change_flights_denied,
    "change_cabin_ok": u_change_cabin_ok, "baggage_add": u_baggage_add,
    "baggage_remove_denied": u_baggage_remove_denied, "insurance_denied": u_insurance_denied,
    "passenger_change_ok": u_passenger_change_ok, "passenger_count_denied": u_passenger_count_denied,
    "compensation_ok": u_compensation_ok, "compensation_denied": u_compensation_denied,
}

# Units whose correct outcome includes a write. A composed task needs at least one, or the whole task
# collapses into "change nothing", which the single-rule refusal cases already cover and which would
# quietly push the set's share of no-write tasks above the reference.
ACTION_UNITS = {"cancel_ok", "change_flights_ok", "change_cabin_ok", "baggage_add",
                "passenger_change_ok", "compensation_ok"}

# Ordered pairs that act on one record instead of two. These are the combinations where the first
# rule changes what the second one computes, so an agent cannot treat them independently.
# What a unit needs the *user* to be. A unit used to set this itself, which silently invalidated any
# earlier unit that had already read it: composing baggage_add with compensation_ok moved the member
# from regular to gold after the free allowance had been worked out, so the expected number of paid
# bags was stale and an agent that computed it correctly scored zero. The composer now resolves these
# before any unit runs, and a combination with no membership in common is simply not drawn.
REQUIRES_MEMBERSHIP: dict[str, tuple[str, ...]] = {
    "compensation_ok": ("silver", "gold"),
    "compensation_denied": ("regular",),
}


def required_memberships(names: tuple[str, ...]) -> list[str]:
    """The memberships that satisfy every unit in the combination, in table order."""
    out = list(adb.MEMBERSHIPS)
    for n in names:
        req = REQUIRES_MEMBERSHIP.get(n)
        if req:
            out = [m for m in out if m in req]
    return out


SHARED_RECORD: list[tuple[str, str]] = [
    ("change_cabin_ok", "baggage_add"),     # the free allowance follows the new cabin
    ("change_flights_ok", "baggage_add"),   # same reservation, two chargeable changes
    ("compensation_ok", "cancel_ok"),       # the airline cancelled a flight: compensate, then cancel
]

# Pairs that must not appear together, whether or not they share a record.
INCOMPATIBLE: set[frozenset] = {
    frozenset({"cancel_ok", "cancel_denied"}),          # one account, contradictory expectations
    frozenset({"change_flights_ok", "change_flights_denied"}),
    frozenset({"compensation_ok", "compensation_denied"}),
    frozenset({"baggage_add", "baggage_remove_denied"}),
    frozenset({"passenger_change_ok", "passenger_count_denied"}),
}


def compatible(names: tuple[str, ...]) -> bool:
    if not required_memberships(names):
        return False
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if frozenset({a, b}) in INCOMPATIBLE:
                return False
    return True
