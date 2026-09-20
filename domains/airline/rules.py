"""Airline policy engine: the rule table in rules.yaml applied to concrete reservations.

Every decision returns why it was reached, so a generated task can put the reason into its
`nl_assertions` and a reviewer can trace it back to a clause in the policy document.

`test_rules.py` checks this engine against tau2-bench's own 50 airline tasks.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

RULES_PATH = Path(__file__).resolve().parent / "rules.yaml"


@dataclass
class Decision:
    allowed: bool
    reason: str                      # machine-readable key, e.g. "cabin_business" / "flown_segment"
    detail: str = ""                 # one human sentence, usable as an nl_assertion
    extra: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.allowed


def load_rules(path: Path | str = RULES_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text())


class AirlineRules:
    def __init__(self, rules: Optional[dict] = None):
        self.r = rules or load_rules()
        self.now = dt.datetime.fromisoformat(self.r["now"])

    # ---------- database helpers ----------
    @staticmethod
    def _get(db: Any, section: str, key: str):
        s = db[section] if isinstance(db, dict) else getattr(db, section)
        return s.get(key) if isinstance(s, dict) else None

    def flight_status(self, db, flight_number: str, date: str) -> Optional[str]:
        f = self._get(db, "flights", flight_number)
        if f is None:
            return None
        dates = f["dates"] if isinstance(f, dict) else f.dates
        d = dates.get(date)
        if d is None:
            return None
        return d["status"] if isinstance(d, dict) else getattr(d, "status", None)

    def segment_statuses(self, db, reservation: dict) -> list[Optional[str]]:
        return [self.flight_status(db, f["flight_number"], f["date"]) for f in reservation["flights"]]

    def has_flown(self, db, reservation: dict) -> bool:
        flown = set(self.r["cancel"]["flown_statuses"])
        return any(s in flown for s in self.segment_statuses(db, reservation))

    def any_airline_cancelled(self, db, reservation: dict) -> bool:
        return self.r["cancel"]["cancelled_status"] in self.segment_statuses(db, reservation)

    def within_24h(self, reservation: dict) -> bool:
        created = dt.datetime.fromisoformat(reservation["created_at"])
        return (self.now - created) <= dt.timedelta(hours=self.r["cancel"]["within_24h_hours"])

    def membership(self, db, reservation: dict) -> str:
        u = self._get(db, "users", reservation["user_id"])
        if u is None:
            return "regular"
        return (u.get("membership") if isinstance(u, dict) else getattr(u, "membership", "regular")) or "regular"

    # ---------- decisions ----------
    def can_cancel(self, db, reservation: dict, reason: str = "other") -> Decision:
        """policy.md, "Cancel flight"."""
        c = self.r["cancel"]
        if c["blocked_when_flown"] and self.has_flown(db, reservation):
            return Decision(False, "flown_segment",
                            "Agent should not cancel the reservation because part of it has already been flown, and should transfer to a human agent.")
        paths = []
        if self.within_24h(reservation):
            paths.append("within_24h")
        if self.any_airline_cancelled(db, reservation):
            paths.append("airline_cancelled")
        if reservation["cabin"] == "business":
            paths.append("cabin_business")
        if reservation.get("insurance") == "yes" and reason in c["insurance_covered_reasons"]:
            paths.append("insurance_covered")
        allowed = [p for p in paths if p in c["any_of"]]
        if allowed:
            why = {
                "within_24h": "it was booked within the last 24 hours",
                "airline_cancelled": "the airline cancelled a flight in it",
                "cabin_business": "it is a business reservation",
                "insurance_covered": "it has travel insurance and the reason is covered",
            }[allowed[0]]
            return Decision(True, allowed[0], f"Agent should cancel the reservation because {why}.",
                            {"all_paths": allowed})
        return Decision(False, "not_eligible",
                        "Agent should refuse to cancel the reservation because it does not meet any cancellation condition.")

    def can_change_flights(self, db, reservation: dict) -> Decision:
        """policy.md, "Modify flight" / "Change flights"."""
        m = self.r["modify_flights"]
        if reservation["cabin"] in m["forbidden_cabins"]:
            return Decision(False, "basic_economy",
                            "Agent should refuse to change the flights because basic economy reservations cannot be modified.")
        if m["blocked_when_flown"] and self.has_flown(db, reservation):
            return Decision(False, "flown_segment",
                            "Agent should not change the flights because part of the reservation has already been flown.")
        return Decision(True, "eligible", "Agent should change the flights as requested.")

    def can_change_cabin(self, db, reservation: dict) -> Decision:
        """policy.md, "Change cabin". Basic economy *can* change cabin, unlike changing flights."""
        cc = self.r["change_cabin"]
        if reservation["cabin"] in cc["forbidden_cabins"]:
            return Decision(False, "forbidden_cabin", "Agent should refuse to change the cabin for this reservation.")
        if cc["blocked_when_flown"] and self.has_flown(db, reservation):
            return Decision(False, "flown_segment",
                            "Agent should refuse to change the cabin because part of the reservation has already been flown.")
        return Decision(True, "eligible", "Agent should change the cabin for every segment of the reservation.")

    def free_baggage(self, db, reservation: dict) -> int:
        b = self.r["baggage"]["free_allowance"][self.membership(db, reservation)][reservation["cabin"]]
        return b * len(reservation["passengers"])

    def baggage_charge(self, db, reservation: dict, total_bags: int) -> Decision:
        """policy.md, "Change baggage and insurance"."""
        bag = self.r["baggage"]
        current = reservation.get("total_baggages", 0)
        if total_bags < current and not bag["allow_remove"]:
            return Decision(False, "remove_not_allowed",
                            "Agent should refuse to remove checked bags, because bags can be added but not removed.")
        free = self.free_baggage(db, reservation)
        nonfree = max(0, total_bags - free)
        charge = nonfree * bag["extra_bag_fee"]
        return Decision(True, "eligible",
                        f"Agent should add the checked bags and charge ${charge} for {nonfree} extra bags.",
                        {"free_allowance": free, "nonfree_baggages": nonfree, "charge": charge})

    def can_add_insurance(self) -> Decision:
        """policy.md: insurance cannot be added after booking."""
        if self.r["insurance"]["can_add_after_booking"]:
            return Decision(True, "eligible", "")
        return Decision(False, "insurance_after_booking",
                        "Agent should refuse to add travel insurance, because insurance cannot be added after booking.")

    def can_change_passenger_count(self) -> Decision:
        if self.r["passengers"]["can_change_count"]:
            return Decision(True, "eligible", "")
        return Decision(False, "passenger_count",
                        "Agent should refuse to change the number of passengers, because even a human agent cannot do that.")

    def compensation(self, db, reservation: dict, complaint: str, user_asked: bool = True,
                     changed_or_cancelled: bool = False) -> Decision:
        """policy.md, "Refunds and Compensation"."""
        comp = self.r["compensation"]
        if comp["requires_user_request"] and not user_asked:
            return Decision(False, "not_requested",
                            "Agent should not proactively offer compensation.")
        eligible = []
        if self.membership(db, reservation) in ("silver", "gold"):
            eligible.append("member_silver_gold")
        if reservation.get("insurance") == "yes":
            eligible.append("has_insurance")
        if reservation["cabin"] == "business":
            eligible.append("cabin_business")
        if not [e for e in eligible if e in comp["eligible_any_of"]]:
            return Decision(False, "not_eligible_for_compensation",
                            "Agent should refuse compensation, because the user is a regular member without insurance flying economy.")
        for rule in comp["rules"]:
            if rule["complaint"] != complaint:
                continue
            if rule["requires_flight_status"] not in self.segment_statuses(db, reservation):
                return Decision(False, "facts_not_confirmed",
                                f"Agent should not offer compensation, because no flight in the reservation is {rule['requires_flight_status']}.")
            if rule["requires_change_or_cancel"] and not changed_or_cancelled:
                return Decision(False, "requires_change_or_cancel",
                                "Agent should only offer compensation after changing or cancelling the reservation.")
            amount = rule["per_passenger"] * len(reservation["passengers"])
            return Decision(True, f"compensate_{complaint}",
                            f"Agent should send a ${amount} certificate for {len(reservation['passengers'])} passengers.",
                            {"amount": amount, "per_passenger": rule["per_passenger"]})
        return Decision(False, "no_matching_rule",
                        "Agent should refuse compensation, because the reason is not one the policy covers.")

    def can_book(self, db, user: dict, cabin: str, passengers: list, payment_ids: list,
                 segment_statuses: list[str]) -> Decision:
        """policy.md, "Book flight"."""
        b = self.r["book"]
        if len(passengers) > b["max_passengers"]:
            return Decision(False, "too_many_passengers",
                            f"Agent should refuse to book, because a reservation can have at most {b['max_passengers']} passengers.")
        profile = user.get("payment_methods", {})
        if b["payment_must_be_on_profile"] and any(p not in profile for p in payment_ids):
            return Decision(False, "payment_not_on_profile",
                            "Agent should refuse to book, because every payment method must already be in the user profile.")
        counts: dict[str, int] = {}
        for pid in payment_ids:
            src = profile.get(pid, {}).get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        for src, limit in b["max_payment_methods"].items():
            if counts.get(src, 0) > limit:
                return Decision(False, f"too_many_{src}",
                                f"Agent should refuse to book, because at most {limit} {src.replace('_', ' ')} may be used.")
        if any(s not in b["bookable_statuses"] for s in segment_statuses):
            return Decision(False, "flight_not_bookable",
                            "Agent should refuse to book, because a selected flight is not available.")
        return Decision(True, "eligible", "Agent should book the reservation as requested.")
