"""Airline database deltas: new users, flights and reservations for one generated task.

Each task carries only what it needs. tau2's `update_db` deep-merges dict-keyed sections, so a delta
of a handful of entities lands in the default database without disturbing the rest of it, and the
task stays small. Every generated id is disjoint from tau2-bench's own data:

- flight numbers use the carrier prefix `NVA` (tau2-bench uses `HAT`)
- reservation ids are 6 characters starting with `Z` (tau2-bench's never do)
- user ids are `<first>_<last>_<5 digits>` with 5 digits (tau2-bench uses 4)
- payment ids carry 8-digit numbers (tau2-bench uses 7)
"""
from __future__ import annotations

import datetime as dt
import random
import re
from typing import Any, Optional

from common.db import ADDRESSES, FIRST, LAST

NOW = dt.datetime(2024, 5, 15, 15, 0, 0)
AIRPORTS = ["ATL", "BOS", "CLT", "DEN", "DFW", "DTW", "EWR", "IAH", "JFK", "LAS", "LAX", "LGA",
            "MCO", "MIA", "MSP", "ORD", "PHL", "PHX", "SEA", "SFO"]
CABINS = ["basic_economy", "economy", "business"]
MEMBERSHIPS = ["regular", "silver", "gold"]
CARRIER = "NVA"
BASE_PRICE = {"basic_economy": (55, 110), "economy": (110, 260), "business": (280, 950)}


def _digits(rng: random.Random, n: int) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(n))


class AirlineDB:
    """Builds one task's delta and hands out the ids the task should reference."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.flights: dict[str, dict] = {}
        self.users: dict[str, dict] = {}
        self.reservations: dict[str, dict] = {}
        self._used: set[str] = set()

    # ---------- ids ----------
    def _uniq(self, make) -> str:
        for _ in range(10_000):
            v = make()
            if v not in self._used:
                self._used.add(v)
                return v
        raise RuntimeError("identifier space exhausted")

    def flight_number(self) -> str:
        return self._uniq(lambda: f"{CARRIER}{self.rng.randint(100, 999)}")

    def reservation_id(self) -> str:
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return self._uniq(lambda: "Z" + "".join(self.rng.choice(alphabet) for _ in range(5)))

    def payment_id(self, source: str) -> str:
        return self._uniq(lambda: f"{source}_{_digits(self.rng, 8)}")

    # ---------- entities ----------
    def add_flight(self, origin: str, destination: str, date: str, status: str,
                   min_seats: int = 1) -> tuple[str, dict]:
        """One flight with one dated entry in the given status. Returns (flight_number, date_entry).

        `min_seats` is the floor on free seats in every cabin. A reservation's own flights are built
        with min_seats = passenger count, so that upgrading the cabin or moving the whole party to
        another flight is actually possible; without it a three-passenger booking can land on a
        flight with one business seat and the expected action fails to execute."""
        fn = self.flight_number()
        dep_h = self.rng.randint(5, 20)
        dur = self.rng.randint(1, 6)
        dep = f"{dep_h:02d}:{self.rng.choice(['00', '15', '30', '45'])}:00"
        arr = f"{(dep_h + dur) % 24:02d}:{self.rng.choice(['00', '15', '30', '45'])}:00"
        prices = {c: float(self.rng.randint(*BASE_PRICE[c])) for c in CABINS}
        entry: dict[str, Any] = {"status": status}
        if status == "available":
            entry["available_seats"] = {c: max(min_seats, self.rng.randint(1, 18)) for c in CABINS}
            entry["prices"] = prices
        elif status in ("on time", "delayed"):
            entry["estimated_departure_time_est"] = f"{date}T{dep}"
            entry["estimated_arrival_time_est"] = f"{date}T{arr}"
        elif status == "flying":
            entry["actual_departure_time_est"] = f"{date}T{dep}"
            entry["estimated_arrival_time_est"] = f"{date}T{arr}"
        elif status == "landed":
            entry["actual_departure_time_est"] = f"{date}T{dep}"
            entry["actual_arrival_time_est"] = f"{date}T{arr}"
        self.flights[fn] = {
            "flight_number": fn, "origin": origin, "destination": destination,
            "scheduled_departure_time_est": dep, "scheduled_arrival_time_est": arr,
            "dates": {date: entry},
        }
        return fn, entry

    def add_alternative_flights(self, origin: str, destination: str, date: str, n: int = 2,
                                min_seats: int = 1) -> list[str]:
        """Extra available flights on the same city pair, so a change request has somewhere to go."""
        return [self.add_flight(origin, destination, date, "available", min_seats)[0] for _ in range(n)]

    def add_user(self, membership: Optional[str] = None, payment_sources: Optional[list[str]] = None) -> str:
        rng = self.rng
        first, last = rng.choice(FIRST), rng.choice(LAST)
        uid = self._uniq(lambda: f"{first.lower()}_{re.sub(r'[^a-z]', '', last.lower())}_{_digits(rng, 5)}")
        street, city, state, zipc = rng.choice(ADDRESSES)
        methods: dict[str, dict] = {}
        for src in payment_sources or ["credit_card", "gift_card"]:
            pid = self.payment_id(src)
            if src == "credit_card":
                methods[pid] = {"source": "credit_card", "id": pid,
                                "brand": rng.choice(["visa", "mastercard", "amex"]),
                                "last_four": _digits(rng, 4)}
            elif src == "gift_card":
                methods[pid] = {"source": "gift_card", "id": pid, "amount": float(rng.choice([50, 100, 150, 200, 250]))}
            else:
                methods[pid] = {"source": "certificate", "id": pid, "amount": float(rng.choice([50, 100, 150, 250]))}
        self.users[uid] = {
            "user_id": uid,
            "name": {"first_name": first, "last_name": last},
            "address": {"address1": street, "address2": f"Suite {rng.randint(100, 999)}",
                        "city": city, "country": "USA", "state": state, "zip": zipc},
            "email": f"{first.lower()}.{re.sub(r'[^a-z]', '', last.lower())}{rng.randint(100, 9999)}@example.net",
            "dob": f"{rng.randint(1950, 2002)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            "payment_methods": methods,
            "saved_passengers": [],
            "membership": membership or rng.choice(MEMBERSHIPS),
            "reservations": [],
        }
        return uid

    def passengers(self, user_id: str, n: int) -> list[dict]:
        rng = self.rng
        u = self.users[user_id]
        out = [{"first_name": u["name"]["first_name"], "last_name": u["name"]["last_name"], "dob": u["dob"]}]
        while len(out) < n:
            out.append({"first_name": rng.choice(FIRST), "last_name": rng.choice(LAST),
                        "dob": f"{rng.randint(1950, 2005)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"})
        return out[:n]

    def add_reservation(self, user_id: str, cabin: str, insurance: str, created_at: dt.datetime,
                        segments: list[tuple[str, str, str, str]], n_passengers: int = 1,
                        trip: Optional[str] = None, total_baggages: int = 0,
                        nonfree_baggages: int = 0) -> str:
        """segments: list of (origin, destination, date, status)."""
        rng = self.rng
        rid = self.reservation_id()
        flights, total = [], 0
        for origin, destination, date, status in segments:
            fn, _ = self.add_flight(origin, destination, date, status, min_seats=n_passengers)
            price = float(rng.randint(*BASE_PRICE[cabin]))
            total += price * n_passengers
            flights.append({"origin": origin, "destination": destination, "flight_number": fn,
                            "date": date, "price": price})
        pay_id = next(iter(self.users[user_id]["payment_methods"]))
        self.reservations[rid] = {
            "reservation_id": rid, "user_id": user_id,
            "origin": flights[0]["origin"], "destination": flights[0]["destination"],
            "flight_type": trip or ("round_trip" if len(flights) > 1 and flights[-1]["destination"] == flights[0]["origin"] else "one_way"),
            "cabin": cabin, "flights": flights,
            "passengers": self.passengers(user_id, n_passengers),
            "payment_history": [{"payment_id": pay_id, "amount": int(total)}],
            "created_at": created_at.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_baggages": total_baggages, "nonfree_baggages": nonfree_baggages,
            "insurance": insurance,
        }
        self.users[user_id]["reservations"].append(rid)
        return rid

    def delta(self) -> dict:
        return {"flights": self.flights, "users": self.users, "reservations": self.reservations}


# ---------- date helpers ----------
def future_date(rng: random.Random, lo: int = 2, hi: int = 14) -> str:
    return (NOW + dt.timedelta(days=rng.randint(lo, hi))).strftime("%Y-%m-%d")


def past_date(rng: random.Random, lo: int = 2, hi: int = 12) -> str:
    return (NOW - dt.timedelta(days=rng.randint(lo, hi))).strftime("%Y-%m-%d")


def created_within_24h(rng: random.Random) -> dt.datetime:
    return NOW - dt.timedelta(hours=rng.randint(1, 23), minutes=rng.randint(0, 59))


def created_long_ago(rng: random.Random) -> dt.datetime:
    return NOW - dt.timedelta(days=rng.randint(3, 14), hours=rng.randint(0, 23))


def city_pair(rng: random.Random) -> tuple[str, str]:
    a, b = rng.sample(AIRPORTS, 2)
    return a, b
