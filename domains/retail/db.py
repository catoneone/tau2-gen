"""Retail database deltas: a small product catalogue, users and orders for one generated task.

As with airline, tau2's `update_db` deep-merges dict-keyed sections, so each task carries only its own
entities. Every generated id is disjoint from tau2-bench's own data:

- product and item ids are 11 digits (tau2-bench uses 10)
- order ids look like `#G…` (tau2-bench uses `#W…`)
- user ids end in 5 digits (tau2-bench uses 4)
- payment ids carry 8-digit numbers (tau2-bench uses 7)
"""
from __future__ import annotations

import random
import re
from typing import Any, Optional

from common.db import ADDRESSES, FIRST, LAST

# Product types with their option axes. Variants are the cartesian product, sampled.
CATALOGUE: dict[str, dict[str, list]] = {
    "Desk Lamp": {"color": ["black", "white", "silver"], "brightness": ["low", "medium", "high"], "power source": ["AC adapter", "USB", "battery"]},
    "Running Shoes": {"size": ["8", "9", "10", "11"], "color": ["grey", "navy", "red"], "material": ["mesh", "knit"]},
    "Coffee Grinder": {"burr type": ["conical", "flat"], "color": ["stainless", "matte black"], "capacity": ["200g", "350g"]},
    "Backpack": {"volume": ["18L", "26L", "34L"], "color": ["olive", "charcoal", "tan"], "material": ["nylon", "canvas"]},
    "Bluetooth Speaker": {"color": ["black", "sand", "teal"], "battery life": ["8h", "16h", "24h"], "waterproof": ["yes", "no"]},
    "Yoga Mat": {"thickness": ["4mm", "6mm", "8mm"], "color": ["purple", "slate", "mint"], "material": ["TPE", "rubber"]},
    "Mechanical Watch": {"strap": ["leather", "steel", "nato"], "dial": ["white", "black", "blue"], "case size": ["38mm", "41mm"]},
    "Air Purifier": {"room size": ["small", "medium", "large"], "filter": ["HEPA", "carbon"], "color": ["white", "grey"]},
    "Electric Kettle": {"capacity": ["1.0L", "1.7L"], "material": ["glass", "steel"], "temperature control": ["yes", "no"]},
    "Monitor Stand": {"material": ["oak", "aluminium"], "height": ["fixed", "adjustable"], "color": ["natural", "black"]},
    "Wool Blanket": {"size": ["throw", "queen", "king"], "color": ["heather", "rust", "forest"], "weight": ["light", "heavy"]},
    "Cycling Helmet": {"size": ["S", "M", "L"], "color": ["white", "black", "lime"], "visor": ["yes", "no"]},
}
PAYMENT_SOURCES = ["credit_card", "paypal", "gift_card"]


def _digits(rng: random.Random, n: int) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(n))


class RetailDB:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.products: dict[str, dict] = {}
        self.users: dict[str, dict] = {}
        self.orders: dict[str, dict] = {}
        self._used: set[str] = set()

    def _uniq(self, make) -> str:
        for _ in range(10_000):
            v = make()
            if v not in self._used:
                self._used.add(v)
                return v
        raise RuntimeError("identifier space exhausted")

    # ---------- catalogue ----------
    def add_product(self, name: Optional[str] = None, n_variants: int = 4,
                    all_available: bool = True) -> str:
        rng = self.rng
        name = name or rng.choice(list(CATALOGUE))
        axes = CATALOGUE[name]
        pid = self._uniq(lambda: _digits(rng, 11))
        combos = set()
        variants: dict[str, dict] = {}
        for _ in range(n_variants * 4):
            if len(variants) >= n_variants:
                break
            opts = {k: rng.choice(v) for k, v in axes.items()}
            key = tuple(sorted(opts.items()))
            if key in combos:
                continue
            combos.add(key)
            iid = self._uniq(lambda: _digits(rng, 11))
            variants[iid] = {"item_id": iid, "options": opts,
                             "available": True if all_available else rng.random() < 0.75,
                             "price": round(rng.uniform(25, 420), 2)}
        # Guarantee at least two available variants so an exchange always has a target.
        avail = [v for v in variants.values() if v["available"]]
        for v in list(variants.values())[:2]:
            if len(avail) < 2:
                v["available"] = True
                avail = [x for x in variants.values() if x["available"]]
        self.products[pid] = {"name": name, "product_id": pid, "variants": variants}
        return pid

    def variants(self, product_id: str, available_only: bool = True) -> list[dict]:
        vs = list(self.products[product_id]["variants"].values())
        return [v for v in vs if v["available"]] if available_only else vs

    # ---------- users ----------
    def add_user(self, payment_sources: Optional[list[str]] = None, gift_card_balance: Optional[float] = None) -> str:
        rng = self.rng
        first, last = rng.choice(FIRST), rng.choice(LAST)
        uid = self._uniq(lambda: f"{first.lower()}_{re.sub(r'[^a-z]', '', last.lower())}_{_digits(rng, 5)}")
        street, city, state, zipc = rng.choice(ADDRESSES)
        methods: dict[str, dict] = {}
        for src in payment_sources or ["credit_card", "gift_card"]:
            pid = self._uniq(lambda: f"{src}_{_digits(rng, 8)}")
            if src == "credit_card":
                methods[pid] = {"source": "credit_card", "id": pid,
                                "brand": rng.choice(["visa", "mastercard", "amex"]), "last_four": _digits(rng, 4)}
            elif src == "paypal":
                methods[pid] = {"source": "paypal", "id": pid}
            else:
                methods[pid] = {"source": "gift_card", "id": pid,
                                "balance": gift_card_balance if gift_card_balance is not None else round(rng.uniform(20, 320), 2)}
        self.users[uid] = {
            "user_id": uid, "name": {"first_name": first, "last_name": last},
            "address": {"address1": street, "address2": f"Suite {rng.randint(100, 999)}",
                        "city": city, "country": "USA", "state": state, "zip": zipc},
            "email": f"{first.lower()}.{re.sub(r'[^a-z]', '', last.lower())}{rng.randint(100, 9999)}@example.net",
            "payment_methods": methods, "orders": [],
        }
        return uid

    # ---------- orders ----------
    def add_order(self, user_id: str, status: str, product_ids: Optional[list[str]] = None,
                  n_items: int = 2, payment_id: Optional[str] = None) -> str:
        rng = self.rng
        oid = self._uniq(lambda: "#G" + _digits(rng, 7))
        u = self.users[user_id]
        pids = product_ids or [self.add_product() for _ in range(n_items)]
        items, total = [], 0.0
        for pid in pids:
            v = rng.choice(self.variants(pid))
            items.append({"name": self.products[pid]["name"], "product_id": pid, "item_id": v["item_id"],
                          "price": v["price"], "options": dict(v["options"])})
            total += v["price"]
        pay = payment_id or next(iter(u["payment_methods"]))
        order: dict[str, Any] = {
            "order_id": oid, "user_id": user_id, "address": dict(u["address"]), "items": items,
            "status": status, "fulfillments": [],
            "payment_history": [{"transaction_type": "payment", "amount": round(total, 2), "payment_method_id": pay}],
        }
        if status in ("delivered", "processed"):
            order["fulfillments"] = [{"tracking_id": [_digits(rng, 12)], "item_ids": [i["item_id"] for i in items]}]
        if status == "cancelled":
            order["cancel_reason"] = rng.choice(["no longer needed", "ordered by mistake"])
            order["payment_history"].append({"transaction_type": "refund", "amount": round(total, 2), "payment_method_id": pay})
        u["orders"].append(oid)
        self.orders[oid] = order
        return oid

    def delta(self) -> dict:
        return {"products": self.products, "users": self.users, "orders": self.orders}
