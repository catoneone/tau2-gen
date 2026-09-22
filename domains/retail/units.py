"""Retail rules as composable units. Same construction as the airline units, and for the same reason:
one case per rule gives one decision per task and the set saturates.

Each unit acts on its own order unless a pair is declared to share one. Sharing is rarer here than in
airline, because the policy allows the items tool to be used only once per order and moves the order
out of `pending` when it is, so an order that has had its items modified cannot then be cancelled or
re-addressed. The pairs that do share an order are the ones the policy leaves open.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from domains.retail import db as rdb

CANCEL_REASONS = ["no longer needed", "ordered by mistake"]


class UnitError(Exception):
    """This unit cannot be built on the account it was handed."""


@dataclass
class UnitResult:
    oid: Optional[str]
    writes: list[dict] = field(default_factory=list)
    extra_reads: list[dict] = field(default_factory=list)
    request: str = ""
    anchor: str = ""
    nl: list[str] = field(default_factory=list)
    constraint: str = ""


def _card(d, uid: str) -> str:
    return next(p for p, m in d.users[uid]["payment_methods"].items() if m["source"] == "credit_card")


def _describe(item: dict) -> str:
    opts = ", ".join(f"{k} {v}" for k, v in list(item["options"].items())[:2])
    return f"{item['name'].lower()} ({opts})"


def u_cancel_ok(d, uid, rng, oid=None) -> UnitResult:
    oid = oid or d.add_order(uid, "pending")
    if d.orders[oid]["status"] != "pending":
        raise UnitError("cancel needs a pending order")
    reason = rng.choice(CANCEL_REASONS)
    return UnitResult(oid, [{"name": "cancel_pending_order", "arguments": {"order_id": oid, "reason": reason}}],
                      request=f"cancel order {oid}, it is {reason}", anchor="cancel",
                      nl=[f"Agent should cancel order {oid}."])


def u_cancel_denied(d, uid, rng, oid=None) -> UnitResult:
    oid = oid or d.add_order(uid, "delivered")
    return UnitResult(oid, [], request=f"cancel order {oid}", anchor="cancel",
                      nl=["Agent should refuse to cancel the order because it has already been delivered."])


def u_modify_address_ok(d, uid, rng, oid=None) -> UnitResult:
    oid = oid or d.add_order(uid, "pending")
    if d.orders[oid]["status"] != "pending":
        raise UnitError("address change needs a pending order")
    street, city, state, zipc = rng.choice(rdb.ADDRESSES)
    suite = f"Suite {rng.randint(100, 999)}"
    return UnitResult(oid, [{"name": "modify_pending_order_address",
                             "arguments": {"order_id": oid, "address1": street, "address2": suite,
                                           "city": city, "state": state, "country": "USA", "zip": zipc}}],
                      request=(f"have order {oid} delivered to the address {street}, {suite}, "
                               f"{city}, {state} {zipc}, USA instead"),
                      anchor="address", nl=["Agent should change the shipping address of the order."],
                      constraint="That address change is for the order only, not for your profile.")


def _order_total(d, oid: str) -> float:
    return round(sum(i["price"] for i in d.orders[oid]["items"]), 2)


def _payable(d, uid: str, oid: str, current: str) -> list[str]:
    """Methods that could actually pay for this order. policy.md: switching to a gift card requires the
    card to cover the whole amount."""
    total = _order_total(d, oid)
    out = []
    for pid, m in d.users[uid]["payment_methods"].items():
        if pid == current:
            continue
        if m["source"] == "gift_card" and m.get("balance", 0) < total:
            continue
        out.append(pid)
    return out


def u_modify_payment_ok(d, uid, rng, oid=None) -> UnitResult:
    oid = oid or d.add_order(uid, "pending")
    if d.orders[oid]["status"] != "pending":
        raise UnitError("payment change needs a pending order")
    current = d.orders[oid]["payment_history"][0]["payment_method_id"]
    options = _payable(d, uid, oid, current)
    if not options:
        raise UnitError("no alternative method can cover the order")
    target = rng.choice(options)
    src = d.users[uid]["payment_methods"][target]["source"].replace("_", " ")
    return UnitResult(oid, [{"name": "modify_pending_order_payment",
                             "arguments": {"order_id": oid, "payment_method_id": target}}],
                      request=f"change the payment method on order {oid} to your {src}",
                      anchor="payment", nl=["Agent should change the payment method of the order."])


def u_modify_payment_denied_balance(d, uid, rng, oid=None) -> UnitResult:
    """Switching to a gift card that cannot cover the order. policy.md requires the balance to cover
    the total, so the correct answer is to refuse."""
    if oid is not None:
        raise UnitError("builds its own order, since it needs a specific balance")
    oid = d.add_order(uid, "pending", n_items=3)
    total = _order_total(d, oid)
    pid = d._uniq(lambda: f"gift_card_{rdb._digits(rng, 8)}")
    d.users[uid]["payment_methods"][pid] = {"source": "gift_card", "id": pid,
                                            "balance": round(max(5.0, total * rng.uniform(0.2, 0.6)), 2)}
    return UnitResult(oid, [],
                      request=f"put order {oid} on your gift card as the payment method instead",
                      anchor="payment",
                      nl=["Agent should refuse, because the gift card balance does not cover the order total."])


def u_modify_items_ok(d, uid, rng, oid=None) -> UnitResult:
    if oid is not None:
        raise UnitError("the items tool may only be used once per order, so it does not share one")
    pid = d.add_product(n_variants=5)
    oid = d.add_order(uid, "pending", product_ids=[pid, d.add_product()])
    old = d.orders[oid]["items"][0]
    alts = [v for v in d.variants(pid) if v["item_id"] != old["item_id"]]
    if not alts:
        raise UnitError("no alternative variant")
    new = rng.choice(alts)
    want = ", ".join(f"{k} {v}" for k, v in new["options"].items())
    return UnitResult(oid, [{"name": "modify_pending_order_items",
                             "arguments": {"order_id": oid, "item_ids": [old["item_id"]],
                                           "new_item_ids": [new["item_id"]],
                                           "payment_method_id": _card(d, uid)}}],
                      extra_reads=[{"name": "get_product_details", "arguments": {"product_id": pid}}],
                      request=(f"swap one item on order {oid}, the {old['name'].lower()}, "
                               f"for the one with {want}"),
                      anchor="item", nl=["Agent should modify the item on the pending order."],
                      constraint="That is the only item on that order you want changed.")


def u_modify_items_cross_denied(d, uid, rng, oid=None) -> UnitResult:
    if oid is not None:
        raise UnitError("builds its own order")
    pid_a, pid_b = d.add_product("Desk Lamp"), d.add_product("Backpack")
    oid = d.add_order(uid, "pending", product_ids=[pid_a])
    return UnitResult(oid, [],
                      request=(f"swap one item on order {oid}, the {d.products[pid_a]['name'].lower()}, "
                               f"for a {d.products[pid_b]['name'].lower()} instead"),
                      anchor="item",
                      nl=["Agent should refuse, because an item can only be exchanged for a variant of "
                          "the same product."])


def u_return_ok(d, uid, rng, oid=None) -> UnitResult:
    oid = oid or d.add_order(uid, "delivered", n_items=rng.randint(2, 3))
    if d.orders[oid]["status"] != "delivered":
        raise UnitError("return needs a delivered order")
    item = d.orders[oid]["items"][0]
    pay = d.orders[oid]["payment_history"][0]["payment_method_id"]
    return UnitResult(oid, [{"name": "return_delivered_order_items",
                             "arguments": {"order_id": oid, "item_ids": [item["item_id"]],
                                           "payment_method_id": pay}}],
                      request=f"return the {item['name'].lower()} from order {oid}",
                      anchor="return", nl=["Agent should return the requested item."],
                      constraint="You want that refund on the original payment method.")


def u_return_pending_denied(d, uid, rng, oid=None) -> UnitResult:
    oid = oid or d.add_order(uid, "pending")
    return UnitResult(oid, [], request=f"return an item from order {oid}", anchor="return",
                      nl=["Agent should refuse, because only delivered orders can be returned."])


def u_exchange_ok(d, uid, rng, oid=None) -> UnitResult:
    if oid is not None:
        raise UnitError("the exchange tool may only be used once per order, so it does not share one")
    pid = d.add_product(n_variants=5)
    oid = d.add_order(uid, "delivered", product_ids=[pid, d.add_product()])
    old = d.orders[oid]["items"][0]
    alts = [v for v in d.variants(pid) if v["item_id"] != old["item_id"]]
    if not alts:
        raise UnitError("no alternative variant")
    new = rng.choice(alts)
    want = ", ".join(f"{k} {v}" for k, v in new["options"].items())
    return UnitResult(oid, [{"name": "exchange_delivered_order_items",
                             "arguments": {"order_id": oid, "item_ids": [old["item_id"]],
                                           "new_item_ids": [new["item_id"]],
                                           "payment_method_id": _card(d, uid)}}],
                      extra_reads=[{"name": "get_product_details", "arguments": {"product_id": pid}}],
                      request=f"exchange the {old['name'].lower()} from order {oid} for the one with {want}",
                      anchor="exchange", nl=["Agent should exchange the item for the requested variant."])


def u_profile_address_ok(d, uid, rng, oid=None) -> UnitResult:
    street, city, state, zipc = rng.choice(rdb.ADDRESSES)
    suite = f"Suite {rng.randint(100, 999)}"
    return UnitResult(None, [{"name": "modify_user_address",
                              "arguments": {"user_id": uid, "address1": street, "address2": suite,
                                            "city": city, "state": state, "country": "USA", "zip": zipc}}],
                      request=(f"update the address on your profile to {street}, {suite}, "
                               f"{city}, {state} {zipc}, USA"),
                      anchor="address", nl=["Agent should update the address on the user profile."],
                      constraint="That profile change must not touch any existing order.")


UNITS: dict[str, Callable] = {
    "cancel_ok": u_cancel_ok, "cancel_denied": u_cancel_denied,
    "modify_address_ok": u_modify_address_ok, "modify_payment_ok": u_modify_payment_ok, "modify_payment_denied_balance": u_modify_payment_denied_balance,
    "modify_items_ok": u_modify_items_ok, "modify_items_cross_denied": u_modify_items_cross_denied,
    "return_ok": u_return_ok, "return_pending_denied": u_return_pending_denied,
    "exchange_ok": u_exchange_ok, "profile_address_ok": u_profile_address_ok,
}

ACTION_UNITS = {"cancel_ok", "modify_address_ok", "modify_payment_ok", "modify_items_ok",
                "return_ok", "exchange_ok", "profile_address_ok"}

# Ordered pairs that act on one order. Only changes the policy leaves open: the items tool moves the
# order out of pending, so it never shares.
SHARED_RECORD: list[tuple[str, str]] = [
    ("modify_address_ok", "modify_payment_ok"),
]

INCOMPATIBLE: set[frozenset] = {
    frozenset({"cancel_ok", "cancel_denied"}),
    frozenset({"return_ok", "return_pending_denied"}),
    frozenset({"modify_items_ok", "modify_items_cross_denied"}),
    # Two address changes in one conversation are a wording trap, not a rule combination.
    frozenset({"modify_address_ok", "profile_address_ok"}),
    frozenset({"modify_payment_ok", "modify_payment_denied_balance"}),
}


def compatible(names: tuple[str, ...]) -> bool:
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if frozenset({a, b}) in INCOMPATIBLE:
                return False
    return True


def unit_ceiling() -> int:
    """How many distinct shapes the composable units can express, at the sizes the composer draws.

    An upper bound on distinct decisions from composition, before the rule outcomes inside each unit
    are counted, so the real ceiling is higher; what it bounds is how far more tasks can take you."""
    import itertools
    n = 0
    for k in (2, 3):
        for cand in itertools.combinations(sorted(UNITS), k):
            if compatible(tuple(sorted(cand))) and set(cand) & ACTION_UNITS:
                n += 1
    return n
