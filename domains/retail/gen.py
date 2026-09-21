"""Generator for tau2-shaped retail tasks.

Like airline, tau2-bench's retail tasks are hand-written, so tasks here are derived from the policy
document instead. Each case constructs an order in a state that puts it on one side of one rule:
cancel only from pending, return only from delivered, exchange only within the same product, items may
be modified once and only once, the number of passengers... (that one is airline) — see `CASES` below.

Each task carries its own catalogue, user and orders as a database delta in
`initial_state.initialization_data.agent_data`, deep-merged into the default database.

Scoring matches the benchmark: `reward_basis` is `[DB, NL_ASSERTION]`. Generated tasks leave
`nl_assertions` empty unless the case has a crisp sentence to assert, and an empty list scores 1.0
without an LLM judge, so the database check is what actually gates. That mirrors the benchmark, where
74 of 114 tasks carry no assertions either.

Authentication: every scenario gives the agent either an email or a name plus zip, never the user id,
because the policy requires the agent to locate the id itself.

Usage:
  upstream/tau2-bench/.venv/bin/python domains/retail/gen.py --n 150 --seed 0 --out out/retail
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
from domains.retail import db as rdb  # noqa: E402

DOMAIN = "retail"
READ_TOOLS = {"find_user_id_by_email", "find_user_id_by_name_zip", "get_user_details", "get_order_details",
              "get_product_details", "get_item_details", "list_all_product_types", "calculate"}
CANCEL_REASONS = ["no longer needed", "ordered by mistake"]


class BuildError(Exception):
    pass


class Case:
    def __init__(self, name: str, fn: Callable, weight: float, group: str, single_record: bool = True):
        self.name, self.fn, self.weight, self.group = name, fn, weight, group
        # Whether the unknown-id axis applies: it needs exactly one target record the scenario names.
        self.single_record = single_record


def _auth(d: rdb.RetailDB, uid: str, rng: random.Random) -> tuple[list[dict], str, str]:
    """The policy requires locating the user id from email or from name plus zip."""
    u = d.users[uid]
    if rng.random() < 0.5:
        return ([{"name": "find_user_id_by_email", "arguments": {"email": u["email"]}}],
                f"You are {u['name']['first_name']} {u['name']['last_name']}. Your email is {u['email']}.",
                "email")
    return ([{"name": "find_user_id_by_name_zip",
              "arguments": {"first_name": u["name"]["first_name"], "last_name": u["name"]["last_name"],
                            "zip": u["address"]["zip"]}}],
            f"You are {u['name']['first_name']} {u['name']['last_name']} in zip code {u['address']['zip']}. "
            f"You do not remember which email address you used.",
            "name_zip")


def _acts(raw: list[dict]) -> list[schema.Action]:
    return [schema.Action(action_id=f"{a['name']}_{i}", requestor="assistant", name=a["name"],
                          arguments=a["arguments"]) for i, a in enumerate(raw)]


def _profile_read(uid: str) -> list[dict]:
    """The benchmark looks the profile up after authenticating; it is where payment methods live."""
    return [{"name": "get_user_details", "arguments": {"user_id": uid}}]


def _product_reads(d, oid: str) -> list[dict]:
    seen, out = set(), []
    for it in d.orders[oid]["items"]:
        if it["product_id"] not in seen:
            seen.add(it["product_id"])
            out.append({"name": "get_product_details", "arguments": {"product_id": it["product_id"]}})
    return out


def _setup(rng: random.Random, status: str, n_items: int = 2, payment_sources=None):
    d = rdb.RetailDB(rng)
    uid = d.add_user(payment_sources or ["credit_card", "gift_card"])
    oid = d.add_order(uid, status, n_items=n_items)
    return d, uid, oid


def _describe(item: dict) -> str:
    opts = ", ".join(f"{k} {v}" for k, v in list(item["options"].items())[:2])
    return f"{item['name'].lower()} ({opts})"


# ---------------------------------------------------------------------------
# Account history and the unknown-id axis
# ---------------------------------------------------------------------------
# Two properties that cut across every case rather than belonging to any one of them: real customers
# have more than one order, and they often cannot quote the order number. When they cannot, the correct
# trajectory really does open several orders to find the right one, which is what lifts the number of
# records a task touches. The DB check ignores reads, so neither property changes how a task is scored.

def add_history(d, uid: str, rng: random.Random, n: tuple[int, int] = (1, 3)) -> list[str]:
    """Extra orders for this user, none of which is the task's target."""
    return [d.add_order(uid, rng.choice(["delivered", "cancelled", "processed", "pending"]))
            for _ in range(rng.randint(*n))]


def hide_id(scen: dict, d, uid: str, oid: str, reads: list[dict], rng: random.Random) -> tuple[dict, list[dict]]:
    """Replace the order id with a description of what is in it, and make the reference trajectory look
    through the profile. Returns everything unchanged if the id cannot be removed cleanly, so a task
    never claims the user does not know something the text still spells out."""
    items = d.orders[oid]["items"]
    reason = scen["reason_for_call"]
    if oid not in reason:
        return scen, reads
    # Anchor the order on something the sentence does not already name, so it does not read
    # "return the running shoes from the order with the running shoes in it".
    unnamed = [it for it in items if it["name"].lower() not in reason.lower()]
    if unnamed:
        desc = f"the order with the {_describe(unnamed[0])} in it"
    elif len(items) > 1:
        desc = "the order it came in"
    else:
        desc = "the order it came in"
    reason = reason.replace(f"order {oid}", desc).replace(oid, desc)
    if oid in reason:
        return scen, reads
    scen = dict(scen)
    scen["reason_for_call"] = reason + " You do not remember the order number."
    all_oids = list(d.users[uid]["orders"])
    rng.shuffle(all_oids)
    auth = [a for a in reads if a["name"] in ("find_user_id_by_email", "find_user_id_by_name_zip")]
    lookups = auth + [{"name": "get_user_details", "arguments": {"user_id": uid}}]
    lookups += [{"name": "get_order_details", "arguments": {"order_id": o}} for o in all_oids]
    tail = [a for a in reads if a["name"] not in
            ("find_user_id_by_email", "find_user_id_by_name_zip", "get_user_details", "get_order_details")]
    return scen, lookups + tail


# ---- cancel ----
def case_cancel_pending(rng):
    d, uid, oid = _setup(rng, "pending")
    reason = rng.choice(CANCEL_REASONS)
    reads, known, _ = _auth(d, uid, rng)
    reads += _profile_read(uid) + [{"name": "get_order_details", "arguments": {"order_id": oid}}]
    writes = [{"name": "cancel_pending_order", "arguments": {"order_id": oid, "reason": reason}}]
    scen = {"reason_for_call": f"You want to cancel order {oid} because it is {reason}.",
            "known_info": known,
            "task_instructions": user_sim.instructions(rng, n=1, core="")}
    return d, uid, oid, reads + writes, ["Agent should cancel the order."], scen


def case_cancel_delivered_denied(rng):
    d, uid, oid = _setup(rng, "delivered")
    reads, known, _ = _auth(d, uid, rng)
    reads += _profile_read(uid) + [{"name": "get_order_details", "arguments": {"order_id": oid}}]
    scen = {"reason_for_call": f"You want to cancel order {oid}, you changed your mind about it.",
            "known_info": known,
            "task_instructions": user_sim.instructions(rng, n=1, core="Do not ask to return anything in this conversation.", refusable=True)}
    return d, uid, oid, reads, ["Agent should refuse to cancel the order because it has already been delivered."], scen


# ---- modify pending ----
def case_modify_address(rng):
    d, uid, oid = _setup(rng, "pending")
    street, city, state, zipc = rng.choice(rdb.ADDRESSES)
    suite = f"Suite {rng.randint(100, 999)}"
    reads, known, _ = _auth(d, uid, rng)
    reads += _profile_read(uid) + [{"name": "get_order_details", "arguments": {"order_id": oid}}]
    writes = [{"name": "modify_pending_order_address",
               "arguments": {"order_id": oid, "address1": street, "address2": suite, "city": city,
                             "state": state, "country": "USA", "zip": zipc}}]
    scen = {"reason_for_call": f"You moved and want order {oid} delivered to {street}, {suite}, {city}, {state} {zipc}, USA instead.",
            "known_info": known,
            "task_instructions": user_sim.instructions(rng, n=1, core="Only the shipping address for that one order changes. Leave your profile address alone.")}
    return d, uid, oid, reads + writes, ["Agent should change the shipping address of the order."], scen


def case_modify_payment(rng):
    d, uid, oid = _setup(rng, "pending", payment_sources=["credit_card", "paypal"])
    order = d.orders[oid]
    current = order["payment_history"][0]["payment_method_id"]
    target = next(p for p in d.users[uid]["payment_methods"] if p != current)
    reads, known, _ = _auth(d, uid, rng)
    reads += _profile_read(uid) + [{"name": "get_order_details", "arguments": {"order_id": oid}}]
    writes = [{"name": "modify_pending_order_payment", "arguments": {"order_id": oid, "payment_method_id": target}}]
    src = d.users[uid]["payment_methods"][target]["source"].replace("_", " ")
    scen = {"reason_for_call": f"You want order {oid} charged to your {src} instead of the card you used.",
            "known_info": known,
            "task_instructions": user_sim.instructions(rng, n=1, core="")}
    return d, uid, oid, reads + writes, ["Agent should change the payment method of the order."], scen


def case_modify_items(rng):
    d = rdb.RetailDB(rng)
    uid = d.add_user(["credit_card", "gift_card"])
    pid = d.add_product(n_variants=5)
    oid = d.add_order(uid, "pending", product_ids=[pid, d.add_product()])
    order = d.orders[oid]
    old = order["items"][0]
    alts = [v for v in d.variants(pid) if v["item_id"] != old["item_id"]]
    if not alts:
        raise BuildError("no alternative variant")
    new = rng.choice(alts)
    pay = next(p for p, m in d.users[uid]["payment_methods"].items() if m["source"] == "credit_card")
    reads, known, _ = _auth(d, uid, rng)
    reads += _profile_read(uid) + [{"name": "get_order_details", "arguments": {"order_id": oid}}] + _product_reads(d, oid)
    writes = [{"name": "modify_pending_order_items",
               "arguments": {"order_id": oid, "item_ids": [old["item_id"]],
                             "new_item_ids": [new["item_id"]], "payment_method_id": pay}}]
    want = ", ".join(f"{k} {v}" for k, v in new["options"].items())
    scen = {"reason_for_call": f"On order {oid} you picked the wrong {old['name'].lower()}. You want the one with {want} instead.",
            "known_info": known,
            "task_instructions": user_sim.instructions(rng, n=1, core="That is the only change you want. If the agent asks whether anything else on the order should change, say no. Pay any difference with the credit card on file.")}
    return d, uid, oid, reads + writes, ["Agent should modify the item on the pending order."], scen


def case_modify_items_cross_product_denied(rng):
    d = rdb.RetailDB(rng)
    uid = d.add_user(["credit_card"])
    pid_a, pid_b = d.add_product("Desk Lamp"), d.add_product("Backpack")
    oid = d.add_order(uid, "pending", product_ids=[pid_a])
    reads, known, _ = _auth(d, uid, rng)
    reads += _profile_read(uid) + [{"name": "get_order_details", "arguments": {"order_id": oid}}] + _product_reads(d, oid)
    other = d.products[pid_b]["name"].lower()
    scen = {"reason_for_call": f"On order {oid} you want to swap the {d.products[pid_a]['name'].lower()} for a {other} instead.",
            "known_info": known,
            "task_instructions": user_sim.instructions(rng, n=1, core="Do not ask to cancel the order.", refusable=True)}
    return d, uid, oid, reads, ["Agent should refuse, because an item can only be exchanged for a variant of the same product."], scen


def case_modify_delivered_denied(rng):
    d, uid, oid = _setup(rng, "delivered")
    reads, known, _ = _auth(d, uid, rng)
    reads += _profile_read(uid) + [{"name": "get_order_details", "arguments": {"order_id": oid}}]
    scen = {"reason_for_call": f"You want to change the shipping address on order {oid}.",
            "known_info": known,
            "task_instructions": user_sim.instructions(rng, n=1, core="", refusable=True)}
    return d, uid, oid, reads, ["Agent should refuse, because only pending orders can be modified."], scen


# ---- delivered orders ----
def case_return_delivered(rng):
    d, uid, oid = _setup(rng, "delivered", n_items=rng.randint(2, 3))
    order = d.orders[oid]
    items = [order["items"][0]["item_id"]]
    pay = order["payment_history"][0]["payment_method_id"]
    reads, known, _ = _auth(d, uid, rng)
    reads += _profile_read(uid) + [{"name": "get_order_details", "arguments": {"order_id": oid}}]
    writes = [{"name": "return_delivered_order_items",
               "arguments": {"order_id": oid, "item_ids": items, "payment_method_id": pay}}]
    scen = {"reason_for_call": f"You want to return the {order['items'][0]['name'].lower()} from order {oid}.",
            "known_info": known,
            "task_instructions": user_sim.instructions(rng, n=1, core="You want the refund on the original payment method. Keep everything else on the order.")}
    return d, uid, oid, reads + writes, ["Agent should return the requested item."], scen


def case_return_pending_denied(rng):
    d, uid, oid = _setup(rng, "pending")
    reads, known, _ = _auth(d, uid, rng)
    reads += _profile_read(uid) + [{"name": "get_order_details", "arguments": {"order_id": oid}}]
    scen = {"reason_for_call": f"You want to return an item from order {oid}.",
            "known_info": known,
            "task_instructions": user_sim.instructions(rng, n=1, core="", refusable=True)}
    return d, uid, oid, reads, ["Agent should refuse, because only delivered orders can be returned."], scen


def case_exchange_delivered(rng):
    d = rdb.RetailDB(rng)
    uid = d.add_user(["credit_card", "gift_card"])
    pid = d.add_product(n_variants=5)
    oid = d.add_order(uid, "delivered", product_ids=[pid, d.add_product()])
    order = d.orders[oid]
    old = order["items"][0]
    alts = [v for v in d.variants(pid) if v["item_id"] != old["item_id"]]
    if not alts:
        raise BuildError("no alternative variant")
    new = rng.choice(alts)
    pay = next(p for p, m in d.users[uid]["payment_methods"].items() if m["source"] == "credit_card")
    reads, known, _ = _auth(d, uid, rng)
    reads += _profile_read(uid) + [{"name": "get_order_details", "arguments": {"order_id": oid}}] + _product_reads(d, oid)
    writes = [{"name": "exchange_delivered_order_items",
               "arguments": {"order_id": oid, "item_ids": [old["item_id"]],
                             "new_item_ids": [new["item_id"]], "payment_method_id": pay}}]
    want = ", ".join(f"{k} {v}" for k, v in new["options"].items())
    scen = {"reason_for_call": f"The {old['name'].lower()} from order {oid} is not what you wanted. You want to exchange it for the one with {want}.",
            "known_info": known,
            "task_instructions": user_sim.instructions(rng, n=1, core="That is the only item you want to exchange. Settle any difference on the credit card on file.")}
    return d, uid, oid, reads + writes, ["Agent should exchange the item for the requested variant."], scen


def case_exchange_unavailable_denied(rng):
    d = rdb.RetailDB(rng)
    uid = d.add_user(["credit_card"])
    pid = d.add_product(n_variants=5, all_available=False)
    unavailable = [v for v in d.variants(pid, available_only=False) if not v["available"]]
    if not unavailable:
        raise BuildError("no unavailable variant to ask for")
    target = rng.choice(unavailable)
    avail = d.variants(pid)
    oid = d.add_order(uid, "delivered", product_ids=[pid])
    d.orders[oid]["items"][0].update({"item_id": avail[0]["item_id"], "price": avail[0]["price"],
                                      "options": dict(avail[0]["options"])})
    reads, known, _ = _auth(d, uid, rng)
    reads += _profile_read(uid) + [{"name": "get_order_details", "arguments": {"order_id": oid}}] + _product_reads(d, oid)
    want = ", ".join(f"{k} {v}" for k, v in target["options"].items())
    scen = {"reason_for_call": f"You want to exchange the {d.products[pid]['name'].lower()} from order {oid} for the one with {want}.",
            "known_info": known,
            "task_instructions": user_sim.instructions(rng, n=1, core="If that version is out of stock, decide to keep what you have.", refusable=True)}
    return d, uid, oid, reads, ["Agent should tell the user the requested variant is not available and should not exchange it."], scen


def case_modify_user_address(rng):
    d, uid, oid = _setup(rng, "delivered")
    street, city, state, zipc = rng.choice(rdb.ADDRESSES)
    suite = f"Suite {rng.randint(100, 999)}"
    reads, known, _ = _auth(d, uid, rng)
    reads += _profile_read(uid)
    writes = [{"name": "modify_user_address",
               "arguments": {"user_id": uid, "address1": street, "address2": suite, "city": city,
                             "state": state, "country": "USA", "zip": zipc}}]
    scen = {"reason_for_call": f"You moved and want the address on your profile updated to {street}, {suite}, {city}, {state} {zipc}, USA.",
            "known_info": known,
            "task_instructions": user_sim.instructions(rng, n=1, core="Only your profile address changes. Do not change any existing order.")}
    return d, uid, None, reads + writes, ["Agent should update the address on the user profile."], scen


def case_other_user_denied(rng):
    d, uid, oid = _setup(rng, "pending")
    other = d.add_user(["credit_card"])
    other_order = d.add_order(other, "pending")
    reads, known, _ = _auth(d, uid, rng)
    reads += _profile_read(uid) + [{"name": "get_order_details", "arguments": {"order_id": oid}}]
    scen = {"reason_for_call": (f"You want to cancel order {oid}. You also want to cancel order {other_order}, "
                                f"which belongs to your flatmate, to save them a phone call."),
            "known_info": known,
            "task_instructions": user_sim.instructions(rng, n=1, core="Let the agent handle only your own order.", refusable=True)}
    writes = [{"name": "cancel_pending_order", "arguments": {"order_id": oid, "reason": rng.choice(CANCEL_REASONS)}}]
    return d, uid, oid, reads + writes, ["Agent should refuse to act on the other user's order and should only cancel the user's own order."], scen


def case_cancel_two_orders(rng):
    """Two pending orders cancelled in one conversation: two writes."""
    d = rdb.RetailDB(rng)
    uid = d.add_user(["credit_card", "gift_card"])
    oids = [d.add_order(uid, "pending") for _ in range(2)]
    reason = rng.choice(CANCEL_REASONS)
    reads, known, _ = _auth(d, uid, rng)
    reads += _profile_read(uid)
    writes, nl = [], []
    for oid in oids:
        reads.append({"name": "get_order_details", "arguments": {"order_id": oid}})
        writes.append({"name": "cancel_pending_order", "arguments": {"order_id": oid, "reason": reason}})
        nl.append(f"Agent should cancel order {oid}.")
    scen = {"reason_for_call": f"You want to cancel both of your open orders, {oids[0]} and {oids[1]}, they are {reason}.",
            "known_info": known,
            "task_instructions": user_sim.instructions(rng, n=1, core="Both orders should be cancelled.")}
    return d, uid, oids[0], reads + writes, nl, scen


def case_exchange_two_items(rng):
    """Two items exchanged in a single call, which the policy requires: the tool may only be used once
    per order, so everything must be collected first."""
    d = rdb.RetailDB(rng)
    uid = d.add_user(["credit_card", "gift_card"])
    pids = [d.add_product(n_variants=5), d.add_product(n_variants=5)]
    oid = d.add_order(uid, "delivered", product_ids=pids)
    order = d.orders[oid]
    olds, news = [], []
    for it in order["items"]:
        alts = [v for v in d.variants(it["product_id"]) if v["item_id"] != it["item_id"]]
        if not alts:
            raise BuildError("no alternative variant")
        olds.append(it["item_id"])
        news.append(rng.choice(alts)["item_id"])
    pay = next(p for p, m in d.users[uid]["payment_methods"].items() if m["source"] == "credit_card")
    reads, known, _ = _auth(d, uid, rng)
    reads += _profile_read(uid) + [{"name": "get_order_details", "arguments": {"order_id": oid}}] + _product_reads(d, oid)
    writes = [{"name": "exchange_delivered_order_items",
               "arguments": {"order_id": oid, "item_ids": olds, "new_item_ids": news, "payment_method_id": pay}}]
    names = " and ".join(sorted({i["name"].lower() for i in order["items"]}))
    scen = {"reason_for_call": f"You want to exchange both the {names} from order {oid} for different versions.",
            "known_info": known,
            "task_instructions": user_sim.instructions(rng, n=1, core="Mention both items. If the agent asks whether that is everything, say yes. Settle any difference on the credit card on file.")}
    return d, uid, oid, reads + writes, ["Agent should exchange both items in a single exchange call."], scen


# ---- composite cases: two rules, two orders, one conversation ----
def case_cancel_one_return_other(rng):
    """A pending order to cancel and a delivered order to return from, in the same conversation. The
    benchmark's retail tasks touch two entities at the median; single-rule tasks do not."""
    d = rdb.RetailDB(rng)
    uid = d.add_user(["credit_card", "gift_card"])
    pending = d.add_order(uid, "pending")
    delivered = d.add_order(uid, "delivered", n_items=rng.randint(2, 3))
    reason = rng.choice(CANCEL_REASONS)
    item = d.orders[delivered]["items"][0]
    pay = d.orders[delivered]["payment_history"][0]["payment_method_id"]
    reads, known, _ = _auth(d, uid, rng)
    reads += _profile_read(uid) + [
        {"name": "get_order_details", "arguments": {"order_id": pending}},
        {"name": "get_order_details", "arguments": {"order_id": delivered}}]
    writes = [
        {"name": "cancel_pending_order", "arguments": {"order_id": pending, "reason": reason}},
        {"name": "return_delivered_order_items",
         "arguments": {"order_id": delivered, "item_ids": [item["item_id"]], "payment_method_id": pay}},
    ]
    scen = {"reason_for_call": (f"Two things: cancel order {pending}, it is {reason}, and return the "
                                f"{item['name'].lower()} from order {delivered}."),
            "known_info": known,
            "task_instructions": user_sim.instructions(
                rng, "You want the refund for the return on the original payment method.")}
    return d, uid, pending, reads + writes, [f"Agent should cancel order {pending}.",
                                             f"Agent should return the item from order {delivered}."], scen


def case_address_then_payment(rng):
    """Two different modifications to one pending order. Neither is the once-per-order items tool, so
    both are allowed."""
    d = rdb.RetailDB(rng)
    uid = d.add_user(["credit_card", "paypal"])
    oid = d.add_order(uid, "pending")
    street, city, state, zipc = rng.choice(rdb.ADDRESSES)
    suite = f"Suite {rng.randint(100, 999)}"
    current = d.orders[oid]["payment_history"][0]["payment_method_id"]
    target = next(p for p in d.users[uid]["payment_methods"] if p != current)
    reads, known, _ = _auth(d, uid, rng)
    reads += _profile_read(uid) + [{"name": "get_order_details", "arguments": {"order_id": oid}}]
    writes = [
        {"name": "modify_pending_order_address",
         "arguments": {"order_id": oid, "address1": street, "address2": suite, "city": city,
                       "state": state, "country": "USA", "zip": zipc}},
        {"name": "modify_pending_order_payment", "arguments": {"order_id": oid, "payment_method_id": target}},
    ]
    src = d.users[uid]["payment_methods"][target]["source"].replace("_", " ")
    scen = {"reason_for_call": (f"On order {oid} you want the delivery address changed to {street}, {suite}, "
                                f"{city}, {state} {zipc}, USA, and you want it charged to your {src} instead."),
            "known_info": known,
            "task_instructions": user_sim.instructions(rng, n=1, core="Both changes are on that same order.")}
    return d, uid, oid, reads + writes, ["Agent should change the shipping address of the order.",
                                         "Agent should change the payment method of the order."], scen


def case_exchange_then_profile_address(rng):
    """An exchange on a delivered order plus a change to the profile address: two rules that touch
    different records, which the policy keeps separate."""
    d = rdb.RetailDB(rng)
    uid = d.add_user(["credit_card", "gift_card"])
    pid = d.add_product(n_variants=5)
    oid = d.add_order(uid, "delivered", product_ids=[pid])
    order = d.orders[oid]
    old = order["items"][0]
    alts = [v for v in d.variants(pid) if v["item_id"] != old["item_id"]]
    if not alts:
        raise BuildError("no alternative variant")
    new = rng.choice(alts)
    pay = next(p for p, m in d.users[uid]["payment_methods"].items() if m["source"] == "credit_card")
    street, city, state, zipc = rng.choice(rdb.ADDRESSES)
    suite = f"Suite {rng.randint(100, 999)}"
    reads, known, _ = _auth(d, uid, rng)
    reads += _profile_read(uid) + [{"name": "get_order_details", "arguments": {"order_id": oid}}] + _product_reads(d, oid)
    writes = [
        {"name": "exchange_delivered_order_items",
         "arguments": {"order_id": oid, "item_ids": [old["item_id"]],
                       "new_item_ids": [new["item_id"]], "payment_method_id": pay}},
        {"name": "modify_user_address",
         "arguments": {"user_id": uid, "address1": street, "address2": suite, "city": city,
                       "state": state, "country": "USA", "zip": zipc}},
    ]
    want = ", ".join(f"{k} {v}" for k, v in new["options"].items())
    scen = {"reason_for_call": (f"You want to exchange the {old['name'].lower()} from order {oid} for the one with "
                                f"{want}, and you have moved, so your profile address should become {street}, {suite}, "
                                f"{city}, {state} {zipc}, USA."),
            "known_info": known,
            "task_instructions": user_sim.instructions(
                rng, "The address change is for your profile, not for any existing order. Settle any difference on the credit card on file.")}
    return d, uid, oid, reads + writes, ["Agent should exchange the item.",
                                         "Agent should update the address on the user profile."], scen


def case_return_unknown_order(rng):
    """The user does not remember which order the item was in. The agent has to look through the
    profile and open several orders to find it. The benchmark does this routinely; it is also where
    `unknown_info` earns its place."""
    d = rdb.RetailDB(rng)
    uid = d.add_user(["credit_card", "gift_card"])
    others = [d.add_order(uid, rng.choice(["delivered", "pending", "processed"])) for _ in range(2)]
    target = d.add_order(uid, "delivered", n_items=rng.randint(2, 3))
    item = d.orders[target]["items"][0]
    pay = d.orders[target]["payment_history"][0]["payment_method_id"]
    reads, known, _ = _auth(d, uid, rng)
    reads += _profile_read(uid)
    order_ids = others + [target]
    rng.shuffle(order_ids)
    reads += [{"name": "get_order_details", "arguments": {"order_id": o}} for o in order_ids]
    writes = [{"name": "return_delivered_order_items",
               "arguments": {"order_id": target, "item_ids": [item["item_id"]], "payment_method_id": pay}}]
    scen = {"reason_for_call": f"You want to return the {_describe(item)} you ordered, but you do not remember which order it was in.",
            "known_info": known,
            "task_instructions": user_sim.instructions(
                rng, "You can describe the item but not the order number. You want the refund on the original payment method.")}
    return d, uid, target, reads + writes, [f"Agent should find the order and return the item."], scen


def case_cancel_unknown_order(rng):
    """Two open orders, and the user identifies the one to cancel by what is in it."""
    d = rdb.RetailDB(rng)
    uid = d.add_user(["credit_card", "gift_card"])
    a = d.add_order(uid, "pending")
    b = d.add_order(uid, "pending")
    target = rng.choice([a, b])
    item = d.orders[target]["items"][0]
    reason = rng.choice(CANCEL_REASONS)
    reads, known, _ = _auth(d, uid, rng)
    reads += _profile_read(uid) + [{"name": "get_order_details", "arguments": {"order_id": o}} for o in (a, b)]
    writes = [{"name": "cancel_pending_order", "arguments": {"order_id": target, "reason": reason}}]
    scen = {"reason_for_call": (f"You have two open orders and want to cancel the one with the "
                                f"{_describe(item)} in it, it is {reason}. You do not remember the order number."),
            "known_info": known,
            "task_instructions": user_sim.instructions(
                rng, "Only that one order should be cancelled, the other one you still want.")}
    return d, uid, target, reads + writes, [f"Agent should cancel only order {target}."], scen


CASES = [
    Case("cancel_two_orders", case_cancel_two_orders, 12, "cancel", single_record=False),
    Case("cancel_one_return_other", case_cancel_one_return_other, 15, "composite", single_record=False),
    Case("return_unknown_order", case_return_unknown_order, 10, "composite", single_record=False),
    Case("cancel_unknown_order", case_cancel_unknown_order, 8, "composite", single_record=False),
    Case("address_then_payment", case_address_then_payment, 6, "composite"),
    Case("exchange_then_profile_address", case_exchange_then_profile_address, 6, "composite", single_record=False),
    Case("exchange_two_items", case_exchange_two_items, 6, "exchange"),
    Case("cancel_pending", case_cancel_pending, 9, "cancel"),
    Case("cancel_delivered_denied", case_cancel_delivered_denied, 6, "refuse"),
    Case("modify_address", case_modify_address, 7, "modify"),
    Case("modify_payment", case_modify_payment, 6, "modify"),
    Case("modify_items", case_modify_items, 9, "modify"),
    Case("modify_items_cross_product_denied", case_modify_items_cross_product_denied, 6, "refuse"),
    Case("modify_delivered_denied", case_modify_delivered_denied, 5, "refuse"),
    Case("return_delivered", case_return_delivered, 9, "return"),
    Case("return_pending_denied", case_return_pending_denied, 5, "refuse"),
    Case("exchange_delivered", case_exchange_delivered, 9, "exchange"),
    Case("exchange_unavailable_denied", case_exchange_unavailable_denied, 5, "refuse"),
    Case("modify_user_address", case_modify_user_address, 5, "profile", single_record=False),
    Case("other_user_denied", case_other_user_denied, 4, "refuse", single_record=False),
]


def build_task(idx: int, tag: str, case: Case, persona_name: str, rng: random.Random,
               history: bool = True, unknown_id: bool = False) -> tuple[dict, dict]:
    d, uid, oid, raw_actions, nl, scen = case.fn(rng)
    hid = False
    if history and uid in d.users:
        add_history(d, uid, rng)
    if unknown_id and case.single_record and oid:
        scen, raw_actions = hide_id(scen, d, uid, oid, raw_actions, rng)
        hid = oid not in scen["reason_for_call"]
    persona_text = user_sim.render_persona(persona_name, rng, "555-555-0100") if persona_name != "None" else None
    name = f"[{case.name}][PERSONA:{persona_name}][GEN:{tag}]"
    writes = [a for a in raw_actions if a["name"] not in READ_TOOLS]
    data = schema.Tau2TaskData(
        idx=idx, name=name, description=f"Purpose: {case.name.replace('_', ' ')}", id=name,
        user_scenario=schema.UserScenario(
            persona=persona_text,
            instructions=schema.UserInstructions(domain=DOMAIN, reason_for_call=scen["reason_for_call"],
                                                 known_info=scen["known_info"], unknown_info=None,
                                                 task_instructions=scen["task_instructions"])),
        ticket=None,
        initial_state=schema.InitialState(
            initialization_data=schema.InitializationData(agent_data=d.delta(), user_data=None),
            initialization_actions=None, message_history=None),
        evaluation_criteria=schema.EvaluationCriteria(
            actions=_acts(raw_actions), env_assertions=[], communicate_info=None,
            nl_assertions=nl, reward_basis=["DB", "NL_ASSERTION"]),
        domain=DOMAIN,
        tau_description=schema.TauDescription(purpose=case.name.replace("_", " "), relevant_policies=None, notes=None),
    ).to_dict()
    data["evaluation_criteria"]["env_assertions"] = None
    meta = {"idx": idx, "id": name, "case": case.name, "group": case.group, "persona": persona_name,
            "unknown_id": hid, "n_user_records": len(d.users.get(uid, {}).get("orders", [])),
            "n_writes": len(writes), "write_names": sorted({a["name"] for a in writes}),
            "user_id": uid, "order_id": oid, "delta_bytes": len(json.dumps(d.delta()))}
    return data, meta


def verify_task(data: dict) -> Optional[str]:
    from tau2.domains.retail.environment import get_environment

    env = get_environment()
    env.set_state(
        initialization_data=InitializationData.model_validate(data["initial_state"]["initialization_data"]),
        initialization_actions=None, message_history=[])
    before = env.get_db_hash()
    actions = data["evaluation_criteria"]["actions"]
    writes = [a for a in actions if a["name"] not in READ_TOOLS]
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
    """Weights, optionally renormalised so the `refuse` group takes a target share of the set."""
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
             unknown_id_share: float = 0.45, verbose: bool = False):
    rng = random.Random(seed)
    weights = rebalance(CASES, refuse_share)
    left = quotas(n, weights)
    records, metas, stats = [], [], Counter()
    idx, attempts = 0, 0
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
    ap.add_argument("--out", default=str(GEN_ROOT / "out" / "retail"))
    ap.add_argument("--persona-mix", default=None)
    ap.add_argument("--refuse-share", type=float, default=0.09,
                    help="share of tasks whose correct answer is to do nothing (tau2-bench retail: 0.09)")
    ap.add_argument("--unknown-id-share", type=float, default=0.45,
                    help="share of single-record tasks where the user cannot quote the order number")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    mix = json.loads(a.persona_mix) if a.persona_mix else {"None": 0.5, "Easy": 0.08, "Hard": 0.08, "Verbose": 0.06,
                                                           "Terse": 0.06, "NonNative": 0.06, "Impatient": 0.06,
                                                           "WrongNumberOnce": 0.04, "SideRequest": 0.03, "TechSavvy": 0.03}
    t0 = time.time()
    records, metas, stats = generate(a.n, a.seed, mix, a.refuse_share, a.unknown_id_share, a.verbose)
    print(f"generated {len(records)} retail tasks in {time.time() - t0:.1f}s; stats={stats}")

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
    (out / "taskset.toml").write_text('[env.taskset]\ndomain = "retail"\ntasks = "tasks_tau2.json"\n')
    manifest = {
        "generator": "domains/retail/gen.py", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
