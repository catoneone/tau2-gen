"""Telecom DB instances: fresh names, phone numbers, emails, ids, IMEIs, plans and addresses.

Every generated task gets entities that do not appear in tau2-bench's own `db.toml` or in its task set,
so a model cannot solve a task by recalling the single customer baked into the benchmark.

Conventions:
- Phone numbers use the North American fiction range `<area>-555-01xx` (tau2-bench uses `555-123-xxxx`).
- Ids are 5 digits (`C20000..C99999`); tau2-bench uses `C1001` / `L1002` / `B1001` / `D1001` / `P1001`.
- Names avoid tau2-bench's four customers; email domains are example.net / example.org (upstream: example.com).
- Timeline keeps tau2-bench's "now = 2025-02-25".
"""
from __future__ import annotations

import random
import re
from typing import Any

FIRST = [
    "Aiden", "Amara", "Anika", "Arjun", "Beatriz", "Calvin", "Camila", "Cedric", "Dalia", "Dmitri",
    "Elena", "Elliot", "Farah", "Felix", "Gabriela", "Gideon", "Hana", "Hugo", "Imani", "Isaac",
    "Jasmine", "Jonas", "Kaito", "Keziah", "Leila", "Lorenzo", "Maeve", "Mateo", "Nadia", "Niko",
    "Olamide", "Oscar", "Priya", "Quentin", "Rafael", "Rosalind", "Santiago", "Selin", "Tomas", "Tessa",
    "Uma", "Ulysses", "Valentina", "Viktor", "Wren", "Wesley", "Ximena", "Yara", "Yusuf", "Zoe",
    "Zane", "Ingrid", "Bram", "Noor", "Kwame", "Lucia", "Ravi", "Sofia", "Emil", "Aurora",
]
LAST = [
    "Abara", "Baptiste", "Castellano", "Delacroix", "Eriksen", "Farouk", "Gallagher", "Haddad", "Ibarra",
    "Jansen", "Kowalczyk", "Lindqvist", "Marchetti", "Nakamura", "Okonkwo", "Petrova", "Quiroga", "Rasmussen",
    "Saldana", "Takahashi", "Uzoma", "Vasquez", "Whitfield", "Xu", "Yilmaz", "Zielinski", "Anand", "Brennan",
    "Chowdhury", "Dubois", "Espinoza", "Fitzgerald", "Gruber", "Hollis", "Iqbal", "Jokinen", "Kaur", "Lombardi",
    "Mensah", "Novak", "Oduya", "Pereira", "Radcliffe", "Sørensen", "Thornton", "Ueda", "Villanueva", "Weber",
    "Yamamoto", "Zamora", "Achterberg", "Bianchi", "Cormier", "Dalgaard", "Ferreira", "Goncalves", "Halloran",
    "Karlsson", "Moreau", "Osei",
]
BENCH_NAMES = {"John Smith", "Sarah Johnson", "Michael Lee", "Emma Wilson"}
EMAIL_DOMAINS = ["example.net", "example.org"]
# (name, data_limit_gb, price_per_month, refuel_price_per_gb) —— 与 benchmark 的 5 个套餐名 / 价格都不同
PLAN_TEMPLATES = [
    ("Starter 4GB", 4.0, 30.0, 6.0),
    ("Everyday 12GB", 12.0, 55.0, 3.0),
    ("Max Unlimited", 999.0, 90.0, 0.5),
    ("Household Share 30GB", 30.0, 130.0, 2.5),
    ("Sensor Lite 1GB", 1.0, 12.0, 8.0),
    ("Traveler 20GB", 20.0, 70.0, 2.0),
]
DEVICE_MODELS = [
    ("phone", "Nova 12"), ("phone", "Aster Pro"), ("phone", "Orbit S4"), ("phone", "Lumen Fold"),
    ("phone", "Halo 9"), ("phone", "Vega Mini"), ("tablet", "Slate 10"), ("tablet", "Canvas Air"),
    ("watch", "Pulse Band"), ("phone", "Quasar X2"),
]
ADDRESSES = [
    ("742 Juniper Way", "Boulder", "CO", "80302"), ("18 Harbor View Rd", "Portland", "ME", "04101"),
    ("2201 Alder Street", "Tacoma", "WA", "98402"), ("55 Birch Hollow", "Ann Arbor", "MI", "48104"),
    ("910 Sycamore Ave", "Tucson", "AZ", "85701"), ("3 Willow Court", "Burlington", "VT", "05401"),
    ("1460 Magnolia Blvd", "Savannah", "GA", "31401"), ("77 Larkspur Lane", "Madison", "WI", "53703"),
    ("608 Cypress Dr", "Santa Fe", "NM", "87501"), ("129 Beacon St", "Providence", "RI", "02903"),
    ("415 Meadowlark Rd", "Boise", "ID", "83702"), ("36 Quarry Hill", "Asheville", "NC", "28801"),
]
AREA_CODES = [212, 310, 415, 503, 602, 617, 646, 720, 737, 808, 857, 929, 972, 206, 303, 404, 512, 702]


def _pick_unique(rng: random.Random, used: set, make) -> str:
    for _ in range(10_000):
        v = make()
        if v not in used:
            used.add(v)
            return v
    raise RuntimeError("identifier space exhausted")


def _id(rng: random.Random, prefix: str, used: set) -> str:
    return _pick_unique(rng, used, lambda: f"{prefix}{rng.randint(20000, 99999)}")


def _phone(rng: random.Random, used: set) -> str:
    return _pick_unique(rng, used, lambda: f"{rng.choice(AREA_CODES)}-555-01{rng.randint(0, 99):02d}")


def _name(rng: random.Random, used: set) -> str:
    return _pick_unique(
        rng, used | BENCH_NAMES, lambda: f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    )


def _email(name: str, rng: random.Random) -> str:
    f, l = name.lower().split(" ", 1)
    l = re.sub(r"[^a-z]", "", l)
    return f"{f}.{l}{rng.randint(1, 99)}@{rng.choice(EMAIL_DOMAINS)}"


def _dob(rng: random.Random) -> str:
    return f"{rng.randint(1950, 2002)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def _imei(rng: random.Random, used: set) -> str:
    return _pick_unique(rng, used, lambda: "35" + "".join(str(rng.randint(0, 9)) for _ in range(13)))


class DBBuilder:
    """逐个客户地搭一份 TelecomDB。`add_customer(is_target=True)` 返回可作任务目标的线路描述。

    - per-task 模式：每题一个 DBBuilder（1 个目标客户 + 2–4 个填充客户），整份 DB 装进 initialization_data.agent_data。
    - shared 模式：整个生成集共用一个 DBBuilder（每题一个目标客户 + 若干填充），输出 db.toml，initialization_data 为 null。
    """

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.used: dict[str, set] = {k: set() for k in ("id", "phone", "name", "imei")}
        self.plans = [
            {
                "plan_id": _id(rng, "P", self.used["id"]),
                "name": name,
                "data_limit_gb": limit,
                "price_per_month": price,
                "data_refueling_price_per_gb": refuel,
            }
            for name, limit, price, refuel in PLAN_TEMPLATES
        ]
        self.limited_plans = [p for p in self.plans if 4 <= p["data_limit_gb"] < 100]
        self.db: dict[str, list] = {"plans": self.plans, "customers": [], "lines": [], "bills": [], "devices": []}

    def add_customer(self, is_target: bool) -> dict | None:
        rng, used = self.rng, self.used
        name = _name(rng, used["name"])
        cust_id = _id(rng, "C", used["id"])
        n_lines = rng.randint(1, 3)
        target_line_idx = rng.randrange(n_lines) if is_target else -1
        line_ids, line_prices, phones = [], [], []
        target: dict[str, Any] | None = None
        for li in range(n_lines):
            is_t = li == target_line_idx
            dtype, model = rng.choice([d for d in DEVICE_MODELS if d[0] == "phone"]) if is_t else rng.choice(DEVICE_MODELS)
            dev_id = _id(rng, "D", used["id"])
            self.db["devices"].append(
                {
                    "device_id": dev_id, "device_type": dtype, "model": model, "imei": _imei(rng, used["imei"]),
                    "is_esim_capable": rng.random() < 0.7, "activated": True,
                    "activation_date": f"2024-{rng.randint(6, 12):02d}-{rng.randint(1, 28):02d}T{rng.randint(8, 18):02d}:{rng.randint(0, 59):02d}:00",
                    "last_esim_transfer_date": None,
                }
            )
            plan = rng.choice(self.limited_plans) if is_t else rng.choice(self.plans)
            status = "Active" if (is_t or rng.random() >= 0.15) else "Suspended"
            limit = plan["data_limit_gb"]
            used_gb = round(rng.uniform(0.2, min(limit, 40.0) * 0.85), 1) if status == "Active" else 0.0
            line = {
                "line_id": _id(rng, "L", used["id"]), "phone_number": _phone(rng, used["phone"]), "status": status,
                "plan_id": plan["plan_id"], "device_id": dev_id, "data_used_gb": used_gb, "data_refueling_gb": 0.0,
                "roaming_enabled": rng.random() < 0.5,
                "contract_end_date": f"{rng.randint(2026, 2027)}-{rng.randint(1, 12):02d}-28",
                "last_plan_change_date": f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                "last_sim_replacement_date": None,
                "suspension_start_date": "2025-02-01" if status == "Suspended" else None,
            }
            self.db["lines"].append(line)
            line_ids.append(line["line_id"]); phones.append(line["phone_number"])
            line_prices.append((plan["name"], line["phone_number"], plan["price_per_month"]))
            if is_t:
                target = {"name": name, "phone": line["phone_number"], "customer_id": cust_id, "line_id": line["line_id"],
                          "plan_id": plan["plan_id"], "data_limit_gb": limit}
        bill_ids = []
        periods = [("2025-01-01", "2025-01-31", "2025-01-05", "2025-01-19"), ("2025-02-01", "2025-02-28", "2025-02-05", "2025-02-19")]
        statuses = ["Paid", "Issued"] if is_target else [rng.choice(["Paid", "Paid", "Overdue", "Disputed"]), rng.choice(["Issued", "Paid"])]
        for (ps, pe, issue, due), st in zip(periods, statuses):
            items = [{"description": f"{pn} - Line {ph}", "amount": pr, "date": issue, "item_type": "Plan Charge"} for pn, ph, pr in line_prices]
            bill_id = _id(rng, "B", used["id"])
            self.db["bills"].append({"bill_id": bill_id, "customer_id": cust_id, "period_start": ps, "period_end": pe, "issue_date": issue,
                                     "total_due": round(sum(i["amount"] for i in items), 2), "due_date": due, "line_items": items, "status": st})
            bill_ids.append(bill_id)
        draft_id = _id(rng, "B", used["id"])
        self.db["bills"].append({"bill_id": draft_id, "customer_id": cust_id, "period_start": "2025-03-01", "period_end": "2025-03-31",
                                 "issue_date": "2025-03-01", "total_due": 0.0, "due_date": "2025-03-15", "line_items": [], "status": "Draft"})
        bill_ids.append(draft_id)
        street, city, state, zipc = rng.choice(ADDRESSES)
        pms = [{"method_type": rng.choice(["Credit Card", "Debit Card", "PayPal"]), "account_number_last_4": f"{rng.randint(1000, 9999)}",
                "expiration_date": f"{rng.randint(1, 12):02d}/{rng.randint(2026, 2029)}"} for _ in range(rng.randint(1, 2))]
        self.db["customers"].append(
            {
                "customer_id": cust_id, "full_name": name, "date_of_birth": _dob(rng), "email": _email(name, rng),
                "phone_number": phones[0], "address": {"street": street, "city": city, "state": state, "zip_code": zipc},
                "account_status": "Active" if is_target else rng.choice(["Active", "Active", "Active", "Suspended", "Pending Verification"]),
                "payment_methods": pms, "line_ids": line_ids, "bill_ids": bill_ids,
                "created_at": f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T{rng.randint(8, 18):02d}:{rng.randint(0, 59):02d}:00",
                "last_extension_date": None, "goodwill_credit_used_this_year": float(rng.choice([0, 0, 10, 25])),
            }
        )
        if target is not None:
            target["overdue_bill_id"] = _id(rng, "B", used["id"])  # suspension 类故障要新建的账单 id
        return target


def make_instance(rng: random.Random, n_fillers: tuple[int, int] = (2, 4)) -> dict:
    """per-task 模式：{"db": <TelecomDB dict>, "target": {...}}，1 个目标客户 + 2–4 个填充客户。"""
    b = DBBuilder(rng)
    target = b.add_customer(True)
    for _ in range(rng.randint(*n_fillers)):
        b.add_customer(False)
    return {"db": b.db, "target": target}


# ---- identifier extraction (leakage checking) ----
RE_PHONE = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")
RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
RE_IMEI = re.compile(r"\b\d{15}\b")

# Fields whose values identify a record. Matching on field names rather than on shapes keeps product
# names and other free text out of the comparison, and catches bare-numeric ids that no regex would.
ID_FIELDS = {
    "user_id", "customer_id", "order_id", "product_id", "item_id", "reservation_id", "flight_number",
    "line_id", "bill_id", "device_id", "plan_id", "payment_id", "payment_method_id", "tracking_id", "imei",
}
# Fields holding a person's name. `name` alone is excluded: in retail it is the product name.
PERSON_NAME_FIELDS = {"full_name"}
RE_NAME_IN_TEXT = [
    re.compile(r"You are ([A-Z][\w'-]+ [A-Z][\w'-]+)"),
    re.compile(r"Customer name: ([A-Z][\w'-]+ [A-Z][\w'-]+)"),
]


def identifiers(obj: Any) -> dict[str, set[str]]:
    """Identifiers reachable in `obj`, bucketed. Used to prove a generated set shares none with a
    reference set."""
    out: dict[str, set[str]] = {"phones": set(), "ids": set(), "emails": set(), "imeis": set(), "names": set()}

    def add_id(v):
        if isinstance(v, str) and v.strip():
            out["ids"].add(v)
        elif isinstance(v, list):
            for x in v:
                add_id(x)

    def walk(x, key=None):
        if isinstance(x, dict):
            # a {"first_name": ..., "last_name": ...} pair is a person
            if "first_name" in x and "last_name" in x:
                out["names"].add(f"{x['first_name']} {x['last_name']}")
            for k, v in x.items():
                if k in ID_FIELDS:
                    add_id(v)
                elif k in PERSON_NAME_FIELDS and isinstance(v, str):
                    out["names"].add(v)
                walk(v, k)
        elif isinstance(x, list):
            for v in x:
                walk(v, key)
        elif isinstance(x, str):
            out["phones"].update(RE_PHONE.findall(x))
            out["emails"].update(RE_EMAIL.findall(x))
            out["imeis"].update(RE_IMEI.findall(x))
            for rx in RE_NAME_IN_TEXT:
                out["names"].update(rx.findall(x))

    walk(obj)
    return out
