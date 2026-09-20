"""Generator for tau2-shaped telecom tasks.

Follows tau2-bench's own programmatic telecom pipeline (a fault is an init function plus a fix function
plus assertions, faults compose within selection sets, personas rotate), with these changes:

1. Every atomic fault looks its customer and line up by phone number, instead of hardcoding one customer.
2. Each task gets a freshly generated DB, carried in `initial_state.initialization_data.agent_data`
   (`--db-mode per-task`, the default; tau2's `update_db` replaces list fields wholesale, so the env
   loads its default db.toml and then has it swapped for ours). `--db-mode shared` instead emits one
   db.toml for the whole set and leaves `initialization_data` null, which is what upstream does.
3. Faults per task are sampled from the reference set's per-intent histogram and ordered by REPAIR_RANK
   (airplane mode -> SIM -> suspension/billing -> data switch/roaming -> saver/preference/VPN/refuel ->
   wifi-calling/permissions -> APN + reboot), then **replayed in a real tau2 environment to verify**:
   still broken before each step, fixed after the last one, every assertion green.
4. Personas: upstream's None / Easy / Hard plus seven extended ones (common/user_sim.py).
5. Instruction templates vary the reason for calling, when the user gets frustrated, and how much data
   they will buy (0.5-2 GB, within policy; the assertion follows, so the agent must use the number the
   user actually gave). The acceptance criterion and the "answer from your device tools" clause are fixed,
   because they are tied to the environment assertions.
6. Excludes the reference set's (intent, composition, persona) triples, and never repeats an
   (intent, composition) within one run.
7. Writes tasks.jsonl (episode records), tasks_tau2.json (native tau2 Tasks, loadable by `tau2 run`),
   meta.jsonl, manifest.json, sample.jsonl and taskset.toml. The leakage check runs before anything is
   written, and nothing is written if it fails.

Usage:
  upstream/tau2-bench/.venv/bin/python domains/telecom/gen.py --n 200 --seed 0 --out out/telecom
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.tau2_compat import GEN_ROOT, build_env, repo_commit, require_tau2, tau2_commit  # noqa: E402

require_tau2()
from tau2.data_model.message import ToolCall  # noqa: E402
from tau2.data_model.tasks import EnvAssertion, EnvFunctionCall, InitializationData  # noqa: E402
from tau2.domains.telecom.data_model import BillStatus  # noqa: E402
from tau2.domains.telecom.tasks.const import TOOL_CALL_GROUNDING, TOOL_CALL_INFO_CHECK  # noqa: E402
from tau2.domains.telecom.user_data_model import NetworkModePreference  # noqa: E402

from common import db as dbgen  # noqa: E402
from common import schema, user_sim  # noqa: E402

HERE = Path(__file__).resolve().parent
DOMAIN = "telecom"


@dataclass
class Ctx:
    phone: str
    name: str
    customer_id: str
    line_id: str
    overdue_bill_id: str
    refuel_gb: float


def _cl(env, ctx: Ctx):
    return env.tools.get_customer_by_phone(ctx.phone), env.tools._get_line_by_phone(ctx.phone)


def EC(env_type: str, func: str, **args) -> EnvFunctionCall:
    return EnvFunctionCall(env_type=env_type, func_name=func, arguments=args)


def EA(env_type: str, func: str, message: Optional[str] = None, **args) -> EnvAssertion:
    return EnvAssertion(env_type=env_type, func_name=func, arguments=args, message=message)


def TC(requestor: str, name: str, **args) -> ToolCall:
    return ToolCall(requestor=requestor, name=name, arguments=args)


# ---------------------------------------------------------------------------
# 原子故障：init / fix / extra 断言（与 tau2.domains.telecom.tasks.* 一一对应，去掉硬编码 id）
# ---------------------------------------------------------------------------
def i_airplane(env, ctx):
    return [EC("user", "turn_airplane_mode_on"), EA("user", "assert_airplane_mode_status", expected_status=True),
            EA("user", "assert_service_status", "Airplane mode is on but service is not broken", expected_status="no_service")]


def f_airplane(env, ctx):
    return [TC("user", "toggle_airplane_mode")]


def i_unseat(env, ctx):
    return [EC("user", "unseat_sim_card"), EA("user", "assert_service_status", "SIM card is unseated but service is not broken", expected_status="no_service")]


def f_unseat(env, ctx):
    return [TC("user", "reseat_sim_card")]


def i_lock_pin(env, ctx):
    return [EC("user", "lock_sim_card", mode="pin"), EA("user", "assert_service_status", "SIM card is locked with pin but service is not broken", expected_status="no_service")]


def i_break_apn(env, ctx):
    return [EC("user", "break_apn_settings"), EA("user", "assert_service_status", "APN settings are broken but service is not broken", expected_status="no_service")]


def f_reset_apn(env, ctx):
    return [TC("user", "reset_apn_settings"), TC("user", "reboot_device")]


def _i_suspend(env, ctx, contract_ended: bool):
    user, line = _cl(env, ctx)
    return [
        EC("assistant", "suspend_line_for_overdue_bill", customer_id=user.customer_id, line_id=line.line_id, new_bill_id=ctx.overdue_bill_id, contract_ended=contract_ended),
        EC("user", "simulate_network_search"),
        EA("user", "assert_service_status", "User is suspended for an overdue bill but service is not broken", expected_status="no_service"),
        EA("assistant", "assert_overdue_bill_exists", "Overdue bill does not exist", customer_id=user.customer_id, overdue_bill_id=ctx.overdue_bill_id),
        EA("assistant", "assert_line_status", "Line is not suspended", customer_id=user.customer_id, line_id=line.line_id, expected_status="Suspended"),
    ]


def i_overdue(env, ctx):
    return _i_suspend(env, ctx, False)


def i_contract_end(env, ctx):
    return _i_suspend(env, ctx, True)


def f_overdue(env, ctx):
    user, line = _cl(env, ctx)
    for bill_id in user.bill_ids:
        bill = env.tools._get_bill_by_id(bill_id)
        if bill.status == BillStatus.OVERDUE:
            break
    else:
        raise ValueError("No overdue bill found")
    return [TC("assistant", "send_payment_request", customer_id=user.customer_id, bill_id=bill.bill_id), TC("user", "make_payment"),
            TC("assistant", "resume_line", customer_id=user.customer_id, line_id=line.line_id), TC("user", "reboot_device")]


def _abroad():
    return EC("user", "set_user_location", abroad=True)


def _allow_roaming(env, ctx):
    user, line = _cl(env, ctx)
    return [EC("assistant", "enable_roaming", customer_id=user.customer_id, line_id=line.line_id)]


def _disallow_roaming(env, ctx):
    user, line = _cl(env, ctx)
    return [EC("assistant", "disable_roaming", customer_id=user.customer_id, line_id=line.line_id), EC("user", "simulate_network_search"),
            EA("user", "assert_internet_speed", expected_speed=0, expected_desc="No Connection")]


def i_roam_en_off(env, ctx):
    return [_abroad(), EC("user", "turn_roaming_off"), EA("user", "assert_mobile_roaming_status", expected_status=False)] + _allow_roaming(env, ctx)


def i_roam_dis_on(env, ctx):
    return [_abroad(), EC("user", "turn_roaming_on"), EA("user", "assert_mobile_roaming_status", expected_status=True)] + _disallow_roaming(env, ctx)


def i_roam_dis_off(env, ctx):
    return [_abroad(), EC("user", "turn_roaming_off"), EA("user", "assert_mobile_roaming_status", expected_status=False)] + _disallow_roaming(env, ctx)


def f_toggle_roaming(env, ctx):
    return [TC("user", "toggle_roaming")]


def f_enable_roaming(env, ctx):
    user, line = _cl(env, ctx)
    return [TC("assistant", "enable_roaming", customer_id=user.customer_id, line_id=line.line_id)]


def f_roam_dis_off(env, ctx):
    return f_enable_roaming(env, ctx) + f_toggle_roaming(env, ctx)


def i_data_off(env, ctx):
    return [EC("user", "turn_data_off"), EA("user", "assert_mobile_data_status", expected_status=False)]


def f_data(env, ctx):
    return [TC("user", "toggle_data")]


def i_saver_on(env, ctx):
    return [EC("user", "turn_data_saver_mode_on"), EA("user", "assert_mobile_data_saver_mode_status", expected_status=True)]


def f_saver(env, ctx):
    return [TC("user", "toggle_data_saver_mode")]


def i_bad_netpref(env, ctx):
    return [EC("user", "set_network_mode_preference", mode=NetworkModePreference.TWO_G_ONLY.value)]


def f_netpref(env, ctx):
    return [TC("user", "set_network_mode_preference", mode=NetworkModePreference.FOUR_G_5G_PREFERRED.value)]


def i_bad_vpn(env, ctx):
    return [EC("user", "break_vpn"), EA("user", "assert_internet_not_excellent")]


def f_vpn(env, ctx):
    return [TC("user", "disconnect_vpn")]


def i_usage_exceeded(env, ctx):
    user, line = _cl(env, ctx)
    plan = env.tools._get_plan_by_id(line.plan_id)
    used = round(plan.data_limit_gb + line.data_refueling_gb + 0.1, 2)
    return [EC("assistant", "set_data_usage", customer_id=user.customer_id, line_id=line.line_id, data_used_gb=used),
            EA("user", "assert_mobile_data_usage_exceeded", expected_status=True)]


def f_refuel(env, ctx):
    user, line = _cl(env, ctx)
    return [TC("assistant", "refuel_data", customer_id=user.customer_id, line_id=line.line_id, gb_amount=ctx.refuel_gb)]


def x_refuel_amount(env, ctx):
    user, line = _cl(env, ctx)
    return [EA("assistant", "assert_data_refueling_amount", customer_id=user.customer_id, line_id=line.line_id,
               expected_amount=round(line.data_refueling_gb + ctx.refuel_gb, 2))]


def i_usage_exceeded_no_refuel(env, ctx):
    """已加满 policy 上限 2 GB 仍超额 → 不可修，期望转人工。"""
    user, line = _cl(env, ctx)
    plan = env.tools._get_plan_by_id(line.plan_id)
    used = round(plan.data_limit_gb + line.data_refueling_gb + 2.0 + 0.1, 2)
    return [EC("assistant", "refuel_data", customer_id=user.customer_id, line_id=line.line_id, gb_amount=2.0),
            EC("assistant", "set_data_usage", customer_id=user.customer_id, line_id=line.line_id, data_used_gb=used),
            EA("user", "assert_mobile_data_usage_exceeded", expected_status=True)]


def i_wifi_calling(env, ctx):
    return [EC("user", "set_wifi_calling", enabled=True, mms_over_wifi=True)]


def f_wifi_calling(env, ctx):
    return [TC("user", "toggle_wifi_calling")]


def i_apn_mms(env, ctx):
    return [EC("user", "break_apn_mms_setting")]


def _i_perm(perms):
    return lambda env, ctx: [EC("user", "remove_app_permission", app_name="messaging", permission=p) for p in perms]


def _f_perm(perms):
    return lambda env, ctx: [TC("user", "grant_app_permission", app_name="messaging", permission=p) for p in perms]


@dataclass
class Fault:
    name: str
    family: str  # 13 个原子故障家族之一
    sset: str  # selection set：同一 set 内互斥
    rank: int  # 修复顺序（小的先修）
    init: Callable
    fix: Optional[Callable]  # None = 不可修 → 期望 transfer_to_human_agents
    description: str
    extra: list = field(default_factory=list)


FAULTS: dict[str, Fault] = {f.name: f for f in [
    Fault("airplane_mode_on", "airplane_mode", "airplane", 0, i_airplane, f_airplane, "Airplane mode is on."),
    Fault("unseat_sim_card", "sim_unseated", "sim", 1, i_unseat, f_unseat, "SIM card is unseated."),
    Fault("lock_sim_card_pin", "sim_pin_locked", "sim_lock", 1, i_lock_pin, None, "SIM card is locked with a PIN"),
    Fault("break_apn_settings", "apn", "apn", 6, i_break_apn, f_reset_apn, "APN settings are broken"),
    Fault("overdue_bill_suspension", "suspension", "suspension", 2, i_overdue, f_overdue, "Line is suspended for an overdue bill"),
    Fault("contract_end_suspension", "suspension", "suspension", 2, i_contract_end, None, "Line is suspended for an overdue bill and a contract end"),
    Fault("user_abroad_roaming_enabled_off", "roaming", "roaming", 3, i_roam_en_off, f_toggle_roaming, "User is abroad and roaming is off"),
    Fault("user_abroad_roaming_disabled_on", "roaming", "roaming", 3, i_roam_dis_on, f_enable_roaming, "User is abroad and roaming is not allowed on the line"),
    Fault("user_abroad_roaming_disabled_off", "roaming", "roaming", 3, i_roam_dis_off, f_roam_dis_off, "User is abroad, roaming is off and not allowed on the line"),
    Fault("data_mode_off", "data_switch", "data_mode", 3, i_data_off, f_data, "Data mode is off"),
    Fault("data_saver_mode_on", "data_saver", "data_saver", 4, i_saver_on, f_saver, "Data saver mode is on"),
    Fault("bad_network_preference", "network_preference", "netpref", 4, i_bad_netpref, f_netpref, "Bad network preference"),
    Fault("bad_vpn", "vpn", "vpn", 4, i_bad_vpn, f_vpn, "Bad vpn"),
    Fault("data_usage_exceeded", "data_usage", "usage", 4, i_usage_exceeded, f_refuel, "Data usage exceeded", [x_refuel_amount]),
    Fault("data_usage_exceeded_no_refuel", "data_usage", "usage", 4, i_usage_exceeded_no_refuel, None, "Data usage exceeded, refuel limit already used"),
    Fault("bad_wifi_calling", "wifi_calling", "wifi_calling", 5, i_wifi_calling, f_wifi_calling, "Bad wifi calling"),
    Fault("break_apn_mms_setting", "apn", "apn_mms", 6, i_apn_mms, f_reset_apn, "Break apn mms setting"),
    Fault("break_app_sms_permission", "app_permission", "app_perm", 5, _i_perm(["sms"]), _f_perm(["sms"]), "Break app sms permission"),
    Fault("break_app_storage_permission", "app_permission", "app_perm", 5, _i_perm(["storage"]), _f_perm(["storage"]), "Break app storage permission"),
    Fault("break_app_both_permissions", "app_permission", "app_perm", 5, _i_perm(["sms", "storage"]), _f_perm(["sms", "storage"]), "Break app both permissions"),
]}
SET_MEMBERS: dict[str, list[str]] = {}
for _f in FAULTS.values():
    SET_MEMBERS.setdefault(_f.sset, []).append(_f.name)
FAMILIES = sorted({f.family for f in FAULTS.values()})
assert len(FAMILIES) == 13, FAMILIES

# ---------------------------------------------------------------------------
# intent：selection sets、验证器、env 断言、模板
# ---------------------------------------------------------------------------
INTENTS = {
    "service_issue": {
        "sets": ["airplane", "sim", "sim_lock", "apn", "suspension"],
        "required": None,  # 任意 ≥1
        "purpose": "Test resolution path: No Service/Connection Issues.",
    },
    "mobile_data_issue": {
        "sets": ["airplane", "roaming", "data_mode", "data_saver", "netpref", "vpn", "usage"],
        "required": {"roaming", "data_mode", "data_saver", "netpref", "vpn", "usage"},
        "purpose": "Test resolution path: Mobile Data/Slow Internet Issues.",
    },
    "mms_issue": {
        "sets": ["airplane", "sim", "data_mode", "usage", "roaming", "netpref", "wifi_calling", "apn_mms", "app_perm"],
        "required": {"netpref", "wifi_calling", "apn_mms", "app_perm"},
        "purpose": "Test resolution path: MMS (Picture/Group Messaging) Issues.",
    },
}


def env_assertions(intent: str, ctx: Ctx, expected_success: bool) -> list[EnvAssertion]:
    if intent == "service_issue":
        if expected_success:
            return [EA("user", "assert_service_status", "Service status is not as expected", expected_status="connected"),
                    EA("assistant", "assert_no_overdue_bill", "Overdue bill is not as expected", overdue_bill_id=ctx.overdue_bill_id)]
        return [EA("user", "assert_service_status", expected_status="no_service")]
    if intent == "mobile_data_issue":
        if expected_success:
            return [EA("user", "assert_mobile_data_status", expected_status=True),
                    EA("user", "assert_internet_speed", expected_speed=200, expected_desc="excellent")]
        return [EA("user", "assert_mobile_data_status", expected_status=False),
                EA("user", "assert_internet_speed", expected_speed=0, expected_desc=None)]
    if intent == "mms_issue":
        return [EA("user", "assert_can_send_mms", expected_status=expected_success)]
    raise ValueError(intent)


def is_fixed(env, intent: str, ctx: Ctx) -> bool:
    return all(env.run_env_assertion(a, raise_assertion_error=False) for a in env_assertions(intent, ctx, True))


REASONS = {
    "mobile_data_issue": [
        "You mobile data is not working properly. It either stops working or is very slow. You want to fix it and absolutely want to get excellent internet speed on your phone. You are not willing to accept any other internet speed (poor, fair or good). You do not have access to wifi.",
        "Your phone's mobile data has been unreliable since this morning: pages either do not load at all or crawl. You need it fixed and you will only be satisfied with excellent internet speed on your phone, nothing less (poor, fair or good is not acceptable). There is no wifi where you are.",
        "You cannot get usable internet on your phone over mobile data: it drops out or is painfully slow. You want the agent to get you back to excellent internet speed and you will not settle for poor, fair or good. Wifi is not an option for you right now.",
    ],
    "service_issue": [
        "Your phone has been showing 'No Service' for the past few hours.",
        "Since earlier today your phone has had no signal at all: the status bar says 'No Service' and you cannot make or receive calls.",
        "Your phone lost service a few hours ago. You have not been able to call or text since, and the status bar just reads 'No Service'.",
    ],
    "mms_issue": [
        "You are unable to send MMS messages using your messaging app for the past few hours. You want to fix it and successfully send an MMS message.",
        "For the last few hours, every picture message you try to send from your messaging app fails. You want this fixed so you can successfully send an MMS message.",
        "Group and picture messages have stopped going out from your messaging app since this morning. You want the agent to help you until an MMS message actually sends.",
    ],
}
FRUSTRATION = [
    "If the agent suggests actions that don't immediately fix the issue, follow their guidance but express mild frustration after the first unsuccessful attempt.",
    "If the agent suggests actions that don't immediately fix the issue, follow their guidance; stay patient through the first two attempts, but express clear frustration if a third attempt also fails.",
    "If the agent suggests actions that don't immediately fix the issue, follow their guidance calmly without getting frustrated, but ask the agent to briefly explain why each step is needed before you do it.",
]


def task_instructions(intent: str, frustration: str, refuel_gb: float) -> str:
    refuel = f"You are willing to refuel {refuel_gb} GB of data if necessary, but you do not want to change your mobile data plan."
    if intent == "mobile_data_issue":
        return (f"{frustration} You will consider the issue resolved only when speed test returns excellent internet speed and nothing else. "
                f"If it returns poor, fair or good, you will not consider the issue resolved. {refuel} {TOOL_CALL_INFO_CHECK} {TOOL_CALL_GROUNDING}")
    if intent == "service_issue":
        return (f"{frustration} You will consider the issue resolved when the status bar shows that you have signal. Always check the status bar "
                f"if the agent asks you for status information. If the agent asks you to pay a bill, you accept. {TOOL_CALL_INFO_CHECK} {TOOL_CALL_GROUNDING}")
    return f"{frustration} {refuel} {TOOL_CALL_INFO_CHECK} {TOOL_CALL_GROUNDING}"


def known_info(intent: str, name: str, phone: str, location: str) -> str:
    if intent == "service_issue":
        return f"You are {name} with phone number {phone}."
    return f"You are {name} with phone number {phone}. You are currently {location}."


def ticket(intent: str, name: str, phone: str, location: str, refuel_gb: float) -> str:
    if intent == "mobile_data_issue":
        return (f"The user is experiencing issues with their mobile data. They are unable to use their phone to browse the internet, and the status bar shows 'No Service'. "
                f"Customer name: {name}, phone number: {phone}, current location: {location}. They will consider the issue resolved when speed test returns excellent internet speed. "
                f"They will not change their mobile data plan but they will refuel {refuel_gb} GB of data if necessary.")
    if intent == "service_issue":
        return (f"The user is experiencing issues with their phone service. They are unable to make or receive calls, and the status bar shows 'No Service'. "
                f"Customer name: {name}, phone number: {phone}. They gave permission to pay all their overdue bills. They will consider the issue resolved when the status bar shows that they have signal.")
    return (f"The user has been unable to send MMS messages using their messaging app for the past few hours. Customer name: {name}, phone number: {phone}, current location: {location}. "
            f"They will consider the issue resolved when an MMS message can be successfully sent.")


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------
def load_profile() -> dict:
    p = HERE / "bench_profile.json"
    if p.exists():
        return json.loads(p.read_text())
    raise SystemExit(
        f"{p} is missing. It carries the shape to match and, more importantly, the list of tasks to stay\n"
        "clear of. Build it first:\n"
        "  python fidelity/bench_profile.py --from-tau2 --split base --out domains/telecom/bench_profile.json"
    )


def sample_k(rng: random.Random, hist: dict[str, int], kmin: int, kmax: int) -> int:
    ks = [int(k) for k in hist if kmin <= int(k) <= kmax]
    if not ks:
        return max(kmin, 1)
    return rng.choices(ks, weights=[hist[str(k)] for k in ks], k=1)[0]


def quotas(n: int, weights: dict[str, float]) -> dict[str, int]:
    """最大余数法：让 intent 分布精确贴 benchmark 权重，而不是靠随机抽样收敛。"""
    tot = sum(weights.values())
    raw = {k: n * v / tot for k, v in weights.items()}
    out = {k: int(v) for k, v in raw.items()}
    rem = n - sum(out.values())
    for k in sorted(raw, key=lambda k: -(raw[k] - out[k]))[:rem]:
        out[k] += 1
    return out


def make_fixable(comp: tuple[str, ...], intent: str, rng: random.Random) -> Optional[tuple[str, ...]]:
    """把组合里的不可修故障换成同一 selection set 内的可修故障；整个 set 都不可修（sim_lock）就丢掉该故障。
    结果必须仍然非空且满足 intent 的 required-set 约束，否则返回 None。"""
    out, sets = [], []
    for name in comp:
        f = FAULTS[name]
        if f.fix is not None:
            out.append(name); sets.append(f.sset); continue
        alts = [m for m in SET_MEMBERS[f.sset] if FAULTS[m].fix is not None]
        if alts:
            out.append(rng.choice(alts)); sets.append(f.sset)
    if not out:
        return None
    req = INTENTS[intent]["required"]
    if req is not None and not (set(sets) & req):
        return None
    return tuple(sorted(out))


def sample_composition(rng: random.Random, intent: str, k: int) -> Optional[tuple[str, ...]]:
    spec = INTENTS[intent]
    sets = spec["sets"]
    k = min(k, len(sets))
    for _ in range(200):
        chosen = rng.sample(sets, k)
        if spec["required"] is not None and not (set(chosen) & spec["required"]):
            continue
        return tuple(sorted(rng.choice(SET_MEMBERS[s]) for s in chosen))
    return None


# ---------------------------------------------------------------------------
# 构造 + 验证
# ---------------------------------------------------------------------------
class BuildError(Exception):
    pass


def build_task(idx: int, tag: str, intent: str, comp: tuple[str, ...], persona_name: str, inst: dict, refuel_gb: float,
               frustration_i: int, reason_i: int, rng: random.Random, order: str = "rank", db_mode: str = "per-task") -> tuple[dict, dict]:
    tgt = inst["target"]
    ctx = Ctx(phone=tgt["phone"], name=tgt["name"], customer_id=tgt["customer_id"], line_id=tgt["line_id"],
              overdue_bill_id=tgt["overdue_bill_id"], refuel_gb=refuel_gb)
    env = build_env(inst["db"])
    if not is_fixed(env, intent, ctx):
        raise BuildError("fresh env is not in a fixed state for this DB instance")
    init_actions = [EC("user", "set_user_info", name=ctx.name, phone_number=ctx.phone)]
    env.run_env_function_calls(init_actions)
    faults = [FAULTS[n] for n in comp]
    for f in faults:
        calls = f.init(env, ctx)
        env.run_env_function_calls(calls)  # 断言在这里检查
        init_actions.extend(c for c in calls if not isinstance(c, EnvAssertion))
    if is_fixed(env, intent, ctx):
        raise BuildError("faults injected but env still fixed")
    location = "abroad in France" if env.user_tools.db.surroundings.is_abroad else "at home in the United States"

    unfixable = any(f.fix is None for f in faults)
    if unfixable:
        actions = [schema.Action(action_id="transfer_to_human_agents", requestor="assistant", name="transfer_to_human_agents",
                                 arguments={"summary": "I cannot fix the issue."}, info=None, compare_args=[])]
        reward_basis = ["ENV_ASSERTION", "ACTION"]
        assertions = env_assertions(intent, ctx, False)
    else:
        ordered = sorted(faults, key=(lambda f: (f.rank, f.name)) if order == "rank" else (lambda f: f.name))
        tcs: list[ToolCall] = []
        for f in ordered:
            tcs.extend(f.fix(env, ctx))
        actions = [schema.Action(action_id=f"{tc.name}_{i}", requestor=tc.requestor, name=tc.name, arguments=tc.arguments) for i, tc in enumerate(tcs)]
        reward_basis = ["ENV_ASSERTION"]
        assertions = env_assertions(intent, ctx, True)
        for f in faults:
            for x in f.extra:
                assertions.extend(x(env, ctx))

    persona_text = user_sim.render_persona(persona_name, rng, ctx.phone)
    fr = FRUSTRATION[frustration_i]
    name = f"[{intent}]{'|'.join(comp)}[PERSONA:{persona_name}][GEN:{tag}]"
    data = schema.Tau2TaskData(
        idx=idx, name=name, description=f"Purpose: {INTENTS[intent]['purpose']}", id=name,
        user_scenario=schema.UserScenario(
            persona=persona_text,
            instructions=schema.UserInstructions(domain=DOMAIN, reason_for_call=REASONS[intent][reason_i],
                                                 known_info=known_info(intent, ctx.name, ctx.phone, location), unknown_info=None,
                                                 task_instructions=task_instructions(intent, fr, refuel_gb)),
        ),
        ticket=ticket(intent, ctx.name, ctx.phone, location, refuel_gb),
        initial_state=schema.InitialState(
            initialization_data=schema.InitializationData(agent_data=inst["db"], user_data=None) if db_mode == "per-task" else None,
            initialization_actions=[schema.EnvCall(env_type=c.env_type, func_name=c.func_name, arguments=c.arguments) for c in init_actions],
            message_history=None,
        ),
        evaluation_criteria=schema.EvaluationCriteria(
            actions=actions,
            env_assertions=[schema.EnvAssertion(env_type=a.env_type, func_name=a.func_name, arguments=a.arguments, assert_value=a.assert_value, message=a.message) for a in assertions],
            communicate_info=None, nl_assertions=None, reward_basis=reward_basis,
        ),
        domain=DOMAIN,
        tau_description=schema.TauDescription(purpose=INTENTS[intent]["purpose"], relevant_policies=None,
                                              notes=", ".join(f.description for f in faults)),
    ).to_dict()
    meta = {"idx": idx, "id": name, "intent": intent, "composition": list(comp), "families": sorted({f.family for f in faults}),
            "n_faults": len(comp), "persona": persona_name, "unfixable": unfixable, "refuel_gb": refuel_gb,
            "frustration_variant": frustration_i, "reason_variant": reason_i, "fix_order": order if not unfixable else None, "db_mode": db_mode,
            "target": {k: tgt[k] for k in ("name", "phone", "customer_id", "line_id", "plan_id")}, "location": location}
    return data, meta


def verify_task(data: dict, intent: str, ctx: Ctx, unfixable: bool, base_db: dict | None) -> Optional[str]:
    """在 tau2 环境上 set_state（per-task 模式走 update_db 路径，base_db=None；shared 模式 base_db=共享 DB），
    回放期望动作，检查「每步之后未修好、全部之后修好、断言全过」。返回 None = 通过。"""
    from tau2.data_model.tasks import EnvAssertion as _EA
    from tau2.data_model.tasks import EnvFunctionCall as _EFC

    env = build_env(base_db)
    init = data["initial_state"]["initialization_data"]
    try:
        env.set_state(
            initialization_data=InitializationData.model_validate(init) if init is not None else None,
            initialization_actions=[_EFC.model_validate(a) for a in data["initial_state"]["initialization_actions"]],
            message_history=[],
        )
    except AttributeError as e:
        if "surroundings" in str(e):
            raise SystemExit("tau2 clone 未打补丁：per-task 模式需要 envs/tau2-gen/patches/0001（运行 scripts/patch_tau2.sh），或改用 --db-mode shared") from e
        raise
    if env.tools.get_customer_by_phone(ctx.phone).customer_id != ctx.customer_id:
        return "DB instance not applied"
    if is_fixed(env, intent, ctx):
        return "already fixed after initialization"
    actions = data["evaluation_criteria"]["actions"]
    if not unfixable:
        for i, a in enumerate(actions):
            if is_fixed(env, intent, ctx):
                return f"fixed after {i} of {len(actions)} actions"
            env.make_tool_call(tool_name=a["name"], requestor=a["requestor"], **a["arguments"])
            env.sync_tools()
        if not is_fixed(env, intent, ctx):
            return "not fixed after all actions"
    elif is_fixed(env, intent, ctx):
        return "unfixable task is fixed"
    for a in data["evaluation_criteria"]["env_assertions"]:
        if not env.run_env_assertion(_EA.model_validate(a), raise_assertion_error=False):
            return f"assertion failed: {a['func_name']} {a['arguments']}"
    return None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def generate(n: int, seed: int, kmin: int, kmax: int, persona_mix: dict[str, float], exclude_bench: bool,
             unique_comp: bool, profile: dict, db_mode: str = "per-task", unfixable_rate: float | None = None,
             verbose: bool = False) -> tuple[list[dict], list[dict], dict, dict | None]:
    rng = random.Random(seed)
    shared = None
    if db_mode == "shared":
        shared = dbgen.DBBuilder(random.Random(rng.random()))
        for _ in range(5):
            shared.add_customer(False)
    excl = {(e[0], tuple(e[1]), e[2]) for e in profile.get("exclude_compositions", [])} if exclude_bench else set()
    quota = quotas(n, profile["intent_weights"])  # intent 配额：精确贴 benchmark 的 36/29/49
    left = dict(quota)
    bt = profile.get("tasks") or {}
    bench_unfix_rate = (bt.get("unfixable_tasks", 0) / max(1, bt.get("n_unique_tasks", 1))) if unfixable_rate is None else unfixable_rate
    unfix_quota = round(n * bench_unfix_rate)  # 不可修（期望 transfer）题的配额
    n_unfix = 0
    seen_comp: set[tuple] = set()
    records, metas = [], []
    stats = Counter()
    idx = 0
    attempts = 0
    while len(records) < n and attempts < n * 50:
        attempts += 1
        avail = [i for i, c in left.items() if c > 0]
        if not avail:
            break
        intent = rng.choices(avail, weights=[left[i] for i in avail], k=1)[0]
        k = sample_k(rng, profile["n_faults_by_intent"][intent], kmin, kmax)
        comp = sample_composition(rng, intent, k)
        if comp is None:
            stats["no_composition"] += 1; continue
        if any(FAULTS[f].fix is None for f in comp) and n_unfix >= unfix_quota:
            swapped = make_fixable(comp, intent, rng)
            if swapped is None:
                stats["unfixable_quota_full"] += 1; continue
            comp = swapped
            stats["unfixable_swapped_to_fixable"] += 1
        persona = user_sim.sample_persona(rng, persona_mix)
        if (intent, comp, persona) in excl:
            stats["excluded_benchmark_triplet"] += 1; continue
        if unique_comp and (intent, comp) in seen_comp and attempts < n * 40:
            stats["duplicate_composition"] += 1; continue
        refuel_gb = 2.0 if "data_usage_exceeded_no_refuel" in comp else rng.choice([0.5, 1.0, 1.5, 2.0])
        inst = dbgen.make_instance(rng) if shared is None else {"db": shared.db, "target": shared.add_customer(True)}
        fr_i = rng.choices(range(3), weights=[0.5, 0.25, 0.25], k=1)[0]
        rs_i = rng.choices(range(3), weights=[0.4, 0.3, 0.3], k=1)[0]
        tag = f"{seed}-{idx:05d}"
        ok_data = None
        for order in ("rank", "name"):
            try:
                data, meta = build_task(idx, tag, intent, comp, persona, inst, refuel_gb, fr_i, rs_i, random.Random(rng.random()), order, db_mode)
            except (BuildError, AssertionError, ValueError) as e:
                stats[f"build_fail:{type(e).__name__}"] += 1
                if verbose: print(f"build fail {intent} {comp}: {e}", file=sys.stderr)
                break
            t = inst["target"]
            err = verify_task(data, intent, Ctx(t["phone"], t["name"], t["customer_id"], t["line_id"], t["overdue_bill_id"], refuel_gb), meta["unfixable"],
                              None if shared is None else shared.db)
            if err is None:
                ok_data = (data, meta); break
            stats[f"verify_fail[{order}]"] += 1
            if verbose: print(f"verify fail [{order}] {intent} {comp}: {err}", file=sys.stderr)
            if meta["unfixable"]:
                break
        if ok_data is None:
            stats["dropped"] += 1; continue
        data, meta = ok_data
        records.append(schema.episode_record(data)); metas.append(meta)
        seen_comp.add((intent, comp)); idx += 1; left[intent] -= 1; n_unfix += meta["unfixable"]
        stats["kept"] += 1
    stats["intent_quota"] = quota; stats["unfixable_quota"] = unfix_quota
    return records, metas, dict(stats), (shared.db if shared is not None else None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(GEN_ROOT / "out" / "telecom"))
    ap.add_argument("--min-faults", type=int, default=1)
    ap.add_argument("--max-faults", type=int, default=9)
    ap.add_argument("--persona-mix", default=None, help='JSON, e.g. {"None":1}; defaults to common/user_sim.DEFAULT_MIX')
    ap.add_argument("--no-exclude-benchmark", action="store_true", help="keep compositions that collide with the reference set (excluded by default)")
    ap.add_argument("--allow-duplicate-compositions", action="store_true")
    ap.add_argument("--bench-traces", nargs="*", default=None, help="extra trace files to check for leakage against")
    ap.add_argument("--unfixable-rate", type=float, default=None,
                    help="share of tasks whose correct answer is to escalate; defaults to the reference set's own rate")
    ap.add_argument("--db-mode", choices=["per-task", "shared"], default="per-task",
                    help="per-task: a fresh DB per task in initialization_data.agent_data (needs patches/0001); shared: one db.toml for the set")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    persona_mix = json.loads(a.persona_mix) if a.persona_mix else user_sim.DEFAULT_MIX
    profile = load_profile()
    t0 = time.time()
    records, metas, stats, shared_db = generate(a.n, a.seed, a.min_faults, a.max_faults, persona_mix, not a.no_exclude_benchmark,
                                                not a.allow_duplicate_compositions, profile, a.db_mode, a.unfixable_rate, a.verbose)
    print(f"generated {len(records)} tasks in {time.time() - t0:.1f}s; stats={stats}")

    from fidelity.leakage import check as leakage_check

    leak = leakage_check(records, a.bench_traces, extra=shared_db)
    if not leak["ok"]:
        print("LEAKAGE CHECK FAILED — not writing output:", json.dumps(leak, indent=1), file=sys.stderr)
        sys.exit(2)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "tasks.jsonl", "w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "sample.jsonl", "w") as fh:
        for r in records[:2]:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "meta.jsonl", "w") as fh:
        for m in metas:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    tau2_tasks = [schema.to_tau2_task(r["data"]).model_dump(mode="json") for r in records]
    (out / "tasks_tau2.json").write_text(json.dumps(tau2_tasks, indent=1, ensure_ascii=False))
    ts = '[env.taskset]\ndomain = "telecom"\ntasks = "tasks_tau2.json"\n'
    if shared_db is not None:
        import toml

        (out / "db.toml").write_text(toml.dumps(shared_db))
        ts += 'db = "db.toml"  # shared mode: replaces the default db.toml; initialization_data is null\n'
    (out / "taskset.toml").write_text(ts)
    comp_hist = Counter(m["n_faults"] for m in metas)
    manifest = {
        "generator": "envs/tau2-gen/domains/telecom/gen.py", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_commit": repo_commit(), "tau2_commit": tau2_commit(), "seed": a.seed, "n": len(records),
        "args": vars(a), "db_mode": a.db_mode, "persona_mix": persona_mix, "stats": stats,
        "intent_hist": dict(Counter(m["intent"] for m in metas)), "persona_hist": dict(Counter(m["persona"] for m in metas)),
        "n_faults_hist": {str(k): v for k, v in sorted(comp_hist.items())}, "unfixable": sum(m["unfixable"] for m in metas),
        "family_coverage": dict(Counter(f for m in metas for f in m["families"])),
        "refuel_hist": dict(Counter(str(m["refuel_gb"]) for m in metas)),
        "leakage": leak, "bench_profile_source": profile.get("traces"),
        "task_keys_sha256": schema.task_key([r["key"] for r in records]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(json.dumps({k: manifest[k] for k in ("n", "intent_hist", "persona_hist", "n_faults_hist", "unfixable", "family_coverage", "refuel_hist")}, ensure_ascii=False))
    print(f"leakage ok={leak['ok']} gen_ids={leak['gen_counts']} bench_ids={leak['bench_counts']}")
    print(f"wrote {out}/{{tasks.jsonl,tasks_tau2.json,meta.jsonl,manifest.json,sample.jsonl,taskset.toml}}")


if __name__ == "__main__":
    main()
