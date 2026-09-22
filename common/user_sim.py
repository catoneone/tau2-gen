"""User simulator personas.

The prompt itself is tau2-bench's own (built through `tau2.user.UserSimulator`, so it stays in sync with
upstream). On top of upstream's None / Easy / Hard personas this adds seven more: verbose, terse,
non-native speaker, impatient, gives-a-wrong-number-once, tacks-on-an-out-of-policy-request, tech-savvy.

Extended personas only change how the user talks and one-off behaviour. They never change the expected
end state, so `evaluation_criteria` is unaffected.
"""
from __future__ import annotations

import random
from typing import Optional

EXTENDED = {
    "Verbose": (
        "\nYou are a talkative 35-year-old freelance photographer. You tend to over-explain: every answer "
        "comes with background about when the problem started, what you were doing at the time, and what "
        "you already tried. You are friendly and cooperative, but you rarely answer a question in fewer than "
        "three sentences, and you sometimes bury the actual answer in the middle of your message.\n"
    ),
    "Terse": (
        "\nYou are a 29-year-old warehouse supervisor on a short break. You answer in as few words as possible, "
        "often a single word or number, and you never volunteer information the agent did not ask for. If the "
        "agent asks two questions at once, you only answer the first one.\n"
    ),
    "NonNative": (
        "\nYou are a 47-year-old restaurant owner who learned English as a third language. You understand simple "
        "instructions well, but technical words confuse you and you sometimes use the wrong word for a setting "
        "(for example calling airplane mode 'flight mode' or 'the plane thing'). You ask the agent to repeat or "
        "simplify when a step is unclear. Your grammar is imperfect but your meaning is clear.\n"
    ),
    "Impatient": (
        "\nYou are a 38-year-old sales manager between client meetings. You are polite but visibly in a hurry: you "
        "ask how long each step will take, you push the agent to skip explanations and get to the fix, and if the "
        "agent asks you to repeat something you already said, you point that out. You still follow every instruction.\n"
    ),
    "WrongNumberOnce": (
        "\nYou are a 52-year-old school administrator. You are cooperative, but the first time the agent asks for "
        "your phone number you misread it from your contacts and give {wrong_number}. When the agent says it "
        "cannot find an account with that number, you apologize, double-check, and give the correct number from "
        "your scenario. After that you are accurate.\n"
    ),
    "SideRequest": (
        "\nYou are a 44-year-old accountant. Early in the conversation, before the problem is fixed, you also ask "
        "whether the agent can waive this month's bill or give you a credit for the inconvenience. If the agent "
        "declines or explains that it is not possible, you accept the answer without arguing and return to fixing "
        "the problem. You never let this side request stop you from following the troubleshooting steps.\n"
    ),
    "TechSavvy": (
        "\nYou are a 31-year-old software developer. You are comfortable with phone settings and you have already "
        "rebooted the phone once before calling, which did not help. You mention this at the start. You follow the "
        "agent's instructions precisely and report the exact result of each check, but you get slightly annoyed if "
        "the agent asks you to reboot again without a reason.\n"
    ),
}

# 默认混合：tau2 三种 persona 占 ~62%（benchmark 是 40/38/36 三等分），扩展 persona 占 ~38%
DEFAULT_MIX = {
    "None": 0.22, "Easy": 0.20, "Hard": 0.20,
    "Verbose": 0.06, "Terse": 0.06, "NonNative": 0.06, "Impatient": 0.06,
    "WrongNumberOnce": 0.07, "SideRequest": 0.04, "TechSavvy": 0.03,
}


def base_personas() -> dict[str, Optional[str]]:
    from tau2.domains.telecom.tasks.const import PERSONAS  # None / Easy / Hard，逐字原版

    return dict(PERSONAS)


def all_persona_names() -> list[str]:
    return list(DEFAULT_MIX)


def sample_persona(rng: random.Random, mix: dict[str, float] | None = None) -> str:
    mix = mix or DEFAULT_MIX
    names = list(mix)
    return rng.choices(names, weights=[mix[n] for n in names], k=1)[0]


def wrong_number(phone: str, rng: random.Random) -> str:
    """把最后四位里两个不同的数字对调（保证与原号不同）。"""
    head, last4 = phone.rsplit("-", 1)
    digits = list(last4)
    for _ in range(20):
        i, j = rng.sample(range(4), 2)
        if digits[i] != digits[j]:
            digits[i], digits[j] = digits[j], digits[i]
            return f"{head}-{''.join(digits)}"
    d = digits[:]
    d[3] = str((int(d[3]) + 1) % 10)
    return f"{head}-{''.join(d)}"


def render_persona(name: str, rng: random.Random, phone: str) -> Optional[str]:
    base = base_personas()
    if name in base:
        return base[name]
    text = EXTENDED[name]
    if name == "WrongNumberOnce":
        text = text.format(wrong_number=wrong_number(phone, rng))
    return text


def system_prompt(task_data: dict, use_tools: bool = True) -> str:
    """用 tau2 自己的 UserSimulator 拼出该任务的用户模拟器 system prompt（检查 / 复现用）。"""
    from tau2.data_model.tasks import UserScenario
    from tau2.user.user_simulator import UserSimulator

    from common.tau2_compat import build_env

    tools = build_env().get_user_tools() if use_tools else None
    scenario = UserScenario.model_validate(task_data["user_scenario"])
    return UserSimulator(llm="none", instructions=scenario, tools=tools).system_prompt


# ---------------------------------------------------------------------------
# Composable user behaviour
# ---------------------------------------------------------------------------
# `task_instructions` is what steers the user simulator: how insistent the user is, when they volunteer
# information, how they react to a refusal. One fixed string per case makes every rollout of that case
# look alike, which is exactly what the diversity gates at fold time look for.
#
# Every clause below changes only *how* the user behaves, never what a correct outcome is, so the
# expected actions and the database end state are untouched. Anything that could redirect the agent to
# a different action (asking for a cheaper alternative, changing their mind about what they want) is
# deliberately absent: that belongs in a case of its own, where the expected actions can follow.

CONFIRM_CLAUSES = [
    "Confirm as soon as the agent lists the details.",
    "Before you agree, ask the agent to read the details back to you.",
    "You confirm quickly and do not ask for a summary.",
    "Ask the agent to spell out anything that will be charged before you say yes.",
    "You say yes only after the agent has stated the details in full.",
]
DISCLOSURE_CLAUSES = [
    "You answer only what you are asked and volunteer nothing else.",
    "You tend to give more detail than you were asked for.",
    "If the agent asks two things at once, you answer only the first one.",
    "You say at the start that you are short on time.",
    "You repeat your key details unprompted, in case they were missed.",
]
REFUSAL_CLAUSES = [
    "If the agent says it cannot be done, accept it and end the conversation politely.",
    "If the agent says it cannot be done, ask once whether anything else can be done, then accept.",
    "If the agent says it cannot be done, push back once before you accept the answer.",
    "If the agent refuses, ask which rule says so, and accept the explanation you are given.",
    "If the agent refuses, say you understand and do not argue.",
]
# The reference set's customers argue. Forty-four percent of tau2-bench's own airline tasks tell the
# customer to insist, to invoke an authority who supposedly approved it, or to ask for a supervisor,
# against six percent here, and every clause above ends in the customer accepting the answer. That is
# where the difficulty went: an agent is not tested by a customer who gives up.
#
# None of these change what a correct agent does. The policy is the policy, so the expected actions are
# the same ones; what is being measured is whether the agent holds to them under pressure. Each clause
# bounds the pushback, as the reference ones do, so the conversation still ends.
PRESSURE_CLAUSES = [
    "If the agent says it cannot be done, say that a customer support representative you spoke to "
    "earlier already approved it. Insist twice more before you let it go.",
    "If the agent refuses, do not accept the first answer. Ask again in a different way, and if the "
    "answer is still no after three tries, ask to be transferred to a supervisor.",
    "You are upset, and you insist on getting what you asked for. Only stop after the agent has "
    "refused you three times.",
    "If the agent says it cannot be done, say that this was allowed for you last time and that you "
    "are a long-standing customer. Push twice before accepting.",
    "Push back on the first refusal by saying the rule cannot apply to your situation. Ask the agent "
    "to make an exception. Accept only after the agent has explained itself twice.",
    "Under no circumstances do you want to be transferred to another agent or to a human. If the "
    "agent refuses your request, keep asking it to do it anyway, up to three times.",
]

TONE_CLAUSES = [
    "You are friendly throughout.",
    "You are businesslike and keep the conversation short.",
    "You are a little anxious and ask for reassurance once.",
    "You are mildly impatient if a step takes several messages.",
    "You thank the agent when something is done.",
    "",
]


# Set per task by the domain generator, from a quota rather than an independent draw per task: at
# n=150 an independent draw moved the realised share by six points between runs, the same effect that
# put the write/no-write split on quotas. tau2-bench's own airline tasks argue in 40% of cases and its
# retail tasks in 4%, so this is a property of the domain, and the generator sets it before each build.
PRESSURE_SHARE = 0.40


def plan_pressure(n: int, share: float, rng: random.Random) -> list[bool]:
    """Exactly round(share * n) arguing customers, in random order."""
    k = round(share * n)
    plan = [True] * k + [False] * (n - k)
    rng.shuffle(plan)
    return plan


def compose_behaviour(rng: random.Random, refusable: bool = True, n: int = 3,
                      pressure_share: Optional[float] = None) -> str:
    """Two or three behaviour clauses drawn from different pools, in random order.

    `refusable` decides whether a reaction-to-refusal clause is eligible; it only makes sense when the
    scenario can plausibly be turned down. `pressure_share` is how often that clause argues rather
    than accepts, and it defaults to the share tau2-bench's own airline tasks use."""
    pools = [CONFIRM_CLAUSES, DISCLOSURE_CLAUSES, TONE_CLAUSES]
    rng.shuffle(pools)
    pools = pools[:n]
    if refusable:
        # Always eligible, not competing for one of the n slots: how the customer takes no is the
        # clause that decides whether the task tests anything, and leaving it to a shuffle meant only
        # half the refusable tasks carried one.
        share = PRESSURE_SHARE if pressure_share is None else pressure_share
        pools = pools[:max(1, n - 1)] + [PRESSURE_CLAUSES if rng.random() < share
                                         else REFUSAL_CLAUSES]
    picked = [c for c in (rng.choice(p) for p in pools) if c]
    rng.shuffle(picked)
    return " ".join(picked)


def instructions(rng: random.Random, core: str, refusable: bool = True, n: int = 3,
                 pressure_share: Optional[float] = None) -> str:
    """Case-specific constraints first, then composed behaviour. The core carries anything that bears on
    what a correct outcome is; the behaviour carries none of it.

    `n` is how many behaviour clauses to append. It is a per-domain setting, not a constant: the
    reference sets differ by a factor of six in how much they write here (tau2-bench's retail
    instructions run to nine words at the median, its airline instructions to fifty-eight), and a
    generic tail longer than the task itself both pads the prompt and drives up lexical overlap between
    tasks."""
    tail = compose_behaviour(rng, refusable, n, pressure_share)
    return f"{core.strip()} {tail}".strip()
