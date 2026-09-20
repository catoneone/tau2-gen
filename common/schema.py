"""Schema for `Tau2Task.data`, the task payload this toolchain emits.

It mirrors tau2-bench's own `Task` model field for field (and in the same order), wrapped in a flat
episode record so that task sets and rollout traces share one JSONL schema:

    {"type": "Tau2Task", "data": <Tau2TaskData>, "key": <sha256>, "hash": <sha256>}

`schema_diff()` compares a generated payload against a reference payload key by key, so a task set can
be checked against whatever harness is going to consume it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# Field order as tau2-bench serialises it; kept identical so a diff against a reference payload is empty.
DATA_FIELDS = [
    "idx", "name", "description", "prompt", "network_allow", "network_block", "artifacts",
    "timeout", "resources", "id", "user_scenario", "ticket", "initial_state",
    "evaluation_criteria", "domain", "tau_description",
]
DEFAULT_PROMPT = "Hi! How can I help you today?"  # agent opening line; identical across the upstream telecom set


@dataclass
class Action:
    action_id: str
    requestor: str  # "user" | "assistant"
    name: str
    arguments: dict
    info: Optional[str] = None
    compare_args: Optional[list] = None


@dataclass
class EnvCall:
    env_type: str  # "user" | "assistant"
    func_name: str
    arguments: dict


@dataclass
class EnvAssertion:
    env_type: str
    func_name: str
    arguments: dict
    assert_value: bool = True
    message: Optional[str] = None


@dataclass
class UserInstructions:
    domain: str
    reason_for_call: str
    known_info: Optional[str]
    unknown_info: Optional[str]
    task_instructions: str


@dataclass
class UserScenario:
    persona: Optional[str]
    instructions: UserInstructions


@dataclass
class InitializationData:
    agent_data: Optional[dict] = None  # tau2 用 update_db 合并：list 字段整体替换 → 可装整份 DB 实例
    user_data: Optional[dict] = None


@dataclass
class InitialState:
    initialization_data: Optional[InitializationData]
    initialization_actions: Optional[list]  # list[EnvCall]
    message_history: Optional[list] = None


@dataclass
class EvaluationCriteria:
    actions: list  # list[Action]
    env_assertions: list  # list[EnvAssertion]
    communicate_info: Optional[list] = None
    nl_assertions: Optional[list] = None
    reward_basis: list = field(default_factory=lambda: ["ENV_ASSERTION"])


@dataclass
class TauDescription:
    purpose: Optional[str]
    relevant_policies: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class Tau2TaskData:
    idx: int
    name: str
    description: str  # "Purpose: ..." (tau2's Description flattened to a string)
    id: str
    user_scenario: UserScenario
    ticket: Optional[str]
    initial_state: InitialState
    evaluation_criteria: EvaluationCriteria
    domain: str
    tau_description: TauDescription
    prompt: str = DEFAULT_PROMPT
    network_allow: list = field(default_factory=lambda: ["*"])
    network_block: list = field(default_factory=list)
    artifacts: list = field(default_factory=list)
    timeout: dict = field(default_factory=dict)
    resources: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: d[k] for k in DATA_FIELDS}


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def task_key(data: dict) -> str:
    """sha256 over the canonical JSON of `data`. Only required to be stable and unique."""
    return hashlib.sha256(canonical(data).encode("utf-8")).hexdigest()


def episode_record(data: dict) -> dict:
    k = task_key(data)
    return {"type": "Tau2Task", "data": data, "key": k, "hash": k}


def to_tau2_task(data: dict):
    """Episode payload -> tau2 `Task`, for verification and for running locally."""
    from tau2.data_model.tasks import Task

    return Task.model_validate(
        {
            "id": data["id"],
            "description": data["tau_description"],
            "user_scenario": data["user_scenario"],
            "ticket": data.get("ticket"),
            "initial_state": data["initial_state"],
            "evaluation_criteria": data["evaluation_criteria"],
        }
    )


def _keyset(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _keyset(v) for k, v in d.items()}
    if isinstance(d, list) and d and isinstance(d[0], dict):
        return [_keyset(d[0])]
    return None


def schema_diff(data: dict, bench_data: dict) -> list[str]:
    """Recursively compare key sets (not values). Empty list means the schemas match."""
    diffs: list[str] = []

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                if k not in a:
                    diffs.append(f"missing in gen: {path}.{k}")
                elif k not in b:
                    diffs.append(f"extra in gen: {path}.{k}")
                else:
                    walk(a[k], b[k], f"{path}.{k}")
        elif isinstance(a, list) and isinstance(b, list):
            if a and b and isinstance(a[0], dict) and isinstance(b[0], dict):
                walk(a[0], b[0], f"{path}[0]")
        elif (a is None) != (b is None) and not (a is None or b is None):
            pass

    walk(_keyset(data), _keyset(bench_data), "data")
    return diffs
