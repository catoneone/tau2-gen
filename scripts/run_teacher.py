"""Run a reference model over a generated task set with tau2's runner, k trials per task.

The agent is any OpenAI-compatible endpoint (a local vLLM server, for instance). The user simulator is
gpt-4.1 by default (its key read from a gitignored `.env`), or ANY OpenAI-compatible endpoint: pass
`--user-model <name> --user-api-base <url> --user-api-key-var <ENV>` and the customer is played by that
model (e.g. a cheap hosted DeepSeek). With `--user-api-base` the model is routed through litellm's
`openai/` provider automatically, so the name is whatever the endpoint serves. Keys never appear on the
command line: both sides name the environment variable that holds them.

Usage:
  upstream/tau2-bench/.venv/bin/python scripts/run_teacher.py \
      --tasks out/telecom/tasks_tau2.json --agent-model <model> --agent-api-base http://127.0.0.1:8000/v1 \
      --user-llm gpt-4.1 --trials 3 --concurrency 8 --save-to out/telecom/runs/teacher_k3.json

  # same, with a self-hosted / third-party customer simulator
  ... --user-model deepseek-v4.1-flash --user-api-base https://api.example.com/v1 --user-api-key-var CUSTOMER_KEY

Then convert the results to the trace schema:
  python fidelity/convert.py --results out/telecom/runs/teacher_k3.json --tasks out/telecom/tasks.jsonl \
      --out out/telecom/runs/teacher_k3/traces.jsonl.gz

Note: k trials only differ if you sample with a temperature above 0.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.tau2_compat import require_tau2  # noqa: E402

require_tau2()


def user_simulator_config(a: argparse.Namespace) -> tuple[str, dict]:
    """(litellm model string, llm_args) for the customer simulator.

    Default: `gpt-4.1` through litellm's own provider routing, key from OPENAI_API_KEY.
    With `--user-api-base`: an OpenAI-compatible endpoint. litellm needs a provider prefix to know how
    to talk to a custom base, so a bare model name gets `openai/` (a name that already carries a
    provider, `openai/...`, `hosted_vllm/...`, is passed through). The key comes from
    `--user-api-key-var`; a missing variable is an error, not a silent "EMPTY".
    """
    model = a.user_llm
    args: dict = {"temperature": a.user_temperature}
    if a.user_max_tokens:
        args["max_tokens"] = a.user_max_tokens
    if a.user_api_base:
        if "/" not in model:
            model = f"openai/{model}"
        key = os.environ.get(a.user_api_key_var)
        if not key:
            raise SystemExit(f"--user-api-base given but ${a.user_api_key_var} is not set (put it in .env or the environment)")
        args |= {"api_base": a.user_api_base, "api_key": key}
    return model, args


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True, help="tasks_tau2.json (a list of native tau2 Tasks)")
    ap.add_argument("--agent-model", required=True, help="model name as the endpoint expects it")
    ap.add_argument("--agent-api-base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--agent-api-key-var", default="VLLM_API_KEY",
                    help="env var holding the agent-side key; the key itself is never passed on the command line")
    ap.add_argument("--agent-temperature", type=float, default=0.7, help="must be > 0 for k trials to differ")
    ap.add_argument("--agent-max-tokens", type=int, default=8192)
    ap.add_argument("--user-llm", "--user-model", dest="user_llm", default="gpt-4.1",
                    help="model that plays the customer (litellm name; with --user-api-base, the name the endpoint serves)")
    ap.add_argument("--user-api-base", default=None, help="OpenAI-compatible base URL for the user simulator")
    ap.add_argument("--user-api-key-var", default="OPENAI_API_KEY", help="env var holding the user-side key")
    ap.add_argument("--user-temperature", type=float, default=0.0)
    ap.add_argument("--user-max-tokens", type=int, default=None)
    ap.add_argument("--domain", default="telecom", choices=["telecom", "airline", "retail"])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=300)
    ap.add_argument("--task-ids", nargs="*", default=None)
    ap.add_argument("--num-tasks", type=int, default=None)
    ap.add_argument("--save-to", required=True)
    a = ap.parse_args()

    from dotenv import load_dotenv

    load_dotenv()  # OPENAI_API_KEY comes from .env only, which is gitignored
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner import run_tasks
    from tau2.utils import load_file

    from common import schema  # noqa: F401  (kept for symmetry with the rest of the toolchain)

    raw = load_file(a.tasks)
    from tau2.data_model.tasks import Task

    tasks = [Task.model_validate(t) for t in (raw["tasks"] if isinstance(raw, dict) and "tasks" in raw else raw)]
    if a.task_ids:
        tasks = [t for t in tasks if t.id in set(a.task_ids)]
    if a.num_tasks:
        tasks = tasks[: a.num_tasks]
    user_llm, user_args = user_simulator_config(a)
    cfg = TextRunConfig(
        domain=a.domain,
        llm_agent=f"openai/{a.agent_model}",
        llm_args_agent={"temperature": a.agent_temperature, "max_tokens": a.agent_max_tokens, "api_base": a.agent_api_base, "api_key": os.environ.get(a.agent_api_key_var, "EMPTY")},
        llm_user=user_llm, llm_args_user=user_args,
        num_trials=a.trials, max_concurrency=a.concurrency, max_steps=a.max_steps, seed=a.seed,
    )
    Path(a.save_to).parent.mkdir(parents=True, exist_ok=True)
    results = run_tasks(cfg, tasks, save_path=Path(a.save_to), console_display=False)
    n = len(results.simulations); passed = sum(1 for s in results.simulations if s.reward_info and s.reward_info.reward >= 1)
    print(f"{n} simulations, pass rate {passed / max(1, n):.3f}; saved {a.save_to}")


if __name__ == "__main__":
    main()
