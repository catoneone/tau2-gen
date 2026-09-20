"""Run a reference model over a generated task set with tau2's runner, k trials per task.

The agent is any OpenAI-compatible endpoint (a local vLLM server, for instance). The user simulator is a
hosted model and needs its own API key, read from a gitignored `.env`.

Usage:
  upstream/tau2-bench/.venv/bin/python scripts/run_teacher.py \
      --tasks out/telecom/tasks_tau2.json --agent-model <model> --agent-api-base http://127.0.0.1:8000/v1 \
      --user-llm gpt-4.1 --trials 3 --concurrency 8 --save-to out/telecom/runs/teacher_k3.json

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True, help="tasks_tau2.json (a list of native tau2 Tasks)")
    ap.add_argument("--agent-model", required=True, help="model name as the endpoint expects it")
    ap.add_argument("--agent-api-base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--agent-api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    ap.add_argument("--agent-temperature", type=float, default=0.7, help="must be > 0 for k trials to differ")
    ap.add_argument("--agent-max-tokens", type=int, default=8192)
    ap.add_argument("--user-llm", default="gpt-4.1")
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
    from tau2.domains.telecom.environment import load_tasks
    from tau2.runner import run_tasks

    tasks = load_tasks(a.tasks)
    if a.task_ids:
        tasks = [t for t in tasks if t.id in set(a.task_ids)]
    if a.num_tasks:
        tasks = tasks[: a.num_tasks]
    cfg = TextRunConfig(
        domain="telecom",
        llm_agent=f"openai/{a.agent_model}",
        llm_args_agent={"temperature": a.agent_temperature, "max_tokens": a.agent_max_tokens, "api_base": a.agent_api_base, "api_key": a.agent_api_key},
        llm_user=a.user_llm,
        num_trials=a.trials, max_concurrency=a.concurrency, max_steps=a.max_steps, seed=a.seed,
    )
    Path(a.save_to).parent.mkdir(parents=True, exist_ok=True)
    results = run_tasks(cfg, tasks, save_path=Path(a.save_to), console_display=False)
    n = len(results.simulations); passed = sum(1 for s in results.simulations if s.reward_info and s.reward_info.reward >= 1)
    print(f"{n} simulations, pass rate {passed / max(1, n):.3f}; saved {a.save_to}")


if __name__ == "__main__":
    main()
