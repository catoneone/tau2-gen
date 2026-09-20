# tau2-gen

Generate τ²-shaped agent tasks that are **not** τ²-bench tasks.

[τ²-bench](https://github.com/sierra-research/tau2-bench) is a customer-service benchmark: a policy
document, a tool API over a JSON database, an LLM user simulator, and a scorer that checks the final
database state against a reference trajectory. Its telecom domain is built programmatically, from atomic
device faults composed into multi-fault scenarios.

This repo reuses that machinery — the environment, tools, policy, user simulator and scorer — and
replaces only the part that makes tasks: **the scenarios and the databases they run over**. You get task
sets with the same shape as the benchmark, the same schema, and no overlap with it, so the benchmark
stays usable as a held-out evaluation while the generated set is free to be trained on.

Status: **telecom works** and is verified end to end. Retail and airline are specified but not built.

## Why

Training an agent on a benchmark burns the benchmark. But the states the benchmark exercises are exactly
the ones agents tend to be bad at, and there are only 114 telecom tasks. So: keep the environment, throw
away the tasks, and generate new ones that put the agent in the same kinds of situation.

The situations worth generating are the ones where a model has to ask instead of guess:

- ask for an identifier rather than inventing one
- confirm before a write
- refuse what policy does not allow
- escalate to a human when the issue is genuinely unfixable
- use the value the user actually gave, not a plausible one
- do nothing, when doing nothing is correct

Roughly a fifth of generated tasks have no correct write action at all: the right answer is to explain
and escalate. That ratio is taken from the benchmark rather than chosen.

## Install

τ²-bench requires Python 3.12 or 3.13. Everything here runs through the clone's own interpreter.

```bash
git clone https://github.com/sierra-research/tau2-bench.git upstream/tau2-bench
cd upstream/tau2-bench && uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e . websockets && cd ../..
scripts/patch_tau2.sh
```

Two things that will bite you otherwise:

- `websockets` is an unconditional import of `tau2`'s voice module but lives in an optional extra, so
  plain `pip install -e .` gives you an `ImportError` on `import tau2`.
- `scripts/patch_tau2.sh` applies [`patches/0001`](patches/0001-set_state-keep-separate-user-db.patch).
  Upstream's `Environment.set_state` points `user_tools.db` at `tools.db` whenever a task carries
  `initialization_data.agent_data`. Airline and retail share one database type, so this is harmless
  there. Telecom keeps a separate `TelecomUserDB`, which then gets overwritten by the agent-side
  `TelecomDB` and every user tool breaks. The patch syncs only when both sides are the same type. It is
  needed for the default per-task database mode; `--db-mode shared` works without it.

Set `TAU2_ROOT` if you put the clone somewhere else.

## Quickstart

```bash
P=upstream/tau2-bench/.venv/bin/python

# 1. Extract the shape to match, and the list of tasks to stay clear of.
$P fidelity/bench_profile.py --from-tau2 --split base --out domains/telecom/bench_profile.json

# 2. Generate. Deterministic: same seed, same 200 tasks, about a second.
$P domains/telecom/gen.py --n 200 --seed 0 --out out/telecom

# 3. Check the result against the benchmark's shape.
$P fidelity/report.py --gen-tasks out/telecom/tasks.jsonl --out out/telecom/fidelity_report.md
```

Step 2 writes `tasks.jsonl` (flat episode records), `tasks_tau2.json` (native τ² `Task` objects, loadable
by `tau2 run`), `meta.jsonl`, `manifest.json` and `taskset.toml`. See
[`examples/sample_task.json`](examples/sample_task.json) for one task in full.

## How the telecom generator works

Thirteen atomic fault families (twenty variants): airplane mode, SIM unseated, SIM PIN lock, APN, MMS
APN, line suspension for an overdue bill, contract-end suspension, roaming in three flavours, data
switch, data saver, network preference, VPN, data cap, wifi-calling, and three app-permission variants.

Each fault is an init function, a fix function and assertions, exactly as upstream models them. Faults
compose within selection sets, and composition is constrained per intent so an MMS task always contains
at least one MMS-relevant fault.

What is different here:

**Fresh databases.** Every task gets newly generated customers, phone numbers, ids, IMEIs, plans, bills
and addresses, carried inside the task. Upstream has exactly one customer, `John Smith` on
`555-123-2002` with ids `C1001` / `L1002`, in every telecom task. A model that memorises those solves the
first step of every task without asking. Here it cannot.

**Repair order.** Faults have a rank, so the reference trajectory fixes things in a workable order:
airplane mode before anything radio, SIM before service, suspension and billing before connectivity,
then the data path, then app-level settings, then APN with its reboot.

**Verified by replay.** No task ships unless it survives a replay in a real τ² environment: the fault
injection leaves it broken, it is still broken before each repair step, it is fixed after the last one,
and every assertion passes. Tasks that fail are retried in a different order, then dropped.

**Ten personas.** Upstream's None / Easy / Hard, plus verbose, terse, non-native speaker, impatient,
tech-savvy, gives-a-wrong-number-once, and tacks-on-an-out-of-policy-request. The extended ones change
how the user talks, never what counts as success.

**Varied instructions.** Three phrasings of the reason for calling, three frustration triggers, and a
data top-up amount drawn from 0.5–2 GB. The top-up matters: the environment assertion follows the
number, so an agent that assumes the usual 2 GB fails.

## Not copying the benchmark

Two separate guarantees, both enforced before anything is written to disk.

**Task identity.** Generated tasks exclude every `(intent, fault composition, persona)` triple in the
split you declare held out, and no generated task id may equal a held-out task id. The report also
counts collisions against τ²'s full 2,285-task programmatic enumeration; those are reported but allowed,
since that enumeration is a combination space rather than anyone's evaluation set.

**Entity identity.** Generated phone numbers, ids, emails, IMEIs and names are checked for intersection
with everything in the τ²-bench clone. The intersection must be empty.

`fidelity/leakage.py` runs both. `domains/telecom/gen.py` calls it before writing and exits non-zero
without producing output if either fails.

One more rule, enforced in code rather than by convention: `common/entrypoints.py` refuses to turn
traces into tasks unless every task in them came from this generator. Point it at benchmark traces and
it will only tell you how many decision points it *would* have harvested.

## Checking fidelity

`fidelity/report.py` compares a generated set against a reference set on two levels.

Task level needs nothing but the task file: intent mix, faults per task, expected actions per task,
share of escalate-only tasks, reward basis, assertion kinds. A 200-task run against τ²-bench telecom:

| metric | generated | τ²-bench telecom |
|---|---|---|
| intent mix | 43 / 32 / 26 % | 43 / 32 / 25 % |
| faults per task, median | 3 | 3 |
| expected actions, median | 5 | 4 |
| escalate-only tasks | 18 % | 18 % |
| identifier overlap | 0 | — |
| held-out task-id overlap | 0 | — |

Rollout level needs a reference model run over the generated set, and checks that the tasks are actually
solvable but not trivially so: pass rate inside 0.6–0.85, no category of five or more tasks pinned at 0
or 1, second-message uniqueness at least 0.78 with pairwise Jaccard at most 0.25, and, at k ≥ 2, enough
tasks with two usable and non-identical rollouts to be worth sampling from.

Pass `--bench-traces` to fill the reference column from real traces instead of leaving it blank.

## Reference rollouts

Needs an OpenAI-compatible endpoint for the agent and an API key for the user simulator. The key is read
from `.env`, which is gitignored; it never belongs in a command line or a config file.

```bash
vllm serve <model> --port 8000 --tensor-parallel-size 2 --max-model-len 131072

echo 'OPENAI_API_KEY=sk-...' >> .env

P=upstream/tau2-bench/.venv/bin/python
$P scripts/run_teacher.py --tasks out/telecom/tasks_tau2.json \
   --agent-model <model> --agent-api-base http://127.0.0.1:8000/v1 --agent-temperature 0.7 \
   --user-llm gpt-4.1 --trials 3 --concurrency 8 --save-to out/telecom/runs/teacher_k3.json

$P fidelity/convert.py --results out/telecom/runs/teacher_k3.json --tasks out/telecom/tasks.jsonl \
   --out out/telecom/runs/teacher_k3/traces.jsonl.gz --label teacher-k3
```

Temperature must be above zero or your k trials are k copies. Budget the user simulator, not the agent:
it is the hosted model and it talks on every turn.

With traces in hand you can drop unsolvable tasks, harvest mid-conversation entry points, and run the
full report:

```bash
$P common/solvability.py --tasks out/telecom/tasks.jsonl \
   --teacher-traces out/telecom/runs/teacher_k3/traces.jsonl.gz --out out/telecom/tasks.solvable.jsonl

$P common/entrypoints.py --tasks out/telecom/tasks.solvable.jsonl \
   --traces out/telecom/runs/teacher_k3/traces.jsonl.gz --out out/telecom/tasks.entrypoints.jsonl

$P fidelity/report.py --gen-tasks out/telecom/tasks.solvable.jsonl \
   --gen-traces out/telecom/runs/teacher_k3/traces.jsonl.gz --out out/telecom/fidelity_report.md
```

Entry points are worth the trouble. Cutting a rollout after the identifier is given, after the lookup
chain, after a repair that did not work, after the user gets annoyed, or after a tool call fails turns
one task into several, each starting at a decision rather than at "hello". On τ²-bench's own telecom
traces, 114 rollouts yield 339 such points.

## Layout

| path | what |
|---|---|
| `common/schema.py` | task payload schema, field-for-field with τ²'s `Task`; `schema_diff()` to verify |
| `common/db.py` | database instance generation and identifier extraction |
| `common/user_sim.py` | personas; prompts come from τ²'s own `UserSimulator` |
| `common/solvability.py` | keep tasks somebody solved, drop the rest |
| `common/entrypoints.py` | mid-conversation entry points from rollouts |
| `common/tau2_compat.py` | locating the clone, building an environment over a custom database |
| `domains/telecom/gen.py` | the telecom generator |
| `domains/telecom/bench_profile.json` | shape to match plus the exclusion list (generated; commit it) |
| `fidelity/taxonomy.py` | profile any trace file |
| `fidelity/bench_profile.py` | build the profile, from the clone or from traces |
| `fidelity/report.py` | generated vs reference, with verdicts |
| `fidelity/leakage.py` | the two non-overlap guarantees |
| `fidelity/convert.py` | τ² runner results into the trace schema |
| `scripts/run_teacher.py` | run a reference model over a task set |
| `scripts/patch_tau2.sh` | apply `patches/` to the clone, idempotently |

Generated task sets are gitignored. They are reproducible from a seed, and `manifest.json` records the
commit, the seed, the distributions and the leakage result.

## Not done yet

- **Retail and airline.** Retail is order-lifecycle actions crossed with preconditions. Airline needs the
  policy document turned into a rules table, unit-tested against the benchmark's 50 tasks without those
  tasks entering any generated set.
- **A published reference pass rate.** The task-level and leakage checks pass. Whether a given model
  lands in the 0.6–0.85 band on a generated set is a number you have to produce for your own model.

## License

MIT, see [LICENSE](LICENSE).

Built against τ²-bench, MIT, Copyright (c) 2025 Sierra Research. τ²-bench is not vendored here; it is
cloned at install time and used as a library. The telecom fault definitions follow the structure of its
programmatic task pipeline. See [NOTICE](NOTICE).
