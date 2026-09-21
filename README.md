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

Status: **all three domains work** — telecom, airline and retail — each verified end to end.

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

## The three domains

They are not variations on one generator. tau2-bench builds telecom programmatically from atomic device
faults, and writes airline and retail by hand, so only telecom had a pipeline to adapt. The other two are
derived from their policy documents instead. They are also scored differently, which decides what can be
checked offline:

| | how tasks are made here | `reward_basis` | offline verification |
|---|---|---|---|
| telecom | fault table composed with repair-order dependencies, following upstream's own pipeline | `ENV_ASSERTION` | replay, then assert device state |
| airline | policy compiled into `rules.yaml`, cases built to land on one side of one rule | `DB, COMMUNICATE` | replay, then compare database hashes |
| retail | order lifecycle crossed with preconditions from the policy | `DB, NL_ASSERTION` | replay, then compare database hashes |

Retail's basis contains an LLM-judged component, but tau2 only invokes the judge when the assertion list
is non-empty, and 74 of the benchmark's own 114 retail tasks carry none. Generated retail tasks leave it
empty and keep the sentences as notes, so the database check is what gates and nothing here needs an API
key. `--judge` puts the assertions back for anyone who wants them graded.

## What the agent must say

A task whose correct answer is to change nothing is passed, under a database check alone, by any agent
that changes nothing — including one that says nothing, or invents a reason. That is a third of the
airline set rewarding silence as much as a correct refusal.

So each task carries one short must-mention string, matched case-insensitively as a substring. Choosing
it is the whole problem: too specific and a correct agent fails on phrasing rather than on substance.
Measured over the 36 airline tasks tau2-bench's own teacher passes:

| candidate must-mention string | recall on passing trajectories |
|---|---|
| a topic word from the user's own request | 35/36 (97 %) |
| "basic economy", where that rule is the blocker | 11/12 (92 %) |
| the reservation id the agent acted on | 35/48 (73 %) |
| "insurance", where cancellation is not eligible | 13/18 (72 %) |
| "human agent", where a segment has been flown | 3/5 (60 %) |

The requirement is therefore anchored on the subject the user raised, not on the reason the agent has to
give. Silence cannot satisfy it, and a correct refusal is not punished for its wording. The
reason-anchored strings were measured and rejected: at 60 to 73 % they would fail a correct agent
between a quarter and two-fifths of the time, which is worse than a vacuous check. The one exception is
"basic economy", which is well enough attested to be required on top of the topic anchor.

That an anchor really is a word the user used is checked when the task is built, not assumed, so
scenario wording that drifts away from its anchor fails loudly instead of quietly producing tasks no
correct agent can pass.

Airline scores on `[DB, COMMUNICATE]`, so its anchors gate. Retail's benchmark basis is
`[DB, NL_ASSERTION]`, under which `communicate_info` is recorded but does not gate, exactly as in the
benchmark's own 36 retail tasks that carry it. `--gate-communicate` switches retail to
`[DB, COMMUNICATE]` and makes them gate, at the price of departing from the reference basis.

### Airline: the policy as a rule table

[`domains/airline/rules.yaml`](domains/airline/rules.yaml) encodes each clause of the policy document
with a citation, and [`rules.py`](domains/airline/rules.py) applies it to a concrete reservation. Every
decision carries the reason it was reached, which becomes the task's natural-language assertion.

The table is held to the benchmark's own 50 tasks by
[`test_rules.py`](domains/airline/test_rules.py), which checks the direction the policy states
unconditionally. Every reservation a benchmark task cancels must be eligible; every flight change must be
permitted; every certificate amount must match; and a reservation with a flown segment must always be
refused. The converse is not asserted, because a task may leave an eligible reservation alone simply
because the user asked about a different one.

```
airline rules vs 50 benchmark tasks and 200 flown reservations
  baggage_nonfree:pass               5
  cabin_change_eligible:pass        17
  cancel_eligible:pass              11
  flight_change_eligible:pass        3
  flown_blocks_cancel:pass         200
  flown_blocks_change:pass         200

all checks passed
```

Those 50 tasks are fixtures. They are read to test the table and never written into a generated set.

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

# 2. Generate. Deterministic: same seed, same tasks.
$P domains/telecom/gen.py --n 200 --seed 0 --out out/telecom
$P domains/airline/gen.py --n 150 --seed 0 --out out/airline
$P domains/retail/gen.py  --n 150 --seed 0 --out out/retail

# 3. Check each against the benchmark's shape.
$P fidelity/report.py --gen-tasks out/telecom/tasks.jsonl --domain telecom --out out/telecom/fidelity_report.md
$P fidelity/report.py --gen-tasks out/airline/tasks.jsonl --domain airline --bench-tasks --out out/airline/fidelity_report.md
$P fidelity/report.py --gen-tasks out/retail/tasks.jsonl  --domain retail  --bench-tasks --out out/retail/fidelity_report.md

# 4. Hold the airline rule table to the benchmark's own 50 tasks.
$P domains/airline/test_rules.py
```

`--bench-tasks` reads tau2-bench's own task file as the reference column, which is what airline and
retail ship instead of traces.

Step 2 writes `tasks.jsonl` (flat episode records), `tasks_tau2.json` (native τ² `Task` objects, loadable
by `tau2 run`), `meta.jsonl`, `manifest.json` and `taskset.toml`. See
`examples/` for one task from each domain in full: [telecom](examples/sample_task.json),
[airline](examples/sample_airline_task.json), [retail](examples/sample_retail_task.json).

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

## User behaviour is composed, not templated

`task_instructions` is what steers the user simulator: how insistent the user is, when they volunteer
information, how they react to being turned down. One fixed string per case makes every rollout of that
case look alike, which is exactly what a diversity gate would catch. So it is assembled per task from
independent pools in [`common/user_sim.py`](common/user_sim.py) — confirmation style, disclosure pace,
reaction to refusal, tone — shuffled and combined.

Every clause changes only *how* the user behaves, never what a correct outcome is, so the expected
actions and the target database state are untouched. Anything that could redirect the agent to a
different action, such as asking for a cheaper alternative or changing their mind about what they want,
is deliberately absent: that belongs in a case of its own, where the expected actions can follow.

Airline and retail also carry composite cases, which is how the benchmark's own scenarios look. Two
rules in one conversation (upgrade the cabin and then add bags, where the free allowance follows the
*new* cabin), two entities in one conversation (cancel one order and return an item from another), and
scenarios where the user cannot name the record at all and the agent has to search the profile for it.

Two further properties cut across every case rather than belonging to any one of them, so they are
applied by the generator rather than written into cases:

**Account history.** Every user gets a few extra records beyond the one the task is about. It costs
nothing in the reference trajectory, and it stops every profile from looking identical.

**The unknown-id axis.** Real customers often cannot quote the record number. Any single-record case can
have its identifier removed from the scenario and replaced with a description of the record, at which
point the correct trajectory really does read through the profile to find it. The share is a knob,
`--unknown-id-share`. A task never claims the user has forgotten something the text still spells out:
the substitution is checked, and is abandoned if the id cannot be removed cleanly.

**Contingencies, derived from the rule engine.** What fills the reference set's airline instructions is
not varied phrasing of generic behaviour but pressure aimed at the exact rule the task turns on: *if the
agent tells you cancellation is not possible, mention that you were told you didn't need insurance*. The
rule engine already names that rule, since every decision carries its reason, so the contingency is
looked up rather than written per case, and it stays correct when the rules change. Each one ends by
closing the loop, with the user declining the alternative they floated: a user who suggests "could we
cancel and rebook instead" and then agrees has authorised a different outcome, and the expected actions
would no longer describe a correct trajectory.

**How much generic behaviour to append is per-domain.** The reference sets differ by a factor of six in
how much they write here: nine words at the median for retail, fifty-eight for airline. A generic tail
longer than the task itself both pads the prompt and drives up overlap between tasks, so retail gets one
clause and airline two, on top of a contingency that carries the task-specific part.

Case mixes are allocated by quota rather than drawn independently, so `--refuse-share` and the rest are
settings rather than suggestions. Drawn independently at n=150, any one group moved by about eight
points between runs.

Measured against the benchmark's own task files:

| | airline gen | airline ref | retail gen | retail ref |
|---|---|---|---|---|
| distinct `task_instructions` | 0.98 | 0.98 | 0.83 | 0.61 |
| pairwise Jaccard on those | 0.22 | 0.15 | 0.17 | 0.09 |
| words in `task_instructions`, median | 43 | 58 | 29 | 9 |
| tasks touching ≥2 records | 21 % | 18 % | 61 % | 56 % |
| records per task, median | 1 | 1 | 2 | 2 |
| tasks with no correct write | 49 % | 48 % | 8 % | 9 % |

One gap remains and is worth knowing about: lexical overlap on the instructions is still above the
reference. Closing it further means either writing a contingency per task, which is what a hand-written
set of fifty is, or paraphrasing the pools. Paraphrasing would have to be a build step that freezes a
reviewed pool into the repository, not something the generator does at run time, or the guarantee that a
clause cannot change what counts as correct stops being checkable and the toolchain stops working
without an API key.

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

Task level needs nothing but the task file: intent mix, faults per task, expected and write actions per
task, share of tasks with no correct write, reward basis, assertion kinds.

Telecom, 200 generated against the benchmark's 114:

| metric | generated | τ²-bench telecom |
|---|---|---|
| intent mix | 43 / 32 / 26 % | 43 / 32 / 25 % |
| faults per task, median | 3 | 3 |
| expected actions, median | 5 | 4 |
| escalate-only tasks | 18 % | 18 % |
| identifier overlap | 0 | — |
| held-out task-id overlap | 0 | — |

Airline and retail, 150 generated each:

| metric | airline gen | airline ref | retail gen | retail ref |
|---|---|---|---|---|
| expected actions, median | 4 | 2 | 4 | 5 |
| write actions, median | 1 | 1 | 1 | 1 |
| tasks with no correct write | 39 % | 48 % | 10 % | 9 % |
| `reward_basis` | matches | — | matches | — |
| identifier overlap | 0 | — | 0 | — |
| held-out task-id overlap | 0 | — | 0 | — |

The share of tasks whose correct answer is to do nothing is a knob, `--refuse-share`, defaulting to the
reference set's own share. Raise it when the point is training data for refusal rather than a
distribution match. The one gap left is the tail: the benchmark has tasks with up to five writes, where
generated tasks stop at two.

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
| `domains/airline/rules.yaml` | the airline policy as a rule table, each entry citing its clause |
| `domains/airline/rules.py` | the rule engine; every decision carries its reason |
| `domains/airline/test_rules.py` | holds the table to the benchmark's 50 tasks |
| `domains/airline/db.py`, `gen.py` | airline database deltas and task generator |
| `domains/retail/db.py`, `gen.py` | retail catalogue, orders and task generator |
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

- **A published reference pass rate.** Task-level checks, leakage and replay verification pass for all
  three domains. Whether a given model lands in the 0.6–0.85 band on a generated set is a number you have
  to produce for your own model, with `scripts/run_teacher.py`.
- **Long multi-write tasks.** Generated tasks top out at two writes; the benchmark reaches five.
- **Mid-conversation entry points beyond telecom.** `common/entrypoints.py` is domain-agnostic but has
  only been exercised on telecom traces.

## License

MIT, see [LICENSE](LICENSE).

Built against τ²-bench, MIT, Copyright (c) 2025 Sierra Research. τ²-bench is not vendored here; it is
cloned at install time and used as a library. The telecom fault definitions follow the structure of its
programmatic task pipeline. See [NOTICE](NOTICE).
