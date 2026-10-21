# Engineering notes

Verified against installed packages on **21 Oct 2026**. Everything below was
read from source or exercised in `uv run pytest`, not taken from memory.
Re-verify when bumping a pin.

## Pinned versions

| Package | Version | Source | Notes |
| --- | --- | --- | --- |
| `tau2` (tau2-bench) | 2.0.0 | git `sierra-research/tau2-bench` @ `7233d6a7a7c9b0e0ba551da2105d0b60661aefe5` (main, 21 Oct 2026) | **Not on PyPI.** The PyPI package named `tau2` is an unrelated chemistry library. Python `>=3.12,<3.14`. Pins `litellm>=1.80.15,<1.82.7`. |
| `verifiers` | 0.4.0 (PyPI, released 12 Oct 2026) | `primeintellect-ai/verifiers` | First release whose public API is `verifiers.v1`; the pre-0.4 `MultiTurnEnv` / `ToolEnv` / `Rubric` / `Parser` API no longer exists. Python `>=3.11,<3.15`. |
| `prime-rl` | 0.10.0 (git HEAD 21 Oct 2026) | `PrimeIntellect-ai/prime-rl` | Not a dependency here (pins torch/vLLM/transformers 5.19). Requires `verifiers[harbor]>=0.4.0`, Python `~=3.12`. Configs in `configs/` target it. |
| `trl` | 1.16.0 (PyPI) | | Fallback only; `GRPOConfig` fields referenced in `configs/rl/trl_grpo_lora.yaml` were read from `trl/trainer/grpo_config.py` on main. |
| `unsloth` | 2026.10.7 (PyPI) | | Fallback only. |
| Python | 3.12 (`.python-version`) | | tau2 caps at <3.14. |

## tau2-bench: what the package actually exposes

Import path is `tau2` (`src/tau2` in the repo, installed as a plain package).

### Data directory

The wheel ships **no data**. `tau2.utils.utils` computes `DATA_DIR` at import
time: `$TAU2_DATA_DIR` if set, else `Path(__file__).parents[3] / "data"`,
which in a venv resolves to `.venv/lib/python3.12/data` and does not exist.
`tooluse_rl/__init__.py` therefore calls `ensure_tau2_data_dir()` *before* any
submodule imports tau2, and `uv run tooluse-rl-fetch-data` does a sparse,
blobless checkout of the pinned commit into `data/tau2-bench/` (gitignored):

- `data/tau2/domains/retail/{db.json (2.8 MB), policy.md, tasks.json (114 tasks), split_tasks.json}`
- `data/tau2/domains/airline/{db.json (7 MB), policy.md, tasks.json (50 tasks), split_tasks.json}`
- `data/tau2/multilingual/en/pack.yaml` (the English language pack: user-sim guidelines, agent greeting, personas)

Splits (from `split_tasks.json`): retail `train` 74 / `test` 40 / `base` 114;
airline `train` 30 / `test` 20 / `base` 50. `tests/test_data.py` asserts the
code pin (`uv.lock`, `pyproject.toml`) and the data pin (`tooluse_rl.data`) are
the same commit.

### Retail environment API (verified)

```python
from tau2.registry import registry                     # module-level singleton
env = registry.get_env_constructor("retail")(db=None, solo_mode=False, policy=None, regional_settings=None)
tasks = registry.get_tasks_loader("retail")("train")   # list[tau2.data_model.tasks.Task]; None = all
splits = registry.get_task_splits_loader("retail")()   # {"train": [...ids], "test": [...], "base": [...]}
# equivalently: tau2.domains.retail.environment.get_environment / get_tasks / get_tasks_split
```

- `get_environment(db=...)` deep-copies the DB you pass (`db.model_copy(deep=True)`), otherwise it re-reads `db.json`. We cache `RetailDB.load(RETAIL_DB_PATH)` once and pass it, which is how `tooluse_rl.env.fresh_env` gets a pristine DB per episode at ~0 cost (`tests/test_env.py::test_fresh_env_is_deterministic_and_isolated`).
- `Environment` (`tau2.environment.environment`): `get_tools() -> list[Tool]`, `Tool.openai_schema` (OpenAI function-tool dict), `get_policy() -> str`, `get_response(ToolCall) -> ToolMessage` (errors become `ToolMessage(error=True, content="Error: ...")`, never exceptions), `get_db_hash()`, `set_state(initialization_data, initialization_actions, message_history, strict=True)` (replays a message history; only tools marked `mutates_state` are re-executed and their outputs compared to the recorded ones).
- Retail tools (16): `calculate, cancel_pending_order, exchange_delivered_order_items, find_user_id_by_email, find_user_id_by_name_zip, get_item_details, get_order_details, get_product_details, get_user_details, list_all_product_types, modify_pending_order_address, modify_pending_order_items, modify_pending_order_payment, modify_user_address, return_delivered_order_items, transfer_to_human_agents`. Marked with `@is_tool(ToolType.READ|WRITE|GENERIC, mutates_state=...)` on `tau2.domains.retail.tools.RetailTools`.
- Agent system prompt: `tau2.agent.llm_agent.SYSTEM_PROMPT.format(agent_instruction=AGENT_INSTRUCTION, domain_policy=policy)` (`<instructions>…</instructions><policy>…</policy>`). Opener: `"Hi! How can I help you today?"` (orchestrator default and `pack.yaml:agent_greeting`).
- Messages: `tau2.data_model.message.{SystemMessage, UserMessage, AssistantMessage, ToolMessage, ToolCall}`. `ToolCall(id, name, arguments: dict, requestor="assistant")`. `ToolMessage(id=<tool call id>, role="tool", content, requestor, error, execution_status)`.
- Task: `Task(id, description, user_scenario: UserScenario(persona, instructions: str | StructuredUserInstructions), initial_state, evaluation_criteria: EvaluationCriteria(actions, env_assertions, communicate_info, nl_assertions, reward_basis))`. No retail/airline task sets `initial_state`.
- Orchestrator semantics mirrored in `tooluse_rl.env.episode`: termination `user_stop` (user text contains `###STOP###`, `###TRANSFER###` or `###OUT-OF-SCOPE###`), `max_steps` (default 100 in `Orchestrator`, 200 in the CLI), `too_many_errors` (10 tool errors), `agent_error`. Each generated message, and each batch of tool results, is one step.

### User simulator (verified)

`tau2.user.user_simulator.UserSimulator(llm, instructions, tools, llm_args, persona_config)` drives litellm. Its prompt is reproducible without litellm:

```python
from tau2.user.guidelines import simulation_guidelines
from tau2.user.user_simulator import build_user_system_prompt
persona = PersonaConfig()
guidelines = simulation_guidelines(persona, mode="text", use_tools=False)   # reads multilingual/en/pack.yaml
prompt = build_user_system_prompt(str(task.user_scenario), persona_config=persona, mode="text", guidelines=guidelines)
```

The user model sees the conversation with roles flipped (`UserState.flip_roles`: assistant text → `user`, its own lines → `assistant`, tool traffic hidden). `tooluse_rl.env.users.OpenAIUser` does exactly this over the `openai` SDK so any OpenAI-compatible endpoint can play the customer (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `USER_SIM_MODEL`). For leaderboard-comparable runs use tau2's own CLI (`tau2 run --domain retail --user-llm openai/<model>`).

### Evaluator and reward basis (verified; this matters)

`tau2.evaluator.evaluator.evaluate_simulation(simulation: SimulationRun, task, evaluation_type, solo_mode, domain, *, communicate_scoring, strict_replay=True, env_kwargs=None) -> RewardInfo`.

- `EvaluationType.ENV`: replay the trajectory's mutating tool calls on a fresh env (`set_state`), replay the task's gold `actions` on another, compare `get_db_hash()`; plus `env_assertions`. Deterministic, offline, ~0.3 s.
- `EvaluationType.ALL`: ENV × ACTION (diagnostic unless `ACTION` in basis) × COMMUNICATE (if in basis) × NL_ASSERTION (if in basis). Final reward is the **product** of the components in `reward_basis`.
- **tau2 2.0 changed the retail basis**: 112/114 retail tasks have `reward_basis = [DB, NL_ASSERTION]` (40 of them have non-empty `nl_assertions`); 2 are `[DB]`. NL assertions are judged by an LLM (`gpt-5.4-mini-2026-03-17` by default) — paid and non-deterministic. Airline is `[DB, COMMUNICATE]` for all 50 tasks; `communicate_scoring` defaults to a semantic LLM judge, `CommunicateScoring(scorer=CommunicateScorer.SUBSTRING)` restores the offline 1.x substring check. `communicate_info` is present on 36 retail tasks but not in their basis.
- **Decision:** the training/dev reward uses `ENV` for tasks whose basis includes `NL_ASSERTION` (all retail) and `ALL` with substring communicate otherwise (airline). So `task_success` here = *DB end state matches gold* (+ env assertions / substring communicate). It is a strict superset of the tasks the official metric passes, and judge-free. README numbers must be produced with the official `tau2 run` CLI (judge on) for comparability; `tooluse-rl-eval --with-nl-judge` runs `ALL` if you have a key.
- A simulation that did not end in `user_stop`/`agent_stop` scores 0 without replay. Success = `reward == 1` (`tau2.metrics.agent_metrics.is_successful`). `pass^k = C(successes, k) / C(trials, k)` per task, averaged over tasks (`pass_hat_k`).
- **Measured floor (21 Oct 2026, this runner, DB-only scorer, user hangs up at once, 1 trial):** a policy that never acts scores pass^1 = 0.100 on retail test (4/40), 0.095 on retail train (7/74) and **0.460 on airline (23/50)**. Those are tasks whose gold action list is empty (the correct outcome is to refuse) and whose `communicate_info` is empty or trivially absent. Any airline result must be read against that floor; the official judge-on metric has the same property for its DB component.
- Replay is strict by default: a recorded tool output that differs from a re-execution raises. Our fixtures are recorded from the real env, so they replay cleanly; `strict_replay=False` exists for re-grading old trajectories.

## verifiers 0.4.0 (`verifiers.v1`) — what the adapter uses

`import verifiers.v1 as vf`. The stack: `Taskset.load()` yields `Task(TaskData, TaskConfig)`; an `Env` orchestrates `Agents` (each = harness × model × runtime) for one task into an `Episode` of `Trace`s; scoring is `@vf.reward` / `@vf.metric` methods on the Task reading the `Trace`; task tools are `vf.Toolset` subclasses served as **MCP servers**, launched per rollout as `python -m <toolset module>` with `setup_task(task_data)`.

- Plugin resolution (`verifiers/v1/utils/loaders.py`): taskset id `foo-bar` → `importlib.import_module("foo_bar")` (after trying `verifiers.v1.tasksets.foo_bar`) → the module's `__all__` must export exactly one `Taskset` subclass, and optionally one `Env` subclass (used automatically). Hence `tooluse_rl/__init__.py` exports `Tau2Taskset` and `Tau2UserSimEnv` lazily (PEP 562 `__getattr__`), and the taskset id is `tooluse-rl`.
- Built-in `user-sim` env (`verifiers.v1.envs.user_sim.UserSimEnv`): two agents, `assistant` and `user`; the task's `prompt_text` becomes the user's scenario (spliced into `persona` at `{scenario}`), the assistant plays the task with `prompt=None` and only `system_prompt`; the user opens after the assistant's greeting `"Hello! How can I help you today?"` and ends with `done_marker`. `Tau2UserSimEnvConfig` sets `persona` to tau2's user-sim prompt, `done_marker="###STOP###"`, both harnesses `null`, `user.max_turns=30`. The user is marked `trainable=False` by the base class.
- `Trace.messages` (final branch), `trace.assistant_messages`, `trace.tool_messages`, `trace.info` (persisted), `trace.record_metric/record_reward`. Typed messages: `vf.UserMessage(content)`, `vf.AssistantMessage(content, tool_calls=[vf.ToolCall(id, name, arguments: str)])`, `vf.ToolMessage(tool_call_id, content)`; `verifiers.v1.types.content_text` flattens multimodal content. `tooluse_rl.vf.convert` maps these to tau2 messages for the evaluator (tool errors recognised by tau2's `Error:` prefix).
- Toolsets: `@vf.tool` methods are discovered from the class dict (`discover_decorated` walks the MRO's `vars()`), so `tooluse_rl.vf.toolset.make_toolset_class` builds one class per domain with a method per tau2 tool (signature/annotations/docstring copied via `__signature__`); `Toolset.register(mcp)` → `MCPServer.add_tool`. Verified offline: 16 tools, correct required params and array types (`tests/test_vf_adapter.py`). Stateful per-rollout toolsets take `vf.ToolsetConfig` (shared ones `vf.SharedToolsetConfig`); `Task.toolsets(cls, config)` is a classmethod, so the domain lives on `Tau2TaskConfig.domain`.
- `@vf.reward(weight=…)` weights are static decorator arguments; `trace.reward` is the weighted sum. We expose `task_success` (weight 1) and `format_score` (weight 0.1).
- CLI: `uv run vf-eval tooluse-rl --dry-run`, `uv run vf-eval @ cfg.toml`; TOML keys `[env.taskset] id/domain/split`, `[env.assistant.*]`, `[env.user.*]`, `[client] base_url`, `[sampling]`, `[select] limit/shuffle`.

**Not verified** (needs a model and an MCP transport, i.e. money): an actual `vf-eval` rollout through `Tau2UserSimEnv`. The pieces exercised offline are taskset loading, tool registration/schemas, per-task env reset in the toolset, plugin resolution, config defaults and trace-to-reward conversion.

## prime-rl 0.10 config shape (from `examples/basic/wordle`, `configs/debug/algo/grpo.toml`, `docs/algorithms.md`)

TOML via pydantic-config; `uv run rl @ file.toml`, `uv run sft @ file.toml`, `uv run eval @ file.toml`, `uv run inference --vllm.model …`. Key tables: `[model] name`, `[orchestrator] batch_size, max_off_policy_steps`, `[orchestrator.train] group_size`, `[orchestrator.train.algo] type = "grpo"`, `[orchestrator.train.sampling] max_completion_tokens`, `[[orchestrator.train.source]] env.taskset.id / env.<role>.harness.id / env.<role>.model …`, `[orchestrator.eval] interval` + `[[orchestrator.eval.source]]`, `[trainer.model.lora] rank/alpha/target_modules`, `[trainer.optim] lr`, `[inference.vllm]`, `[deployment] num_train_gpus/num_infer_gpus`, `[orchestrator.renderer] name = "prime-qwen3"`. prime-rl's `grpo` has **no reference-KL term** (only `opd`/`opsd` use `ref_kl`); `mismatch_kl/*` is a monitoring metric. Role names in TOML are the `EnvConfig` field names (`assistant`, `user` for the user-sim env). SFT: `[data] name/seq_len/batch_size/micro_batch_size`, `[optim] lr`, `max_steps`.

## TRL `GRPOConfig` knobs referenced (trl main, Oct 2026)

`num_generations` (group size), `beta` (KL coefficient; default 0.0), `epsilon`/`epsilon_high` (clip), `loss_type` (`grpo`/`dapo`/`bnpo`/…), `scale_rewards` (`group`/`batch`/`none`), `max_completion_length`, `temperature`, `mask_truncated_completions`, `use_vllm` + `vllm_mode` (`server`/`colocate`), `max_tool_calling_iterations` (single-agent tool loops only; the tau2 user-sim loop still needs a custom rollout).

## Trajectory JSONL schema (`tooluse-rl/trajectory/v1`)

`~/code/open-harness` did not exist when this repo was scaffolded (21 Oct 2026), so this is a stand-alone definition; align it with that repo's logger when it lands. One JSON object per line:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | str | `"tooluse-rl/trajectory/v1"` |
| `id`, `run_id` | str | episode id; run id shared by all episodes of one eval |
| `created_at` | ISO-8601 UTC | |
| `benchmark`, `domain`, `task_id`, `trial`, `seed` | | `tau2-bench`, `retail`/`airline`, tau2 task id, 0-based trial |
| `policy_model`, `user_model` | str | endpoint-qualified names (`openai:<model>`, `mock-policy`, …) |
| `system_prompt` | str | the policy's system prompt (not repeated in `messages`) |
| `messages` | list | OpenAI chat shapes: `{role, content, turn_idx}`; assistant may carry `tool_calls: [{id, name, arguments: object}]`; tool results are `{role: "tool", tool_call_id, content, error: bool}` |
| `termination_reason` | str | tau2 `TerminationReason` value |
| `steps`, `tool_errors`, `final_db_hash` | int, int, str | |
| `reward` | object | `{total, task_success, format_score, shaping_weight, db_match, termination_reason, evaluation_type, reward_basis, signals{…}, tau2_info}` |
| `usage` | object | `{policy: {prompt_tokens, completion_tokens}, user: {...}}` |
| `cost_usd`, `duration_s` | float | cost is null until a price table is wired in |

`tooluse_rl.trajectory.record_to_episode` rebuilds an episode from a record so any file in this format can be re-graded by tau2's evaluator.

## Decisions log

1. **Data pinned by commit, fetched, not vendored.** 10 MB of benchmark JSON plus the language pack would bloat the repo and drift; the fetch script and `uv.lock` share one SHA, enforced by a test.
2. **Own episode loop instead of tau2's `Orchestrator`.** The orchestrator hard-wires litellm agents/users; a 150-line loop with `Policy`/`User` protocols gives mocks, any OpenAI-compatible endpoint, and the same message objects tau2's evaluator consumes. Fixtures prove the evaluator accepts them.
3. **Reward = DB check (+ offline communicate) + 0.1 × format.** The LLM-judged NL assertions tau2 2.0 added to retail are not an RL reward (cost, noise); they are an eval-time flag. Shaping is bounded so it can never outrank success.
4. **verifiers v1 adapter shaped after the built-in `user-sim` env** rather than a bespoke `Env.run`, because prime-rl 0.10 is built on exactly these abstractions and the tau2 conversation protocol matches its loop (assistant greets, user opens, user hangs up).
5. **Tool server methods are generated from tau2's toolkit** so names/params are byte-identical to the benchmark; the evaluator's replay keys on tool names.
6. **prime-rl is not a dependency** of this package (it pins torch/vLLM/transformers); configs live here, the trainer is installed on the GPU box and `uv pip install -e`s this repo.

## Open questions

- Which user-sim model for RL: the budget line ($50–200) assumes a mini-class API model; a local 8B user on the same vLLM is free but weaker and may be gamed.
- Whether to keep `format_score` at all once the SFT warm start fixes formatting (risk: shaping rewards verbosity-free "clean" failures).
- How the official leaderboard handles the 2.0 NL judge for retail; our README table should state which scorer produced each number.
- The airline floor (46% for a do-nothing policy) means "+W on held-out airline" needs to be reported as a delta over the base model *and* over the floor; consider also reporting pass^1 on the 27 airline tasks that require a write.
