# tooluse-rl

**RL post-training (GRPO) of a small open model on [tau2-bench](https://github.com/sierra-research/tau2-bench) retail, with the airline domain held out.**

Headline this repo is built to earn, with the numbers still blank:

> GRPO on an 8B model raised tau2-bench retail pass^1 from **X%** to **Y%** for **$Z** of compute, with **+W** on the held-out airline domain.

Status: **skeleton** (Oct 2026). Environment, reward, baseline evaluator, verifiers/prime-rl adapter and config stubs are in place and tested offline with mocks. No training run has happened yet.

## Results

Retail = the 40-task `test` split (never trained on). Airline = all 50 tasks (never trained on). pass^k is τ-bench's metric: the chance that *k* independent trials of a task all succeed, averaged over tasks, from 4 trials. Numbers come from the official `tau2 run` CLI unless marked otherwise.

| Model | Retail pass^1 | Retail pass^4 | Airline pass^1 (held-out) | Airline pass^4 | Compute |
| --- | --- | --- | --- | --- | --- |
| Null policy (never acts; this repo's DB-only scorer, 1 trial) | 0.100 | — | 0.460 | — | $0 |
| Qwen3-8B (base) | — | — | — | — | $0 (eval only) |
| + SFT warm start (rejection-sampled) | — | — | — | — | — |
| + GRPO (this repo) | — | — | — | — | — |
| Reference: frontier model A | — | — | — | — | API only |
| Reference: frontier model B | — | — | — | — | API only |

The null-policy row is the floor, not a bug: 4/40 retail-test and 23/50 airline tasks are solved by correctly *refusing* to change anything (the user asks for something the policy forbids), and the offline scorer only checks the database end state plus airline's required phrases. Report every airline number next to that floor.

### Compute budget (from the plan)

| Item | Budget |
| --- | --- |
| GPU time (1–2 × H100 on RunPod / Prime Intellect / Lambda) | $300 – 1,500 |
| Simulated-user API calls | $50 – 200 |
| Baseline + eval API calls (frontier references) | within the above |

## What is here

```
src/tooluse_rl/
  data.py            pinned, sparse fetch of the tau2-bench data; exports TAU2_DATA_DIR
  env/
    domain.py        deterministic env per episode (cached DB, deep copy), tasks, tool schemas, policy prompt
    users.py         User protocol; OpenAIUser = tau2's user simulator over any OpenAI-compatible endpoint; MockUser
    policies.py      Policy protocol; OpenAIPolicy (native tool calling); MockPolicy / NullPolicy
    episode.py       half-duplex episode loop mirroring tau2's orchestrator (termination reasons, step counting)
    reward.py        task success via tau2's replay evaluator + bounded format shaping
    metrics.py       pass^k (tau2's implementation), run summaries, Markdown tables
  trajectory.py      JSONL trajectory records (schema in docs/notes.md), re-gradable
  vf/                verifiers 0.4 (v1) adapter: Tau2Taskset, per-rollout MCP tool server, Tau2UserSimEnv
  eval/baseline.py   `tooluse-rl-eval`: N tasks x T trials -> pass^1 / pass^4 table (+ JSONL)
  eval/mock_scripts.py  five hand-written episodes on real retail tasks (the fixtures' source)
configs/             prime-rl SFT / GRPO / eval TOML stubs, TRL + Unsloth LoRA fallback (commented knobs)
tests/               offline test suite; tests/fixtures/*.json are recorded episodes
docs/plan.md         the Project B plan; docs/notes.md verified APIs, decisions, open questions
```

## How to reproduce

### 0. Setup (no GPU, no API key)

```bash
uv sync                          # Python 3.12; installs tau2 (git-pinned) and verifiers 0.4
uv run tooluse-rl-fetch-data     # ~45 MB sparse checkout of the pinned tau2-bench data into data/
uv run pytest                    # offline suite incl. recorded-episode regrading
uv run tooluse-rl-eval --mock    # end-to-end on real retail tasks with a scripted policy + user
```

The mock run prints (deterministic, so pass^4 = pass^1):

| script | task | tasks | trials | pass^1 | pass^4 | avg reward | avg steps | tool errors/ep | policy tokens/ep |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cancel_single_order_success | retail/88 | 1 | 4 | 1.000 | 1.000 | 1.100 | 17.0 | 0.000 | 0.000 |
| cancel_wrong_reason_fails | retail/88 | 1 | 4 | 0.000 | 0.000 | 0.100 | 17.0 | 0.000 | 0.000 |
| hallucinated_tool_then_recovers | retail/88 | 1 | 4 | 1.000 | 1.000 | 1.075 | 19.0 | 1.000 | 0.000 |
| no_change_task_success | retail/24 | 1 | 4 | 1.000 | 1.000 | 1.100 | 15.0 | 0.000 | 0.000 |
| runs_out_of_steps | retail/88 | 1 | 4 | 0.000 | 0.000 | 0.075 | 12.0 | 0.000 | 0.000 |
| ALL (mock) | — | 5 | 4 | 0.600 | 0.600 | 0.690 | 16.0 | 0.200 | 0.000 |

### 1. Baseline: base model served by vLLM, API model as the user

```bash
vllm serve Qwen/Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes --port 8000
cp .env.example .env   # OPENAI_API_KEY, USER_SIM_MODEL=gpt-4.1-mini, POLICY_MODEL=Qwen/Qwen3-8B, POLICY_BASE_URL=http://localhost:8000/v1
uv run tooluse-rl-eval --domain retail  --split test --n-tasks 40 --trials 4 --out outputs/base-retail.jsonl
uv run tooluse-rl-eval --domain airline --split base --n-tasks 0  --trials 4 --out outputs/base-airline.jsonl
```

For the leaderboard-comparable number (tau2 2.0 adds an LLM-judged NL-assertion component to retail; see `docs/notes.md`), run the same model through tau2's own CLI: `tau2 run --domain retail --agent-llm openai/Qwen/Qwen3-8B --user-llm gpt-4.1-mini --num-trials 4`.

### 2. SFT warm start (optional)

Rejection-sample successful trajectories from a stronger model with `tooluse-rl-eval --out`, keep `reward.task_success == 1`, push as a chat dataset, then `uv run sft @ configs/sft/prime_rl_sft.toml` from a prime-rl checkout with this repo installed (`uv pip install -e /path/to/tooluse-rl`).

### 3. GRPO

```bash
# in a prime-rl checkout, 2 x H100 (1 trainer + 1 vLLM)
uv pip install -e /path/to/tooluse-rl && uv run tooluse-rl-fetch-data
uv run vf-eval tooluse-rl --dry-run                 # the taskset + user-sim env resolve
uv run rl @ /path/to/tooluse-rl/configs/rl/prime_rl_grpo.toml
```

Single GPU: `configs/rl/trl_grpo_lora.yaml` (TRL GRPOTrainer + Unsloth LoRA; needs a custom multi-turn rollout, see the file).

### 4. Held-out eval

`configs/eval/prime_rl_eval.toml` evaluates a served checkpoint on retail-test and all of airline through the same env; `tooluse-rl-eval --domain airline` does it with this repo's runner.

## Design notes (short version; details in `docs/notes.md`)

- **Deterministic reset.** Each episode gets a deep copy of a once-loaded domain DB; the task's `initial_state` is applied like tau2's orchestrator does.
- **Reward.** `task_success` is tau2's verdict: replay the agent's writes, compare the DB hash with the gold-action DB (airline also checks the substring `communicate_info`). The LLM-judged NL assertions tau2 2.0 added to retail are off by default (cost, noise) and available as `--with-nl-judge` for eval. `format_score` (no hallucinated tools, no mixed text+tool turns, clean hang-up, under the step budget) is added with weight 0.1, so a tidy failure never beats a success.
- **Simulated user.** tau2's exact user-sim prompt and role flipping, over the `openai` SDK: `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `USER_SIM_MODEL`.
- **verifiers / prime-rl.** The taskset id `tooluse-rl` resolves to this package; the assistant uses the `null` harness with the domain tools served over MCP per rollout; the user is verifiers' built-in user-sim agent with tau2's prompt and `###STOP###`.

## License

MIT. tau2-bench (Sierra Research) is MIT-licensed and fetched, not vendored.
