# Configs

Stubs, not tuned values. Every knob that matters has a comment saying why.

| File | Stack | Purpose |
| --- | --- | --- |
| `sft/prime_rl_sft.toml` | prime-rl `uv run sft` | One-epoch SFT warm start on rejection-sampled retail trajectories |
| `rl/prime_rl_grpo.toml` | prime-rl `uv run rl` | GRPO against the `tooluse-rl` verifiers env (first choice, 1–2 H100) |
| `rl/trl_grpo_lora.yaml` | TRL `GRPOTrainer` + Unsloth LoRA | Single-GPU fallback |
| `eval/prime_rl_eval.toml` | prime-rl `uv run eval` | Retail test + held-out airline through the same env |

prime-rl is installed separately (it is not a dependency of this package; it
pins torch/vLLM). From a prime-rl checkout:

```bash
uv pip install -e /path/to/tooluse-rl          # makes the `tooluse-rl` taskset importable
uv run tooluse-rl-fetch-data                    # or set TAU2_DATA_DIR
uv run rl @ /path/to/tooluse-rl/configs/rl/prime_rl_grpo.toml
```
