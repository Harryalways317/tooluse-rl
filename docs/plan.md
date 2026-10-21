# Project B plan (copied from `~/code/resume/plan-top2.md`, 21 Oct 2026)

This is the Project B section of the two-project plan, copied verbatim so the repo is self-contained. The shared-code note from the parent plan: trajectory format, eval table generator and cost accounting are meant to be shared with the Terminal-Bench harness repo (`open-harness`); see `docs/notes.md` for the current state of that.

## Project B — Tool-Use RL: post-train a small open model with RL on tau2-bench

**Headline you're building toward:** "GRPO on an 8B model raised tau2-bench retail pass^1 from X% to Y% for $Z of compute, with +W on the held-out airline domain."

**Why it's credible:** "I trained an agent with RL and it got measurably better on a known benchmark" is the strongest single line a backend engineer can add in 2026. tau2-bench is the standard tool-use/customer-service benchmark and its environments are pure API simulation, so no sandboxes per rollout.

**Stack:** Python; **verifiers + prime-rl** (Prime Intellect) as the first choice because tool-use environments already exist on their Environments Hub, with **TRL GRPO + Unsloth** as the single-GPU LoRA fallback; vLLM for rollouts; a cheap API model as the simulated user; Weights & Biases for curves; rented H100s (RunPod, Prime Intellect, Lambda).

**Components:**
1. Environment: tau2-bench retail (tools + policy document + simulated user). Deterministic DB reset per episode. Reward = task success plus light format shaping.
2. Baselines: the base model (Qwen3-8B class) and two frontier models through the same harness. Record pass^1 and pass^4.
3. SFT warm start: rejection-sample successful trajectories from a stronger model, one epoch of SFT. Optional but it saves RL steps.
4. RL: GRPO, group size 8–16, KL penalty, 200–500 steps. Expect reward hacking (early "done", gaming the simulated user) and budget a week for it.
5. Eval: held-out airline domain for generalisation, re-run the baseline harness, tokens and cost per episode.

**Six-week plan:**
- **W1:** run tau2-bench with the base model served by vLLM on one rented GPU; get the baseline number. Run the verifiers tool-use example end to end.
- **W2:** wrap tau2 retail as a verifiers environment (or adopt an existing one); write the reward function and tests for it; build the rejection-sampled SFT set.
- **W3:** SFT, then first GRPO runs with LoRA on 1–2 H100s. Fix the reward hacking you find.
- **W4–6:** scale runs; ablate SFT-vs-none and group size; held-out eval; push the model to Hugging Face; write-up.

**Read first:** DeepSeek-R1 paper (GRPO as used in practice); Tülu 3 post-training report; tau-bench and tau2-bench papers; verifiers docs and Environments Hub; Unsloth GRPO notebooks; OpenPipe's ART write-ups on RL for tool use.

**Budget:** $300–1,500 GPU, $50–200 simulated-user API.
**Done means:** public repo with configs and training curves, model on Hugging Face, README table, blog post.

---
