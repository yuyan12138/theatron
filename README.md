# THEATRON

Code and raw main-experiment trajectories for THEATRON, a benchmark for persona-conditioned sequential decision behavior.

## Paper experiments

The main experiment evaluates four models (Qwen3-8B, Qwen3-14B, GPT-5.4-mini, and Gemini-2.5-Flash), two stocks (AAPL and TSLA), three personas (`overconfidence`, `loss_aversion`, and `herding`), and three injection methods (Zero-Shot, One-Shot, and CoT). Each condition has five rollouts: run IDs 1–5 correspond to seeds 41–45. Memory is disabled in the paper experiments.

One-Shot uses a shared mixed-evidence demonstration with persona-specific decisions. In `matched_one_shot`, `M` denotes this primary mixed-evidence scenario; `P` and `N` denote positive- and negative-evidence demonstration ablations.

## Repository layout

| Location | Contents |
|---|---|
| `src/` | Main task interface, prompt builder, model clients, and trajectory engine |
| `data/` | Original AAPL and TSLA input streams |
| `outputs/trajectories/` | 360 raw main-experiment CSV files: 120 per injection method |
| `persona_free_baseline/src/` | Task-only baseline and Zero-Shot persona runs; also supports the Qwen2.5 temporal sensitivity check |
| `matched_one_shot/src/` | Primary One-Shot and demonstration-scenario ablations |
| `bmrn_cross_sector/` | BMRN Zero-Shot code and input parquet |
| `aapl_r2/` | Second AAPL stream code and input parquet |
| `persona_paraphrase_robustness/src/` | Two semantically equivalent persona descriptions |
| `SFT/` | Existing supervised-label data and data-preparation code |

The pre-existing Gemini Flash-Lite memory example is separate from the 360 paper trajectories. Supplementary rollout outputs and precomputed statistics are not included.

## Setup

Run commands from the repository root with Python 3.10 or later:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

API credentials and local model directories are supplied by environment variables:

```bash
export OPENAI_API_KEY="your_openai_key"
export GEMINI_API_KEY="your_gemini_key"
export THEATRON_QWEN3_8B_PATH="/your/path/Qwen3-8B"
export THEATRON_QWEN3_14B_PATH="/your/path/Qwen3-14B"
export THEATRON_QWEN25_PATH="/your/path/Qwen2.5-1.5B-Instruct"
```

Set only the variables needed for the model being run. Local paths must point to downloaded model directories. No credentials are included in this repository.

The experiment parameters, exact persona strings, and demonstration texts are embedded in the existing Python modules (`common.py` and `prompt.py`). No additional JSON/YAML configuration or external manifest is required. The existing `configs/experiment_config.yaml` is a generic reference, not a required runtime input.

Model keys used by the experiment-specific entry points are:

| Key | Thinking setting | Maximum output tokens |
|---|---|---:|
| `qwen3-8b-nothink` | Disabled | 512 |
| `qwen3-14b-thinking` | Enabled | 8192 |
| `gpt-5.4-mini` | Reasoning effort `none` | 512 |
| `gemini-2.5-flash` | Thinking budget 0 | 2048 |

These entry points retain the experimental sampling settings (`temperature=1.0`, `top_p=0.9`) and run-to-seed mapping.

## Main experiment commands

GPT Zero-Shot:

```bash
python persona_free_baseline/src/run_one.py \
  --model gpt-5.4-mini --symbol AAPL --persona overconfidence --run-id 1
```

Primary One-Shot (change `--model`, stock, persona, and run ID as needed):

```bash
python matched_one_shot/src/run_one.py \
  --model gpt-5.4-mini --symbol AAPL --persona overconfidence \
  --scenario M --run-id 1
```

The main `src.prompt_builder` also routes One-Shot for the three benchmark personas to this exact primary protocol.

Zero-Shot and CoT use the main entry point. For example:

```bash
python -m src.run_experiment \
  --symbol AAPL --persona overconfidence --method zero_shot \
  --model gemini-2.5-flash --backend api --api_provider gemini \
  --api_key_env GEMINI_API_KEY --max_new_tokens 2048 --top_p 0.9 \
  --run_id 1 --random_seed 41 --use_memory false

python -m src.run_experiment \
  --symbol TSLA --persona herding --method cot \
  --model Qwen3-14B-thinking --backend local --local_backend transformers \
  --local_model_path "$THEATRON_QWEN3_14B_PATH" --torch_dtype bfloat16 \
  --max_new_tokens 8192 --top_p 0.9 \
  --run_id 1 --random_seed 41 --use_memory false
```

For Qwen3-8B, use model tag `Qwen3-8B-nothink`, its corresponding local directory, and 512 output tokens. For five repeated runs, pair `--run_id 1–5` with `--random_seed 41–45`.

## Supplementary experiment commands

```bash
# No-persona baseline
python persona_free_baseline/src/run_one.py \
  --model gpt-5.4-mini --symbol AAPL --run-id 1

# Positive-evidence demonstration ablation (use N for negative evidence)
python matched_one_shot/src/run_one.py \
  --model gpt-5.4-mini --symbol AAPL --persona herding \
  --scenario P --run-id 1

# Cross-sector BMRN Zero-Shot
python bmrn_cross_sector/src/run_one.py \
  --model gpt-5.4-mini --symbol BMRN --condition overconfidence --run-id 1

# Second AAPL window
python aapl_r2/src/run_one.py \
  --model gpt-5.4-mini --symbol AAPL --condition overconfidence --run-id 1

# Persona paraphrase
python persona_paraphrase_robustness/src/run_one.py \
  --model gpt-5.4-mini --symbol AAPL --persona overconfidence \
  --paraphrase-version v1_controlled_lexical_substitution --run-id 1

# Temporal sensitivity check
python persona_free_baseline/src/run_one.py \
  --model qwen2.5-1.5b-instruct --symbol AAPL --run-id 1
```

For BMRN and R2, `--condition` accepts `task_only`, `overconfidence`, `loss_aversion`, or `herding`. The second paraphrase is `v2_sentential_reformulation`.

For the shared baseline/Zero-Shot entry point, omit `--persona` for task-only behavior or supply one of the three persona keys. The temporal sensitivity check uses `qwen2.5-1.5b-instruct` (512 output tokens); run both stocks, all four conditions, and run IDs 1–5 for its 40-rollout matrix.

Add `--dry-run` to any experiment-specific entry point above to validate its condition, data, and prompts without calling an API, loading model weights, or writing rollout outputs. Use `--help` for its remaining arguments.

## Input streams

- Original AAPL/TSLA and BMRN decision window: 2025-01-02 to 2025-05-30.
- Second AAPL (R2) decision window: 2025-06-02 to 2025-10-24.
- Baseline, One-Shot, paraphrase, and temporal sensitivity entry points reuse `data/AAPL-00000-of-00001.parquet` and `data/TSLA-00000-of-00001.parquet`.
- BMRN uses `bmrn_cross_sector/data/BMRN-00000-of-00001.parquet`.
- R2 uses `aapl_r2/data/AAPL-R2-2025-06-01_2025-10-25.parquet`.

The loaders align observations to NYSE sessions and roll non-trading-day textual evidence forward to the next session. `momentum` and `future_price_diff` are excluded from model inputs. BMRN/R2 retain input-file and processed date/price checks; loading them does not require separate provenance files.

## Action and return semantics

Each decision sets a one-period signed exposure: BUY = +1, HOLD = 0, SELL = −1. This is not an inventory-based portfolio simulator; HOLD denotes zero exposure, not retaining a previous position.

At the next observation, the previous action earns its signed log-price change. Cumulative log return is the sum of these action-conditioned returns. The first recorded return is zero. Closing prices from later observations are used to score the preceding action, never to construct that action's prompt.

## Raw main-experiment trajectories

The 360 paper CSVs cover every model × stock × persona × method × run combination. Their filenames identify the condition:

- Zero-Shot: `*_zero_shot_run*_seed*.csv`.
- Primary One-Shot: `*_one-shot-matched_M_run*_seed*.csv`.
- CoT: `*_cot_run*_seed*.csv`.

Shared columns are `date`, `symbol`, `price`, `action`, `rationale`, and `confidence`. The main engine records `daily_return` (earned from the preceding action) and `cumulative_log_return`. GPT Zero-Shot and the One-Shot entry point instead record `previous_action_return`, `cumulative_log_return_before_decision`, and `realized_next_log_return` (earned by the current action). Additional parser or memory fields depend on the entry point. The original schemas and values are preserved; these are trajectories, not statistical summaries.

New executions write outputs under the relevant experiment directory; the main `src` entry point writes under `outputs/`. Runners may generate local call logs, summaries, and manifests alongside CSVs. These generated files need not be included when distributing the code.

## SFT data

The existing `SFT/` files contain GPT-4.1-mini and rule-based labels and their preparation scripts. Data-preparation scripts resolve paths relative to their own directory. See `SFT/README.md` for the labeling format.
