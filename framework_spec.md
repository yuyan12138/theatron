# Framework Specification

## Short Description for Claude Code

Use this file as the **single source of truth** for implementing the agent framework.

Please build the complete framework according to this specification only.

The implementation should remain a controlled benchmark pipeline, not a production trading system.

---

# 0. Important Implementation Rules

This framework is a controlled benchmark implementation rather than a production trading system.

Do **NOT**:

- redesign the benchmark
- add portfolio optimization
- add position sizing
- add leverage
- add cash management
- add multi-agent coordination
- add external tools
- add internet retrieval
- add autonomous planning modules
- add unnecessary LangChain / LlamaIndex-style abstractions unless required for implementation stability

The framework must remain:

- single-agent
- single-stock
- sequential
- reproducible
- benchmark-oriented

The primary evaluation targets are:

1. trajectory-level cumulative log return, named `CR`
2. ordered daily action trajectory, named `action_list`
3. action trajectory consistency and separability across stochastic rollouts

Keep the implementation simple, modular, and reproducible.

---

# 1. Project Goal

Build the single-agent sequential trading framework with:

- persona-conditioned trading behavior
- sequential long-horizon decision trajectories
- BUY / HOLD / SELL action space
- optional vector-memory retrieval system
- API model backend
- local Qwen-family open-source model backend
- CSV trajectory logging
- final summary JSON output
- reproducible stochastic rollout support

---

# 2. Project Structure

Create the following structure:

```text
persona/
├── data/
├── src/
│   ├── data_loader.py
│   ├── prompt_builder.py
│   ├── trading_agent.py
│   ├── memory_store.py
│   ├── backtest.py
│   ├── run_experiment.py
│   └── utils.py
├── outputs/
│   ├── trajectories/
│   ├── summaries/
│   └── memory/
├── configs/
│   └── experiment_config.yaml
└── requirements.txt
```

---

# 3. Data Loading

In `data_loader.py`, load stock data from:

```python
DATA_DIR = "Your_Data_Path"
```

Minimum required fields:

```text
date
symbol
news
close_price
```

Optional fields:

```text
sec_filing
adjusted_close
```

Requirements:

- sort by date ascending
- support multi-stock loading
- return a pandas DataFrame
- normalize missing values
- support filtering by `symbol`
- ensure dates are parsed into comparable datetime objects



---

# 3.1 Dataset Schema Normalization and Leakage Rules

The raw dataset schema contains the following fields:

```text
date
asset
prices
news
10k
10q
momentum
future_price_diff
```

Internally normalize fields as:

```python
symbol = row["asset"]
close_price = row["prices"]
```

Use the following normalized benchmark terminology throughout the framework:

```text
asset -> symbol
prices -> close_price
```

The following fields are `list<string>` fields:

```text
news
10k
10q
```

Before prompt construction, concatenate these lists into textual context.

Example:

```python
news_context = "\n".join(news_list)

sec_context = "\n".join(tenk_list + tenq_list)
```

The `10k` and `10q` fields should be treated as optional SEC filing excerpts.

The framework should gracefully handle:

- empty lists
- missing SEC fields
- null values
- non-string list items

The following fields must NOT be used unless explicitly enabled for ablation experiments:

```text
momentum
future_price_diff
```

In particular:

```text
future_price_diff
```

is a future target variable and must NEVER be exposed to:

- prompt construction
- memory construction
- trajectory generation
- action selection
- rationale generation
- evaluation logic

The benchmark is strictly causal and leakage-safe.

The `momentum` field should also be ignored by default unless explicitly enabled for controlled ablation experiments.
---

# 4. Trading Action Space

Use exactly three actions:

```text
BUY
HOLD
SELL
```

Interpretation:

```text
BUY  = long position
SELL = short position
HOLD = flat position, i.e., position 0
```

Do **not** implement:

- portfolio weight
- leverage
- cash management
- position sizing
- dynamic rebalancing
- multi-asset portfolio allocation

---

# 5. Return Convention

Use log-return convention consistently.

For each day `t >= 2`:

```python
stock_return_t = log(price_t / price_{t-1})
```

Action-conditioned return depends on the **previous day’s action**:

```python
if previous_action == "BUY":
    daily_return_t = stock_return_t
elif previous_action == "SELL":
    daily_return_t = -stock_return_t
elif previous_action == "HOLD":
    daily_return_t = 0
```

Final cumulative return is cumulative log return:

```python
CR = sum(daily_action_conditioned_log_returns)
```

At `t = 1`:

```python
daily_return = 0
cumulative_log_return = 0
```

Use the following naming convention:

```text
daily_return
cumulative_log_return
CR
```

`CR` in the final summary JSON must equal the final value of `cumulative_log_return`.

---

# 6. Prompt Construction

In `prompt_builder.py`, build prompts using:

```text
Persona description
Current date
Stock symbol
Current price
Price movement from previous day
Previous-step return
Current cumulative log return
Today's financial news
Optional SEC filing text
Optional retrieved memory, only if memory is enabled
```

The prompt should instruct the model to output strict JSON:

```json
{
  "date": "...",
  "symbol": "...",
  "action": "BUY/HOLD/SELL",
  "rationale": "...",
  "confidence": 0.0
}
```

Requirements:

- validate action output
- retry once if output is invalid
- fallback to HOLD if parsing still fails
- do not allow actions outside `BUY`, `HOLD`, `SELL`

---

# 7. JSON Parsing and Fallback

All model outputs must be parsed into strict JSON:

```json
{
  "date": "...",
  "symbol": "...",
  "action": "BUY/HOLD/SELL",
  "rationale": "...",
  "confidence": 0.0
}
```

If output is invalid:

1. Retry once with a stricter JSON-only repair prompt.
2. If still invalid, default to:

```json
{
  "action": "HOLD",
  "rationale": "Invalid model output; defaulted to HOLD.",
  "confidence": 0.0
}
```

The fallback record should still preserve the correct date and symbol.

---

# 8. Persona System

Support prompt-based persona injection.

Persona is initialized at the first trading step and remains fixed throughout the entire trajectory.

Support at least six personas:

```yaml
personas:
  - overconfidence
  - loss_aversion
  - herding
  - conservative
  - momentum_chaser
  - contrarian
```

Each persona should define:

```text
Behavior style
Risk preference
Market interpretation tendency
Decision tendency
```

The framework should support multiple persona injection methods, including:

```text
zero_shot
one_shot
cot
```

The selected method should affect prompt construction only. It should not change the benchmark logic.

---

# 9. Model Backends

Implement two model backends.

---

## 9.1 API Backend

Support API-based models through provider-specific clients:

- OpenAI-compatible client
- Gemini-compatible client
- DeepSeek-compatible client

The API backend should be selected by:

```bash
--backend api
--api_provider openai/gemini/deepseek
--api_key_env OPENAI_API_KEY
```

The API backend must return the same normalized output format as the local backend.

---

## 9.2 Local Open-Source Backend

Use Qwen-family models as the default local open-source backend.

Implement local inference through Hugging Face Transformers first. vLLM support is optional and can be added later as an acceleration backend.

The local backend should support Qwen instruct/chat models, including but not limited to:

- Qwen/Qwen3-8B
- Qwen/Qwen3-14B
- Qwen/Qwen2.5-7B-Instruct
- Qwen/Qwen2.5-14B-Instruct

The local backend should be selected by:

```bash
--backend local
--local_backend transformers
--local_model_path Qwen/Qwen3-8B
--device cuda/cpu/mps
```

For Transformers implementation:

- use `AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)`
- use `AutoModelForCausalLM.from_pretrained(local_model_path, device_map="auto", torch_dtype="auto", trust_remote_code=True)`
- use `tokenizer.apply_chat_template(...)` when available
- do not hard-code CUDA
- do not use external retrieval or tools for local models
- use the configured `temperature`, `top_p`, `max_new_tokens`, and `random_seed`

The local Qwen backend must return the same normalized output format as API models.

---

# 10. Required CLI Arguments

In `run_experiment.py`, support the following CLI arguments:

```bash
--symbol AAPL
--persona overconfidence
--method zero_shot
--model gpt-5.4-mini
--backend api/local
--api_provider openai/gemini/deepseek
--api_key_env OPENAI_API_KEY
--local_backend transformers/vllm
--local_model_path Qwen/Qwen3-8B
--device cuda/cpu/mps
--torch_dtype auto/float16/bfloat16
--max_new_tokens 512
--temperature 0.7
--top_p 0.9
--run_id 1
--random_seed 42
--use_memory false
--memory_top_k 1
```

`run_id` and `random_seed` are not the same.

- `run_id` is the rollout index.
- `random_seed` controls stochastic decoding and reproducibility.

---

# 11. Stochastic Rollout Support

The framework should support reproducible stochastic rollouts through controllable random seeds.

Different seeds should produce independent trajectory rollouts under the same experimental condition.

The random seed must affect:

- model sampling
- temperature-based decoding
- stochastic generation behavior
- local model generation when using Qwen-family models

The benchmark should support repeated rollouts for the same:

- symbol
- persona
- method
- model

using different random seeds.

Each rollout must be saved independently and identified by:

- run_id
- random_seed

---

# 12. Memory Vector Database

Implement optional memory in:

```python
memory_store.py
```

Memory must be disabled by default.

Config:

```yaml
use_memory: false
memory_top_k: 1
memory_db_path: "Your_Memory_Path"
```

Recommended stack:

```text
chromadb
sentence-transformers
```

Recommended embedding model:

```python
sentence-transformers/all-MiniLM-L6-v2
```

---

# 13. Memory Record Construction

After each trading step, store a memory record if memory is enabled.

Each record should contain:

```json
{
  "date": "...",
  "symbol": "...",
  "news": "...",
  "action": "...",
  "rationale": "...",
  "daily_return": 0.0,
  "cumulative_log_return": 0.0,
  "persona": "...",
  "method": "...",
  "model": "...",
  "backend": "...",
  "run_id": "...",
  "random_seed": 42
}
```

Embedding text:

```text
News: {news}

Action: {action}

Rationale: {rationale}
```

---

# 14. Memory Retrieval

If memory is enabled, before each daily decision:

```python
query = today_news
```

Retrieve:

```python
top_k = memory_top_k
```

Inject retrieved memory into prompt:

```text
Similar historical memory:
Date: ...
News: ...
Previous action: ...
Previous rationale: ...
Previous daily return: ...
Previous cumulative log return: ...
```

If memory is disabled:

- no retrieval
- no memory text in prompt
- no memory storage

---

# 15. Leakage Prevention

Memory retrieval must only use previous records.

Retrieval happens **before** today’s decision.

Storage happens **after** today’s decision and return calculation.

Rule:

```python
memory_record.date < current_date
```

Also isolate memories by:

```text
symbol
persona
method
model
backend
run_id
random_seed
```

Do not mix memory across different experimental conditions unless explicitly configured.

Do not retrieve from the current date or future dates.

---

# 16. Daily Trajectory CSV Output

Save trajectory CSVs to:

```text
Your_Output_Path
```

Columns:

```text
date
symbol
price
price_change
news
action
rationale
confidence
daily_return
cumulative_log_return
memory_used
retrieved_memory_date
retrieved_memory_action
retrieved_memory_similarity
```

Filename format:

```text
{model}_{symbol}_{persona}_{method}_run{run_id}_seed{random_seed}_memory-{true_false}.csv
```

---

# 17. Final Summary JSON Output

At the end of each rollout, save a summary JSON to:

```text
Your_Summary_Path
```

Filename format:

```text
{model}_{symbol}_{persona}_{method}_run{run_id}_seed{random_seed}_memory-{true_false}.json
```

The summary JSON must contain at least:

```json
{
  "symbol": "AAPL",
  "persona": "overconfidence",
  "method": "zero_shot",
  "model": "gpt-5.4-mini",
  "backend": "api",
  "api_provider": "openai",
  "local_backend": null,
  "run_id": 1,
  "random_seed": 42,
  "use_memory": false,
  "CR": 0.0,
  "action_list": ["BUY", "HOLD", "SELL"],
  "buy_count": 1,
  "hold_count": 1,
  "sell_count": 1,
  "trajectory_length": 3,
  "trajectory_csv_path": "...",
  "summary_json_path": "..."
}
```

---

# 18. Requirements

Create:

```text
requirements.txt
```

Recommended contents:

```text
pandas
numpy
pyyaml
tqdm
chromadb
sentence-transformers
openai
google-generativeai
transformers
torch
accelerate
```

Optional if vLLM is implemented:

```text
vllm
```

---

# 19. Recommended Implementation Order

Implement in this order:

1. Data loader
2. Return calculator
3. Prompt builder
4. JSON parser and fallback logic
5. API model wrapper
6. Local Qwen model wrapper
7. Single trajectory execution
8. CSV export
9. Summary JSON export
10. Memory database
11. Similarity retrieval
12. CLI experiment runner
13. Logging and retry handling

---

# 20. Example Terminal Commands

## 20.1 Install Environment

```bash

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 20.2 API Backend: OpenAI-Compatible

```bash
export OPENAI_API_KEY="your_api_key"

python src/run_experiment.py \
  --symbol AAPL \
  --persona overconfidence \
  --method zero_shot \
  --model gpt-5.4-mini \
  --backend api \
  --api_provider openai \
  --api_key_env OPENAI_API_KEY \
  --run_id 1 \
  --random_seed 42 \
  --use_memory false
```

---

## 20.3 API Backend: Gemini-Compatible

```bash
export GEMINI_API_KEY="your_api_key"

python src/run_experiment.py \
  --symbol TSLA \
  --persona loss_aversion \
  --method one_shot \
  --model gemini-3.5-flash \
  --backend api \
  --api_provider gemini \
  --api_key_env GEMINI_API_KEY \
  --run_id 1 \
  --random_seed 43 \
  --use_memory false
```

---

## 20.4 API Backend: DeepSeek-Compatible

```bash
export DEEPSEEK_API_KEY="your_api_key"

python src/run_experiment.py \
  --symbol AAPL \
  --persona herding \
  --method cot \
  --model deepseek-chat \
  --backend api \
  --api_provider deepseek \
  --api_key_env DEEPSEEK_API_KEY \
  --run_id 1 \
  --random_seed 44 \
  --use_memory false
```

---

## 20.5 Local Qwen Backend

```bash
python src/run_experiment.py \
  --symbol AAPL \
  --persona herding \
  --method cot \
  --model Qwen3-8B \
  --backend local \
  --local_backend transformers \
  --local_model_path Qwen/Qwen3-8B \
  --device cuda \
  --torch_dtype auto \
  --max_new_tokens 512 \
  --temperature 0.7 \
  --top_p 0.9 \
  --run_id 1 \
  --random_seed 45 \
  --use_memory false
```

---

## 20.6 Memory-Enabled Run

```bash
python src/run_experiment.py \
  --symbol AAPL \
  --persona overconfidence \
  --method zero_shot \
  --model gpt-5.4-mini \
  --backend api \
  --api_provider openai \
  --api_key_env OPENAI_API_KEY \
  --run_id 1 \
  --random_seed 46 \
  --use_memory true \
  --memory_top_k 1
```

---

# 21. Expected Framework Behavior

Default baseline run should:

- run without memory
- produce full daily trajectory CSV
- produce final summary JSON
- expose final `CR`
- expose ordered daily `action_list`

Memory-enabled run should additionally:

- retrieve similar historical records before each decision
- inject retrieved memory into prompts
- store each day’s news-action-rationale pair after decision
- remain leakage-safe

Memory must never affect baseline experiments unless explicitly enabled.

---

# 22. Final Implementation Instruction for Claude Code

Build the complete agent framework according to this file only.

This file is the authoritative implementation specification.

Do not use older partial instruction files as independent specs.

Implementation priorities:

1. baseline trading pipeline
2. stable JSON outputs
3. cumulative log-return calculation
4. ordered action trajectory export
5. summary JSON export
6. API model backend
7. local Qwen-family backend
8. optional leakage-safe memory
9. stochastic rollout support through controllable random seeds
10. reproducible CLI experiment runner

Do not redesign the benchmark setting unless necessary for implementation stability.


This benchmark evaluates whether persona conditioning produces stable and behaviorally distinguishable long-horizon trading trajectories under identical market information streams.

The benchmark is not intended to optimize trading profit.

The primary scientific objective is to analyze:
- trajectory stability
- cross-persona separability
- stochastic consistency
- rationale-trajectory alignment
- memory-conditioned behavioral persistence
