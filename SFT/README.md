# Persona-Conditioned SFT Dataset Builder

End-to-end pipeline that turns daily per-stock parquet files into a supervised
fine-tuning (SFT) dataset with one labelled sample per *(stock, trading-day,
persona)* tuple. Default labeling backbone is **FinBERT** (local, no API);
GPT-4.1-mini and other providers are wired in as optional teacher models.

See `persona_sft_labeling_instruction.md` for the source specification.

---

## 1. Project structure

```
SFT/
├── build_persona_sft_dataset.py        # the pipeline (run this)
├── persona_sft_labeling_instruction.md # spec
├── README.md                           # you are here
├── data/                               # input parquet files (one per ticker)
│   ├── ADBE-00000-of-00001.parquet
│   ├── AMZN-00000-of-00001.parquet
│   ├── BMRN-00000-of-00001.parquet
│   ├── CRM-00000-of-00001.parquet
│   ├── GOOGL-00000-of-00001.parquet
│   ├── META-00000-of-00001.parquet
│   ├── MSFT-00000-of-00001.parquet
│   └── NVDA-00000-of-00001.parquet
├── persona_sft_training_data.json      # generated: fully-shuffled SFT list
└── persona_sft_training_data.csv       # generated: flat tabular mirror
```

### Input schema (per parquet row)

| column              | type           | notes                                         |
|---------------------|----------------|-----------------------------------------------|
| `date`              | string         | calendar day, including weekends/holidays     |
| `asset`             | string         | ticker                                        |
| `prices`            | float          | close (carried forward on non-trading days)   |
| `news`              | list[str]      | one or more news digests for the day          |
| `10k`               | list[str]      | non-empty only on 10-K release dates          |
| `10q`               | list[str]      | non-empty only on 10-Q release dates          |
| `momentum`          | string         | `bullish` / `bearish` / `neutral`             |
| `future_price_diff` | float          | **dropped** — would leak the future           |

---

## 2. Dependencies

Tested with the versions installed in this environment:

| package        | version |
|----------------|---------|
| python         | 3.11    |
| pandas         | 3.x     |
| numpy          | 2.x     |
| pyarrow        | 23.x    |
| transformers   | 5.x     |
| torch          | 2.x     |
| openai         | 2.x (only if `--label-provider openai`) |

Optional API SDKs (each only required if you select that provider):

```bash
pip install openai            # for --label-provider openai
pip install google-generativeai   # for --label-provider gemini
pip install anthropic         # for --label-provider claude
# DeepSeek uses the openai SDK against api.deepseek.com
```

---

## 3. Environment setup

```bash
# 1. (Optional) create a fresh venv / conda env with Python 3.11
conda create -n persona-sft python=3.11 -y
conda activate persona-sft

# 2. Install runtime deps
pip install pandas numpy pyarrow transformers torch

# 3. (Optional) provider SDKs you plan to use
pip install openai
```

### API keys

Only required when API-based labeling is enabled. The pipeline reads them from
environment variables — never from disk.

```bash
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=...
export ANTHROPIC_API_KEY=sk-ant-...
export DEEPSEEK_API_KEY=...
```

### FinBERT setup

FinBERT is downloaded on first run from the Hugging Face Hub
(`ProsusAI/finbert`, ~440 MB). No token required for anonymous downloads,
but you may set `HF_TOKEN` to raise rate limits:

```bash
export HF_TOKEN=hf_...
```

The model is cached under `~/.cache/huggingface/hub/`; subsequent runs are
offline. The pipeline auto-selects device in this order: CUDA → Apple MPS → CPU.

---

## 4. Running the pipeline

Default (FinBERT-only, no API calls, all 8 stocks):

```bash
python3 build_persona_sft_dataset.py
```

Smoke-test on a single ticker:

```bash
python3 build_persona_sft_dataset.py --limit-stocks 1
```

With GPT-4.1-mini as the teacher labeler (FinBERT still runs for the sentiment
features included in each prompt):

```bash
export OPENAI_API_KEY=sk-...
python3 build_persona_sft_dataset.py --label-provider openai
# optional: pin a different OpenAI model
python3 build_persona_sft_dataset.py --label-provider openai --openai-model gpt-4.1
```

Switch providers (each requires its own SDK and key):

```bash
python3 build_persona_sft_dataset.py --label-provider gemini
python3 build_persona_sft_dataset.py --label-provider claude
python3 build_persona_sft_dataset.py --label-provider deepseek
```

CLI flags:

| flag                  | default       | meaning                                              |
|-----------------------|---------------|------------------------------------------------------|
| `--label-provider`    | `finbert`     | `finbert` / `openai` / `gemini` / `claude` / `deepseek` |
| `--openai-model`      | `gpt-4.1-mini`| model id passed to OpenAI Chat Completions           |
| `--limit-stocks N`    | all           | use only the first `N` parquet files (smoke testing) |
| `--quiet`             | off           | suppress per-file progress logs                      |

API calls use `temperature=0` and `response_format={"type":"json_object"}` (for
OpenAI / Gemini); responses are strictly parsed and must yield exactly one of
`Long`, `Short`, `Flat`. Any parse failure falls back to the FinBERT rule label
and is logged to stderr.

---

## 5. Pipeline stages

1. **Load + clean** — read each parquet, drop `future_price_diff`, coerce
   `news`/`10k`/`10q` to `list[str]`.
2. **Trading-day alignment** — compute NYSE 2025 trading days
   (`pd.bdate_range` minus a hard-coded NYSE 2025 holiday set: New Year's,
   Carter day of mourning, MLK, Presidents, Good Friday, Memorial). Each
   calendar row's content is remapped to the **next** trading day — Saturday
   news lands on Monday's row, holiday news lands on the next session.
3. **FinBERT sentiment** — `ProsusAI/finbert` scores every news article and
   filing snippet. Long passages are chunked at 510 tokens; per-chunk softmax
   probabilities are averaged into one (`positive`, `neutral`, `negative`,
   `net = pos − neg`, `label`) record per trading day.
4. **Persona labeling** — three personas, deterministic rules (default) or
   API teacher (optional):
   - **Overconfidence** — strong conviction; picks a side aggressively, never Flat.
   - **Loss Aversion** — Flat unless evidence is strongly one-sided.
   - **Herding** — follows momentum; news sentiment is a tiebreaker.
5. **Sample assembly** — each sample is a chat-format SFT row
   (system / user / assistant). The assistant content is a strict JSON object:
   `{"position": "Long"|"Short"|"Flat"}`.
6. **Shuffle + export** — global shuffle with `random.Random(SEED)`, written to
   JSON (full structured records) and CSV (flat mirror).
7. **Stats + QC** — printed at end of run.

### No future leakage

- `future_price_diff` is dropped at load time.
- Prompts contain only as-of-day inputs (price, momentum, news, filings,
  sentiment of those inputs).
- A QC line at the end of the run counts occurrences of future-leakage
  keywords in the prompt text (expected: 0).

---

## 6. Output formats

### `persona_sft_training_data.json`

A single, fully shuffled top-level list. Each element:

```json
{
  "id": "MSFT-2025-05-07-Loss_Aversion",
  "asset": "MSFT",
  "date": "2025-05-07",
  "persona": "Loss Aversion",
  "label": "Long",
  "label_source": "finbert+rule",
  "momentum": "bullish",
  "prices": 431.0392,
  "sentiment": {
    "positive": 0.717, "neutral": 0.272, "negative": 0.011,
    "net": 0.706, "label": "positive"
  },
  "messages": [
    {"role": "system",    "content": "You are a trader with a LOSS AVERSION bias..."},
    {"role": "user",      "content": "Persona: Loss Aversion\nStock: MSFT\n..."},
    {"role": "assistant", "content": "{\"position\": \"Long\"}"}
  ],
  "meta": {
    "news_excerpt":   "...",
    "filing_excerpt": "...",
    "news_count":     1,
    "filing_count":   0
  }
}
```

### `persona_sft_training_data.csv`

One row per sample. Columns: `id, asset, date, persona, label, label_source,
momentum, prices, sent_positive, sent_neutral, sent_negative, sent_net,
sent_label, news_count, filing_count, system, user, assistant`.

Same rows, same ordering as the JSON (shuffle is applied before both writes).

---

## 7. Reproducibility

- Single random seed (`SEED = 20250523`) seeds `random` and `numpy`.
- The same seed shuffles the final sample list.
- FinBERT inference is deterministic with `model.eval()` and no dropout.
- API mode pins `temperature=0` and strict JSON parsing.
- Trading-day calendar is hard-coded for 2025 H1; no runtime calendar
  dependency.

Running the pipeline twice on the same machine, same FinBERT cache, same
provider yields identical JSON/CSV byte-for-byte.

---

## 8. Last known run (OpenAI `gpt-4.1-mini` teacher, 12 parallel workers)

```
Total samples:                2448
Unique (stock, trading-day):  816   (8 tickers × 102 trading days)
Date range:                   2025-01-02 -> 2025-05-30

Label distribution:
  Long    1229  (50.2%)
  Short    201  ( 8.2%)
  Flat    1018  (41.6%)

Persona x label:
  Overconfidence   Long=728  Short= 88  Flat=  0
  Loss Aversion    Long=  3  Short=  0  Flat=813
  Herding          Long=498  Short=113  Flat=205

Label source: openai+finbert  (all 2448 samples; zero API fallbacks)
```

The three personas converge to clearly different policies under the same
sentiment/momentum inputs:
- **Overconfidence** never goes Flat and tilts heavily Long (728 Long / 88 Short).
- **Loss Aversion** stays Flat ~99.6% of the time, only opening a Long
  position three times out of 816 trading days.
- **Herding** tracks the bullish-tilted momentum distribution while still
  taking Flat positions on ~25% of days when momentum is neutral.

For comparison, the FinBERT-only rule-based run on the same data produces
Long 1575 / Short 422 / Flat 451 — the GPT-4.1-mini teacher is markedly more
cautious about Short positions and far more disciplined about the
loss-aversion persona's risk threshold.
