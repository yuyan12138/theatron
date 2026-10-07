# Persona SFT Labeling Instruction

This pipeline MUST support using one of the following APIs for sentiment-assisted labeling and optional teacher-label generation:

- OpenAI GPT API
- Google Gemini API
- Anthropic Claude API
- DeepSeek API

The implementation should allow switching between providers using a configuration flag.

Example:

```python
LABEL_PROVIDER = "openai"
# options:
# "openai"
# "gemini"
# "claude"
# "deepseek"
```

The remainder of the instruction document should follow the previously provided step-by-step workflow for:
- trading-day alignment,
- weekend/holiday remapping,
- FinBERT sentiment extraction,
- persona-conditioned labeling,
- JSON export,
- CSV export,
- shuffling,
- and quality checks.

## API Requirements

### OpenAI

Support:
- GPT-4.1
- GPT-5 series

Environment variable:

```bash
OPENAI_API_KEY=...
```

### Gemini

Support:
- Gemini 2.5 Flash
- Gemini 2.5 Pro

Environment variable:

```bash
GEMINI_API_KEY=...
```

### Claude

Support:
- Claude Sonnet
- Claude Opus

Environment variable:

```bash
ANTHROPIC_API_KEY=...
```

### DeepSeek

Support:
- DeepSeek Chat
- DeepSeek Reasoner

Environment variable:

```bash
DEEPSEEK_API_KEY=...
```

## Unified Label Interface

All providers should expose the same function:

```python
def generate_persona_label(
    provider,
    persona_prompt,
    input_text
):
    ...
```

Expected output:

```json
{
  "position": "Long"
}
```

Allowed outputs:
- Long
- Short
- Flat

No other labels are allowed.

## Additional Constraints

1. The pipeline should default to FinBERT weak-labeling.
2. API-based labeling should be optional.
3. If API labeling is enabled:
   - use temperature=0,
   - deterministic output,
   - strict JSON parsing.
4. Do not use future returns.
5. Do not leak future information into prompts.
