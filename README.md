# qaz-ru-pii-150k

Synthetic RU/KK Kazakhstan banking PII dataset generator for local
[OpenAI Privacy Filter](https://github.com/openai/privacy-filter) fine-tuning.

The project is a dataset compiler, not a free-form text dump:

```text
policy -> ontology -> LLM template + values -> deterministic injection -> exact spans -> validation -> split -> OPF export
```

## Data Contract

The LLM returns raw records with template markers and a separate value object:

```json
{
  "template": "сәлеметсіз бе менің жсн {{IIN_1}} атым {{NAME_1}}",
  "values": {
    "IIN_1": {"label": "iin_kz", "value": "010203500123"},
    "NAME_1": {"label": "person_name", "value": "Нұрлан Әбдірахманұлы"}
  },
  "meta": {
    "language": "kk",
    "surface": "call_center_asr",
    "scenario": "identity_verification",
    "pii_density": "medium",
    "noise_type": "clean",
    "hard_negative": false
  }
}
```

The code then deterministically injects values left-to-right and computes exact
character offsets. The LLM never returns offsets.

The main 150k and real smoke configs are call-center first: `call_center_asr`
gets 70% of the surface mix, with chat, CRM notes, formal notices, and hard
negatives used as supporting slices. The prompt explicitly asks for dirty ASR:
broken names, repeated fragments, interruptions, lowercase/no punctuation, and
numbers spoken in Russian/Kazakh or mixed with digits.

Canonical JSONL keeps rich metadata and fine labels. OPF export writes the
Privacy Filter train/eval schema:

```json
{
  "text": "меня зовут Айжан Сейдахметова мой иин 010203500123",
  "spans": {
    "private_person: Айжан Сейдахметова": [[11, 29]],
    "account_number: 010203500123": [[38, 50]]
  },
  "info": {"id": "kz_bank_000000_0000", "source": "qaz-ru-pii-150k"}
}
```

## Quick Start

Install dependencies:

```bash
uv sync
```

Run the offline tiny pipeline with the fake LLM provider:

```bash
uv run qaz-ru-pii-150k generate \
  --config configs/generation_mix_tiny.yaml \
  --provider fake \
  --out data/tiny/canonical/raw.jsonl

uv run qaz-ru-pii-150k validate \
  --input data/tiny/canonical/raw.jsonl \
  --out data/tiny/canonical/validated.jsonl \
  --report data/tiny/reports/validation_v1.json

uv run qaz-ru-pii-150k split \
  --input data/tiny/canonical/validated.jsonl \
  --out data/tiny/splits \
  --seed 20260523

uv run qaz-ru-pii-150k export-opf \
  --input data/tiny/splits/train.jsonl \
  --out data/tiny/opf/train.jsonl \
  --mode opf8
```

Run tests:

```bash
uv run pytest
```

## OpenAI Generation

Set `OPENAI_API_KEY`, then use the real provider:

```bash
uv run qaz-ru-pii-150k generate \
  --config configs/generation_mix_150k.yaml \
  --provider openai \
  --out data/canonical/raw.jsonl
```

OpenAI settings live inside the selected generation YAML:

```yaml
openai:
  api_key: null
  api_key_env: OPENAI_API_KEY
  base_url: null
  base_url_env: OPENAI_BASE_URL
  model: gpt-5.1
  temperature: 0.9
  max_output_tokens: 6000
  cache_dir: data/raw_llm
```

Use environment variables for secrets by default. For a local private config,
`api_key` and `base_url` can be set directly in YAML; explicit YAML values take
priority over environment variables.

Raw LLM responses are cached per shard under `data/raw_llm`. Retries reuse
existing raw records, so injection and validation are reproducible from cached
records plus config.

## Temporal Local Mode

Start a local Temporal server separately, then run a worker:

```bash
uv run qaz-ru-pii-150k temporal-worker --config configs/generation_mix_150k.yaml
```

Start generation:

```bash
uv run qaz-ru-pii-150k generate-temporal \
  --config configs/generation_mix_150k.yaml \
  --provider openai
```

For local Temporal smoke runs without OpenAI credentials, pass `--provider fake`.

## OPF Training Command

After split/export:

```bash
uv run qaz-ru-pii-150k opf-train-command \
  --train data/opf/train.jsonl \
  --val data/opf/val.jsonl \
  --out checkpoints/opf-kz-banking-opf8-v1
```

This prints the `opf train ...` command expected by the local Privacy Filter CLI.
