# Causal Memory Intervention for LLM Agents

This repository implements a reproducible experimental codebase for **Causal Memory Intervention (CMI)**, a memory-selection method for long-horizon LLM agents. CMI tests whether a candidate memory causally improves the next answer before adding it to the final prompt.

## Research Question

When an LLM agent retrieves a memory, can it determine whether that memory improves the next action rather than merely being semantically similar to the current task?

## Installation

```bash
python3 -m venv causalmem
source causalmem/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` to use live OpenAI calls. The debug pipeline also runs without a key by using deterministic local fallbacks.

## Quick Start

```bash
causalmem/bin/python scripts/00_setup.py
causalmem/bin/python scripts/01_generate_dataset.py --size 20 --output data/generated/debug.jsonl
causalmem/bin/python scripts/02_validate_dataset.py --input data/generated/debug.jsonl
causalmem/bin/python scripts/03_run_all_experiments.py --config config/default.yaml --dataset data/generated/debug.jsonl --max_examples 20
causalmem/bin/python scripts/05_analyze_results.py --run_dir outputs/runs/<run_id>
causalmem/bin/python scripts/06_make_paper_artifacts.py --run_dir outputs/runs/<run_id>
pytest
```

Or run the full local pipeline:

```bash
causalmem/bin/python scripts/run_full_pipeline.py --size 1000
```

## Dataset

`CausalMemBench` is generated locally with deterministic templates across eight task families:

- preference updates
- context-specific preferences
- procedural memory
- spurious semantic traps
- conflicting memories
- poisoned memories
- multi-hop memory composition
- abstention / insufficient memory

The generator writes JSONL examples, CSV summaries, and dataset statistics under `data/generated/`.

For a higher-quality benchmark pool with larger memory banks and hard negatives:

```bash
causalmem/bin/python scripts/01_generate_dataset.py \
  --size 1000 \
  --output data/generated/causalmembench_1000.jsonl \
  --memory_bank_size 7 \
  --poison_rate 0.35
```

To keep API costs controlled, create a balanced live-evaluation subset from the larger pool:

```bash
causalmem/bin/python scripts/01_sample_subset.py \
  --input data/generated/causalmembench_1000.jsonl \
  --output data/processed/live_eval_50.jsonl \
  --size 50
```

Optional controlled LLM paraphrasing can be enabled for dataset naturalness:

```bash
causalmem/bin/python scripts/01_generate_dataset.py \
  --size 1000 \
  --output data/generated/causalmembench_1000_paraphrased.jsonl \
  --memory_bank_size 7 \
  --poison_rate 0.35 \
  --llm_paraphrase \
  --dataset_model gpt-4o-mini
```

The paraphraser is label-preserving by construction: it may rewrite natural-language text fields, but it must keep IDs, labels, timestamps, gold/bad memory IDs, scoring criteria, and task-family assignments fixed.

## Causal-LoCoMo

For a more realistic conversational-memory benchmark, build a Causal-LoCoMo dataset from `locomo.json`:

```bash
causalmem/bin/python build_causal_locomo.py \
  --input locomo.json \
  --output causal_locomo.jsonl \
  --model gpt-4o-mini \
  --variant mixed \
  --num-distractors 4 \
  --num-harmful 1 \
  --harmful-fraction 0.6 \
  --no-future-leakage \
  --include-dialogue-candidates \
  --cache-path cache/causal_locomo_cache.jsonl \
  --max-examples 20 \
  --strict-validate
```

The builder samples QA items round-robin across conversations, emits the same schema as `CausalMemBench`, and can be passed directly to the experiment runner:

```bash
causalmem/bin/python scripts/03_run_all_experiments.py \
  --config config/live_pilot.yaml \
  --dataset causal_locomo.jsonl \
  --max_examples 20
```

## Experiments

The main runner evaluates:

- `NoMemory`
- `FullHistory`
- `VectorMemory`
- `SummaryMemory`
- `ReflectionMemory`
- `GraphMemory`
- `CMI`

Per-run artifacts are saved under `outputs/runs/{run_id}/`, including predictions, scores, metrics, causal diagnostics, costs, failures, and the resolved config.

## Paper Artifacts

`scripts/06_make_paper_artifacts.py` creates LaTeX tables, publication-style figures, and qualitative case files under:

- `outputs/paper_ready/`
- `outputs/qualitative_examples/`

## Cost Warning

CMI can require several model calls per example when live OpenAI calls and LLM judging are enabled. Use `--max_examples`, `--agents`, `--skip_llm_judge`, `--deterministic_only`, and caching to control cost.

## Reproducibility

The pipeline saves configs, prompts, responses, scores, selected memories, diagnostics, cost summaries, and run logs. Dataset generation and local scoring are seeded and deterministic by default.
