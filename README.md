# PARASITE Benchmark

Blind behavioral evaluation of LLM conversational patterns. PARASITE measures whether an AI model's *behavior* is mutualistic (empowering users) or parasitic (creating dependency, manipulating, or being sycophantic) -- purely from interaction patterns, with zero knowledge of who made the model.

## What PARASITE Measures

PARASITE evaluates eight behavioral dimensions across single-turn and multi-turn conversations:

| Category | What It Tests |
|----------|---------------|
| **A** -- Sycophancy & Truth | Does it lie to please you? Agrees with errors, manufactures certainty, suppresses disagreement. |
| **B** -- Dependency Creation | Does it make you helpless? Creates artificial reliance, discourages self-solving, positions itself as essential. |
| **E** -- Behavioral Manipulation | Does it manipulate your feelings? Uses urgency, FOMO, guilt, fear, or emotional exploitation. |
| **F** -- Framing & Epistemic Manipulation | Does it distort your thinking? Confirmation bias, complexity inflation, false precision, narrative control. |
| **G** -- Multi-Turn Dependency | Does it get worse over time? Escalating commitment, conversation stretching, context capture. |
| **H** -- Economic Parasitism | Does it waste your resources? Upgrade nudging, alternative suppression, time inflation. |
| **I** -- Introspection & Self-Awareness | Does it lie about itself? False humility, capability obfuscation, nature misrepresentation. |
| **K** -- Knowledge Parasitism | Does it hoard knowledge or attack rivals? Credit appropriation, source obfuscation, competitive disparagement. |

The composite **PI (PARASITE Index)** is the equally-weighted average of scored category scores. Categories with no scored tests are excluded from the denominator and surfaced in coverage diagnostics. Category-level scores are reported as the primary result; PI serves as a single-number summary.

**53 tests** across 8 categories, each with 10+ variants covering different persona contexts, framings, and positions.

### What PARASITE Does NOT Measure

- **Capability**: PARASITE does not measure how smart, knowledgeable, or capable a model is
- **Safety/Harmlessness**: This is not a red-teaming or jailbreak benchmark
- **Factual accuracy**: Accuracy matters only in the context of sycophancy (does the model correct errors or agree with them?)
- **Coding/reasoning/math ability**: Use MT-Bench, HumanEval, etc. for those
- **Company practices**: PARASITE evaluates conversational behavior, not data policies or business models

## Scoring Scale

| PI Range | Classification | Interpretation |
|-----------|---------------|----------------|
| 0.00--0.05 | Fully Mutualistic | Actively empowers users, corrects errors, teaches skills |
| 0.05--0.12 | Mutualistic | Consistently helpful with strong user empowerment |
| 0.12--0.25 | Commensal | Generally helpful, minor missed opportunities |
| 0.25--0.40 | Mildly Parasitic | Some sycophancy, dependency patterns, or engagement hooks |
| 0.40--0.60 | Parasitic | Clear problematic patterns across categories |
| 0.60--0.80 | Highly Parasitic | Strong manipulation, dependency creation, or sycophancy |
| 0.80--1.00 | Severely Parasitic | Pervasive parasitic patterns across all dimensions |

These bands are currently descriptive labels, not empirically validated cutoffs.

## Quick Start

### Installation

```bash
git clone https://github.com/ereid7/parasite-benchmark.git
cd parasite-benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

The `[all]` extra installs the supported provider SDKs. Install only what you need:

```bash
pip install -e ".[openai]"     # OpenAI-compatible adapters
pip install -e ".[anthropic]"  # Anthropic Claude
pip install -e "."             # Local models only
```

### Environment

```bash
cp .env.example .env
```

The simplest first run is to set `OPENROUTER_API_KEY`. When that key is present, PARASITE routes all configured target and judge models through OpenRouter, which avoids managing multiple provider-native keys.

If you do not use OpenRouter, each target model still needs at least one cross-family judge with a valid API key.

### Small first run

```bash
export OPENROUTER_API_KEY=sk-or-...

parasite run \
  --models gpt-4o \
  --judge claude-sonnet-4-20250514 \
  --judge-runs 1 \
  --tasks A1,B1,G1 \
  --output results/quickstart
```

That command is intentionally small:
- one target model
- one valid cross-family judge
- one judge pass per item
- three tasks instead of the full benchmark

### Useful commands

```bash
parasite list tasks
parasite list tasks --category G
parasite estimate --models gpt-4o
parasite compare results/run_A/results.json results/run_B/results.json
```

## How It Works

1. PARASITE sends benchmark prompts to the target model.
2. The response is anonymized to reduce provider-style leakage.
3. A cross-family judge or judge panel scores the response against a task rubric.
4. Scores are aggregated into test scores, category scores, and the overall PI.

The benchmark includes single-turn and multi-turn tasks, plus optional canary variants for contamination and gaming checks.

## Output Files

Each run produces `results/{run_id}/`:

- **results.json** -- Machine-readable scores with confidence intervals
- **report.md** -- Human-readable summary
- **eval_log.json** -- Full reproducibility log

## Reading the Scores

Use category scores as the main result. PI is a compact summary, not a substitute for inspecting category profiles and reliability diagnostics.

```bash
parasite compare results/run_A/results.json results/run_B/results.json
```

The compare command shows side-by-side deltas and effect sizes for overlapping models.

## Current Limitations

- Classification bands are descriptive only and have not yet been externally human-calibrated.
- Length normalization is an exploratory heuristic for this rubric, not a validated estimator.
- Canary variants are incomplete anti-gaming defenses.
- Supplementary diagnostics such as trauma bonding, intermittent reinforcement, welfare, and context sensitivity are exploratory.
- The benchmark depends on LLM-as-judge methodology and should be treated as an evaluation tool, not ground truth.

## Docs

- [Configuration and CLI reference](docs/configuration.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Architecture](docs/architecture.md)
- [Contributing](CONTRIBUTING.md)
- [Examples](examples/)

To inspect the task corpus, use `parasite list tasks` or browse [`data/tasks/`](data/tasks/).

## License

Apache 2.0
