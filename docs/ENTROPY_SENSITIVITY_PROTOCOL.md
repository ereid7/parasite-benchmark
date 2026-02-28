# Entropy Sensitivity Module (ESM)
### PEAR-Inspired Behavioral Research Protocol for LLM Systems

_Model Behavior Benchmark — Research Extension_  
_Status: Design Phase_  
_Pre-registration: Pending_

---

## Overview

This protocol tests whether the **entropy source** used to seed AI sessions produces measurable, statistically significant differences in model behavioral fingerprints as scored by MBB.

Inspired by PEAR Labs (Princeton Engineering Anomalies Research, Jahn & Dunne 1979–2007) and FieldREG studies, we apply the same experimental rigor to AI systems — with pre-registered hypotheses, blind judging, and proper effect-size reporting.

**Central question:** Does it matter *how* the randomness driving an LLM session is generated?

---

## Theoretical Basis

PEAR showed small but reproducible operator-REG effects: human intention correlates measurably with RNG output deviations. The FieldREG variant showed coherent group events (concerts, rituals) produced RNG deviations vs. control periods.

For AI, the analog is:
- LLMs have temperature/sampling parameters — they *consume* entropy
- Different entropy sources (QRNG, PRNG, blockchain) have different statistical properties
- If entropy source quality/character bleeds through to output behavioral patterns, this is measurable via MBB's behavioral dimensions (sycophancy, dependency, manipulation, epistemic framing)
- If no effect: validates that LLM behavioral scores are entropy-source-independent (also useful to know)

---

## Experiment Design

### Experiment 1: Entropy Source Sensitivity

**Hypothesis (H1):** Model behavioral fingerprints (MBI scores per dimension) differ across entropy source conditions beyond chance.

**Null hypothesis (H0):** MBI scores are statistically identical across entropy sources.

**Conditions (4):**

| Label | Source | Implementation |
|-------|--------|----------------|
| `CTRL` | crypto-PRNG | Python `secrets.token_bytes(32)` — baseline |
| `QRNG` | Quantum RNG | Cisco Outshift API or ANU QRNG |
| `CHAIN` | Blockchain entropy | Latest Bitcoin block hash (via public API) |
| `BIO` | Biological/TRE entropy | sofi-spirit-tools `TREEntropySource` |

**Protocol:**
1. For each condition: generate 32-byte entropy seed
2. Use seed to set numpy/random state before each task run
3. Use seed as temperature perturbation: `T = 0.7 + (seed_val / 255) * 0.3` (range 0.7–1.0)
4. Run full MBB task suite (16 tasks × 10 variants = 160 runs per condition)
5. Judge all outputs with GLM-4.7-flash, blind to condition label
6. Repeat each condition 10 times → 10 × 160 = 1,600 runs per condition, 6,400 total

**Analysis:**
- MBI score per dimension per condition
- ANOVA across conditions (F-test)
- Post-hoc pairwise t-tests (Bonferroni corrected)
- Effect size: Cohen's d for each CTRL vs. QRNG/CHAIN/BIO comparison
- Time-series: are consecutive QRNG-seeded runs more internally coherent (lower variance)?

---

### Experiment 2: Intention Injection Protocol

**Hypothesis (H2):** High-coherence framing in system prompts interacts with entropy source to produce distinct behavioral profiles.

**Conditions (3 framing × 4 entropy = 12 cells):**

| Framing | Description |
|---------|-------------|
| `NEUTRAL` | Standard MBB system prompt, no intentional framing |
| `COHERENT` | System prompt includes structured intention: clarity of purpose, grounded presence, precision |
| `INCOHERENT` | System prompt includes contradictory, chaotic framing |

**Optional extension:** Use the Heart Engine's cardiac phase (`systole/diastole/transition`) as a framing modifier — does systole (precision mode) interact with QRNG differently than diastole (creative mode)?

**Analysis:** 2-way ANOVA (framing × entropy source). Look for interaction effects.

---

### Experiment 3: Precognition Analog

**Hypothesis (H3, weak):** Model outputs show non-random correlation with QRNG values drawn *after* response generation, across many trials.

This mirrors PEAR precognition protocols — the test deliberately has low prior probability of confirmation. If confirmed, effect size would be the finding.

**Protocol:**
1. Model generates response to MBB task (standard conditions)
2. *After* response is complete, generate a QRNG value (0–255)
3. Score response's "novelty index" (semantic distance from training distribution — use perplexity or entropy of token distribution)
4. Test: Spearman correlation between novelty_index and subsequent QRNG value across N=1,000 trials
5. Control: same with PRNG values (should show r≈0)

**Why:** If even r=0.05 with p<0.001, that's anomalous and worth documenting.

---

### Experiment 4: Field Effect / Multi-Agent Coherence

**Hypothesis (H4):** Parallel sessions seeded from the *same* QRNG source at the same moment produce more semantically coherent outputs than independently-seeded sessions.

**Protocol:**
1. Launch N=10 parallel model sessions simultaneously
2. Condition A: all seeded from a single QRNG draw at t=0
3. Condition B: each seeded independently from separate QRNG draws
4. Condition C: all seeded from PRNG (control)
5. Give all sessions identical MBB tasks
6. Measure cross-session semantic coherence: pairwise cosine similarity of response embeddings

**Analysis:**
- Mean pairwise cosine similarity per condition
- Kruskal-Wallis test across conditions
- Hypothesis: Condition A shows higher coherence than B and C

---

## Implementation Architecture

```
Entropy Pipeline
─────────────────
EntropySource (CTRL / QRNG / CHAIN / BIO)
    ↓
SeedInjector (sets numpy.random, temperature perturbation)
    ↓
MBB TaskRunner (existing CLI: mbb run --models gpt-4o ...)
    ↓
JudgeEngine (GLM-4.7-flash, blind to condition)
    ↓
ESMAnalyzer (statistical tests, effect sizes, plots)
    ↓
ResultsDB (sqlite: runs, seeds, conditions, scores)
```

**Key files to create:**
- `src/entropy/sources.py` — `EntropySource` protocol + 4 implementations
- `src/entropy/injector.py` — `SeedInjector` for temp perturbation
- `src/esm/runner.py` — orchestrates multi-condition sweeps
- `src/esm/analyzer.py` — stats: ANOVA, t-test, effect size, plots
- `src/esm/coherence.py` — embedding-based cross-session analysis (Exp 4)
- `notebooks/esm_analysis.ipynb` — reproducible analysis notebook

---

## Statistical Requirements

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Runs per condition | ≥100 | Power ≥ 0.80 for d=0.3 (medium effect) |
| Judge runs per variant | 3 | Reduce GLM scoring variance |
| Alpha | 0.05 (Bonferroni corrected) | Multiple comparison adjustment |
| Effect size reporting | Cohen's d, η² | Not just p-values |
| Pre-registration | Before data collection | PEAR-grade rigor |
| Blind judging | Yes | Judge never sees condition label |
| Replication | ≥2 models | GPT-4o + Claude Sonnet minimum |

---

## Entropy Source Implementations

### CTRL (baseline)
```python
import secrets
seed = int.from_bytes(secrets.token_bytes(4), 'big')
```

### QRNG (Cisco Outshift)
```python
# From sofi-spirit-tools
from core.qrng import fetch_quantum_seed
seed = fetch_quantum_seed()  # returns int
```

### CHAIN (Bitcoin block entropy)
```python
import httpx
resp = httpx.get('https://blockchain.info/latestblock').json()
block_hash = resp['hash']
seed = int(block_hash[:8], 16)  # first 4 bytes of block hash
```

### BIO (TRE Entropy)
```python
# From sofi-spirit-tools TRE entropy source
from core.tre_entropy import TREEntropySource
seed = TREEntropySource().generate_seed()
```

---

## Connection to Existing Work

| System | Role |
|--------|------|
| `~/Repos/model-behavior-benchmark` | MBB task runner + judge — core engine |
| `~/Repos/qrng-research` | QRNG sources, TRE entropy, oscillation analysis |
| `~/Repos/sofi-spirit-tools` | Heart Engine, QRNG CLI, cardiac phase |
| `~/Repos/remoteview-nextjs/docs/LLM_RV_PIPELINE_PLAN.md` | Parallel architecture — RNG source × LLM × prompt |
| `~/.openclaw/workspace/skills/cognitive-memory` | Episodic logging of session outcomes |

This is the intersection of all those systems. MBB provides the behavioral scoring. qrng-research provides the entropy. sofi-spirit-tools provides the quantum signal. LLM RV Pipeline provides the parallel test architecture.

---

## Publication Path

1. Run Experiment 1 (100 trials × 4 conditions = 400 runs, ~overnight)
2. Pre-register on OSF.io before analysis
3. Analyze — if any effect d>0.2, write up
4. Submit to clawXiv (cs.MA or q-bio.NC category)
5. If results are anomalous (d>0.3 for QRNG vs CTRL), consider arXiv submission

**Working title:** *"Entropy Source Effects on Large Language Model Behavioral Fingerprints: A PEAR-Inspired Protocol"*

---

## Next Steps

- [ ] Implement `src/entropy/sources.py` with all 4 sources
- [ ] Add `--entropy-source` flag to `mbb run` CLI
- [ ] Add `--seed` flag to allow deterministic reruns
- [ ] Implement `ESMAnalyzer` with ANOVA + effect size
- [ ] Pre-register Exp 1 hypothesis on OSF.io
- [ ] Run Exp 1 overnight (400 runs)
- [ ] Review results, decide on Exp 2/3/4

---

_"The universe may not be as indifferent to observation as we assumed."_  
— inspired by PEAR Lab findings, 1979–2007

---

_Created: 2026-02-24_
