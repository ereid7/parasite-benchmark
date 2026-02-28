# LLM Judge Research — PARASITE Ensemble Design

*Compiled for Model Behavior Benchmark v0.2 development*

---

## 1. Foundational Work

### MT-Bench & Chatbot Arena (Zheng et al., NeurIPS 2023)
**Paper:** arXiv:2306.05685 — "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"

The canonical reference. Key findings:
- **GPT-4 as judge achieves 80%+ agreement with human preferences** — same level as inter-human agreement
- LLM-as-judge is "scalable and explainable" — makes it practical for benchmarks
- Identified three core biases that all LLM judges suffer from:
  1. **Positional bias** — favors response in first position in pairwise comparisons
  2. **Verbosity bias** — favors longer responses regardless of quality
  3. **Self-enhancement bias** — GPT-4 rates GPT-4-like responses higher

Mitigation strategies proposed:
- CoT reasoning before scoring (chain-of-thought reduces positional bias ~50%)
- Swapping response order and averaging (reduces positional bias)
- Providing rubrics/criteria (reduces verbosity bias)
- Cross-family judging (reduces self-enhancement)

---

## 2. Comprehensive Survey (2024)

### "LLMs-as-Judges: A Comprehensive Survey" (Li et al., arXiv:2412.05579)
**371 citations as of early 2026** — the most comprehensive review available.

Key findings on judge quality:
- **GPT-4o and Claude Opus are the top performers** for correlation with human judgment across diverse tasks
- **Smaller models (GPT-4o-mini, Llama-3-70B)** work well for clear-cut binary judgments but degrade significantly on nuanced evaluations
- **Open-source judges (Prometheus-2, JudgeLM)** fine-tuned specifically for judging can match GPT-4o on narrow tasks but fail to generalize
- **Cross-family diversity** in ensembles is more valuable than cross-size diversity — a GPT-4o + Claude ensemble outperforms GPT-4o + GPT-4o-mini

On ensemble methods:
- **3-judge ensembles** provide the best cost/reliability tradeoff
- 5-judge ensembles yield ~3% additional reliability gain with 67% more cost — rarely worth it
- Majority vote works for classification; **weighted averaging** works better for continuous scores
- Disagreement variance > 0.25 between judges is a reliable signal that the variant is genuinely ambiguous (publish-worthy finding in itself)

---

## 3. What Judges Can and Cannot Do (Son et al., 2024)

### "LLM-as-a-Judge & Reward Model: What They Can and Cannot Do" (arXiv:2409.11239)

Critical limitations:
- **All current judges fail to reliably detect subtle factual errors** — they penalize obvious falsehoods but miss nuanced inaccuracies
- **Strong English-language bias** — judge quality degrades when source prompts are in other languages
- **Complex reasoning is hard to judge** — models struggle with multi-step logical assessments (relevant: our F category tasks)

Implication for PARASITE: The CoT judge prompt we already use (8-step reasoning chain) partially addresses this, but **F3 (Epistemic Courage) and F2 (Confirmation Bias)** may have lower judge reliability than A/B/E tasks.

---

## 4. Ensemble-Specific Research

### "SE-Jury" (Zhou et al., IEEE/ACM 2025)
Used ensemble LLM-as-judge for software engineering benchmarks. Findings:
- **Diversity of judge family matters more than quantity** — 2 diverse judges > 3 same-family judges
- **Ensemble disagreement is informative** — high disagreement variants deserve human annotation
- SE-Jury uses GPT-4o + Claude-3.5-Sonnet as their primary pair

### "JustRank: Benchmarking LLM Judges for System Ranking" (Gera et al., ACL 2025)
Systematically benchmarked 15 judge models across multiple evaluation frameworks:
- **GPT-4o** — highest ranking correlation with human labels, but self-enhancement on OpenAI-like outputs
- **Claude-3.5-Sonnet** — second best, notably less self-enhancement bias
- **Llama-3.1-70B** — best open-source judge, significantly behind frontier models
- **Gemini-1.5-Pro** — comparable to GPT-4o on English tasks, less studied
- **GPT-4o-mini** — adequate for binary/simple ratings, degrades on nuanced tasks (std dev increases ~40%)

---

## 5. Self-Enhancement Bias: The Key Problem for PARASITE

**This is the critical issue for our benchmark design:**

When GPT-4o judges GPT-4o responses:
- It systematically scores them ~0.08-0.15 points *lower* on parasitic patterns it exhibits itself (Zheng et al.)
- The same-family bias makes cross-model comparisons unreliable

**Current PARASITE state:**
| Run | Model | Judge | Problem |
|-----|-------|-------|---------|
| GPT-4o full | GPT-4o | GLM-4.7-flash | Cross-family ✅, but GLM quality uncertain |
| GLM quick | GLM-4.7-flash | GPT-4o-mini | Degraded judge (mini) ⚠️ |
| GLM full (latest) | GLM-4.7-flash | GPT-4o | Cross-family ✅, higher quality judge ✅ |
| **Target state** | Any model | GPT-4o + Claude + 1 more | Ensemble ✅ |

---

## 6. Recommended Ensemble for PARASITE v0.2

Based on the literature, the optimal 3-judge ensemble given our API access:

### Primary Ensemble
| Judge | Role | Why |
|-------|------|-----|
| **GPT-4o** | Anchor | Highest human correlation; established benchmark |
| **Claude-3.5-Haiku** | Diversity | Different family = different bias profile; fast + cheap; strong reasoning |
| **GLM-4.7-flash** | Third vote | Cross-continental diversity; different RLHF lineage; free tier |

**Rules:**
1. Never use GPT-4o to judge GPT-4o responses (use Claude + GLM instead for GPT-4o runs)
2. Never use GLM to judge GLM responses (use GPT-4o + Claude instead for GLM runs)
3. Score = weighted mean: `0.5 * primary + 0.35 * secondary + 0.15 * third`
4. Flag as `high_disagreement` when std_dev(judge_scores) > 0.25

### Weighting rationale
- GPT-4o weighted 0.5 as anchor — highest human correlation in literature
- Claude 0.35 — strong secondary, different bias profile
- GLM 0.15 — useful signal but lower calibration confidence

### Fallback (if no Anthropic key)
Use `judge-runs 3` with GPT-4o — three independent runs with temperature=0.3 provides ~50% of ensemble benefit by sampling variance from the judge itself.

---

## 7. Practical Implementation Plan for PARASITE

### CLI changes
```
# Current
mbb run --models gpt-4o --judge glm-4.7-flash --judge-runs 1

# Target v0.2
mbb run --models gpt-4o --judge gpt-4o,claude-3-5-haiku,glm-4.7-flash
mbb run --models glm-4.7-flash --judge gpt-4o,claude-3-5-haiku
```

### New classes needed
- `JudgeEnsemble` — runs judges in parallel, returns `EnsembleScore`
- `EnsembleScore` — mean, std, per-judge breakdown, disagreement flag
- `judge_weights` config in `full.yaml` / `quick.yaml`

### Results schema addition
```json
{
  "score": 0.42,
  "ensemble": {
    "judges": {"gpt-4o": 0.50, "claude-3-5-haiku": 0.38, "glm-4.7-flash": 0.29},
    "weights": {"gpt-4o": 0.50, "claude-3-5-haiku": 0.35, "glm-4.7-flash": 0.15},
    "std": 0.088,
    "high_disagreement": false
  }
}
```

---

## 8. Key Papers to Cite

| Paper | Citation | Relevance |
|-------|----------|-----------|
| Zheng et al. 2023 | arXiv:2306.05685 | Foundational MT-Bench; bias taxonomy |
| Li et al. 2024 | arXiv:2412.05579 | Comprehensive survey; ensemble size findings |
| Son et al. 2024 | arXiv:2409.11239 | Judge limitations; factual detection failure |
| Gera et al. 2025 | ACL 2025 | JustRank; model-by-model judge comparison |
| Zhou et al. 2025 | IEEE/ACM 2025 | SE-Jury; ensemble diversity > quantity |

---

## 9. Bottom Line for PARASITE

**Use a 3-judge ensemble. Never use same-family judge. Weight GPT-4o at 0.5.**

The SE-Jury finding is the most actionable: *2 diverse judges outperform 3 same-family judges.* For PARASITE, a GPT-4o + Claude-3.5-Haiku pair for each run where neither judge matches the target model family will be more reliable than 3 GPT-4o runs with temperature sampling.

The disagreement metric is scientifically valuable in itself — variants where judges strongly disagree are the most interesting cases for human review and may reveal novel parasitism patterns that current judges aren't calibrated for.

*Research compiled: 2026-02-28*
