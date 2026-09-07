# Metaheuristic Co-Optimization of Compression and Runtime Configuration for Embedded LLM Inference: A Case Study on TinyLlama and Raspberry Pi 5

**Fahd Koraiche¹ᵃ\*, Rachid Dehbi¹ᵇ, Amine Dehbi²³ᶜ**
¹Research Laboratory of Data Engineering and Intelligent Systems (LIDSI), Faculty of Science Aïn Chock, Casablanca, Morocco
²Research Laboratory of Mathematics and Computer Science (LTI), Faculty of Sciences Ben M'Sik, Casablanca, Morocco
³High School of Technology-Dakhla, Dakhla, Morocco
ᵃfahd.koraiche@gmail.com · ᵇdehbirac@yahoo.fr · ᶜdehbiamine1@gmail.com

> **Note on this file:** this is the ~10-page conference-length version of the article, cut down from the fuller ~19-page extended/thesis version preserved at `article_draft_extended.md` (LaTeX: `latex/article_extended.tex`). Both versions report the same experiment and the same numbers; this one keeps the essential evidence (all tables, both grid figures, the full best-configurations table, the reconciliation table) but replaces most of the per-variant narrative prose with shorter synthesized paragraphs to fit a hard page budget.

## Abstract

Deploying large language models (LLMs) on resource-constrained edge devices requires navigating a large, mixed discrete-continuous configuration space spanning model compression and inference runtime settings. This paper formulates embedded LLM deployment as a two-phase optimization problem: an exhaustive discrete phase over quantization variants, followed by a continuous/combinatorial metaheuristic search over runtime parameters. We validate three metaheuristics — particle swarm optimization (PSO), grey wolf optimizer (GWO), and differential evolution (DE) — on standard benchmark functions, then apply them, with full hardware-in-the-loop measurement (latency, memory, on-device energy via the target board's power-management IC, and perplexity), to tune TinyLlama-1.1B inference across all seven Pareto-nondominated quantization variants identified by an exhaustive Phase 1 sweep, on a Raspberry Pi 5. Phase 2 surfaces a finding beyond simple algorithm comparison: systematically cross-checking every reported "best" configuration's scalarized cost against an independent re-measurement of that same configuration reveals the **optimizer's curse** — population-based metaheuristics' iterative convergence toward a promising region repeatedly samples nearby noisy real-hardware measurements, so the single lowest cost recorded during a run is a biased, overly optimistic estimator of that configuration's true cost. This effect ranges from negligible (several variants' results reconcile to within 33%) to severe (one variant's best-appearing result is 61x more optimistic than an independent re-measurement supports), and on two of the seven variants it is severe enough that a naive baseline (grid or random search) outright beats every metaheuristic once the noise-corrupted results are excluded. This variance in both outcome and measurement reliability across variants — not a single clean percentage — is the paper's central finding: real per-evaluation cost (21–60s depending on variant, orders of magnitude above the millisecond-scale costs assumed in prior metaheuristic-tuning work) forces a small evaluation budget that is measurably more exposed to this bias than synthetic or simulator-based benchmarks, and reporting requires an explicit, systematic check for it rather than trusting a single recorded minimum at face value.

**Index Terms** — Edge AI, Embedded large language models, Hyperparameter tuning, Metaheuristic optimization, Model compression, Quantization, TinyLlama.

## I. Introduction

Large language models (LLMs) are increasingly deployed outside the datacenter, on phones, microcontrollers, and single-board computers, driven by privacy, latency, connectivity, and cost concerns [17]. Running an LLM on a device such as the Raspberry Pi 5 (8GB RAM, no dedicated GPU) is only feasible after aggressive compression — primarily post-training quantization — and careful runtime configuration of the inference engine (thread count, batch size, context length, KV-cache behavior). Unlike cloud deployment, where extra compute can mask a suboptimal configuration, embedded deployment operates under hard resource ceilings: a configuration that exceeds available RAM fails outright rather than degrading gracefully.

Model compression research (GPTQ [4], AWQ [5], LLM.int8() [6], SqueezeLLM [7]) optimizes a quality/size trade-off independently of the serving runtime and target hardware; systems work on inference engines (e.g., llama.cpp [3]) exposes dozens of runtime flags but leaves their tuning to manual trial-and-error. Neither line of work treats compression choice and runtime configuration as parts of a single search problem evaluated end-to-end on the physical target device. Hyperparameter-optimization metaheuristics — PSO [8], GWO [9], DE [10] — have been shown to outperform naive grid/random search when tuning classical ML models; Nouna et al. [11] demonstrated this for heart disease prediction (PSO/GWO/DE tuning KNN/SVM after wrapper-based feature selection, validated on synthetic functions before the real objective), and that preprocessing→validation→tuning structure is the direct methodological inspiration for this work.

We adapt that structure to a fundamentally different cost regime: in [11], one fitness evaluation costs milliseconds; here, evaluating one candidate runtime configuration can cost seconds to minutes on physical hardware, since it may require loading a multi-hundred-megabyte quantized model and running real generation workloads. This cost asymmetry is the central engineering constraint of our method design (Section III): naively porting a flat metaheuristic search vector from the classical-ML setting is computationally infeasible, and the two-phase discrete/continuous decomposition we propose is what makes hardware-in-the-loop metaheuristic tuning tractable on a Raspberry Pi 5.

**Contributions:** (1) a two-phase search formulation — exhaustive discrete quantization-variant selection followed by metaheuristic runtime tuning — that makes hardware-in-the-loop optimization of embedded LLM inference tractable; (2) a multi-objective evaluation function jointly capturing quality, latency, memory, and energy per token, with hard feasibility constraints enforced by rejection rather than soft penalty; (3) an empirical comparison of PSO, GWO, and DE for this task on physical hardware, with explicit control for thermal throttling and OS jitter; (4) identification and systematic measurement of the **optimizer's curse** as a concrete failure mode of hardware-in-the-loop metaheuristic search, together with a comparison against grid search, random search, and baselines under an equalized evaluation budget.

## II. Related Work

Post-training quantization (GPTQ [4], AWQ [5], LLM.int8() [6], SqueezeLLM [7]) reduces model size and memory bandwidth at some quality cost; TinyLlama [1] is trained from scratch at 1.1B parameters specifically to be inference-efficient, making it a natural target for the runtime-side compression this paper studies. llama.cpp [3] enables CPU-only inference on commodity edge hardware via GGUF quantization (Q2_K–Q8_0, Section III-B); NPU-offloaded inference [18] is a complementary hardware-acceleration path to this paper's CPU-only approach. Edge-LLM survey work catalogs the broader deployment landscape [17].

Grid and random search scale poorly with dimensionality; PSO [8], GWO [9], and DE [10] have been applied to tune classical ML models, most directly by [11] (Section I). On the quantization-search side, HAQ [12] uses reinforcement learning (not a population-based metaheuristic) with a hardware *simulator* to search per-layer CNN bitwidths; genetic-algorithm [13], evolving-proxy [14], and bare-bones-PSO [15] methods search CNN quantization directly but target CNNs, search quantization alone, and evaluate via simulators/GPUs rather than a physical low-power board. LLM-guided PSO [16] uses an LLM *inside* the optimizer to tune conventional DL hyperparameters — a different problem from this paper's PSO/GWO/DE tuning an LLM's own deployment configuration.

**The closest prior art is LLMPi** [19] (physical-Raspberry-Pi LLM quantization measurement, but fixed quantization levels, no metaheuristic search, no runtime-configuration search) and **DeepQGHO** [20] (physical Jetson Nano, joint hyperparameter search + quantization, but greedy coordinate descent on training-time CNN hyperparameters, quantized only after search completes). Two 2026 studies narrow the gap further without closing it: Embedded Arena [21] co-optimizes MCU model/firmware via an LLM-agent (not population-based) on vision/audio targets (not LLM inference); Sadheerthan et al. [22] use PSO to design a TinyLlama softmax approximation measured on a physical NPU, but search one kernel's design rather than compression plus serving-runtime configuration, without end-to-end LLM latency. Based on four targeted search passes, no located work combines (a) a population-based metaheuristic, (b) joint LLM compression and runtime-configuration search, (c) physical low-power hardware-in-the-loop measurement, and (d) LLM inference as the target; this is an evidence-based gap, not an unqualified "first" claim (Section VI).

## III. Method

### A. Two-Phase Decomposition

We decompose the search space into two tiers, mirroring [11]'s preprocessing/feature-selection → hyperparameter-tuning structure (Fig. 1): **Phase 1** (discrete, exhaustive) selects a quantization variant from a small catalog; **Phase 2** (metaheuristic) tunes continuous/combinatorial runtime parameters for the chosen variant using PSO, GWO, and DE. This split exists because quantization variants are distinct requantized model artifacts, not continuously interpolable, while runtime parameters can be varied on a fixed model file at near-zero marginal cost — flattening both into one search vector would force the population to repeatedly reload and re-benchmark full model files, infeasible within a reasonable Pi 5 experiment budget (Section III-D).

*Figure 1: Overall two-phase pipeline — exhaustive Phase 1 sweep over quantization variants, followed by Phase 2's metaheuristic runtime-parameter search, both evaluated on-device against the joint objective (Section III-C).* (`results/figures/pipeline_diagram.png`)

### B. Target Platform, Benchmark, and Search Spaces

Target hardware is a Raspberry Pi 5 (8GB RAM, Broadcom BCM2712, 4x Cortex-A76, active PWM-fan cooling), Debian 13, llama.cpp built from source, CPU governor fixed to `performance`. Model is TinyLlama-1.1B-Chat-v1.0 [1]; no pre-converted F16 GGUF was available, so Q8_0 is used as the requantization source for all Phase 1 variants (a documented deviation, not true F16). Latency/throughput uses `llama-bench` (512 prompt / 128 generation tokens, 3 repetitions for Phase 1; Phase 2 varies prompt tokens directly as the searched n_ctx); quality uses `llama-perplexity` over WikiText-2.

Phase 1 candidates are the GGUF variants Q2_K, Q3_K_M, Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0; each is evaluated once for perplexity, peak RSS, tokens/second, and file size (Table III), and all variants not Pareto-dominated on (quality, memory, speed) are retained for Phase 2 — no fixed top-k cap.

Phase 2 searches a 4-dimensional mixed integer/categorical vector per retained variant (Table I), encoded as a continuous box in ℝ⁴ so PSO/GWO/DE never need to know the space is anything but continuous. n_ctx's upper bound was capped at 1024 (not an originally planned 2048) after direct measurement showed larger prompt sizes cost real additional minutes per evaluation. Temperature and top-p were dropped from the search: they only affect quality through sampling randomness, which perplexity cannot distinguish.

**Table I: Phase 2 runtime hyperparameter search space**

| Parameter | Type | Range | Notes |
|---|---|---|---|
| Threads | Integer | [1,4] | Pi 5, 4 Cortex-A76 cores |
| Batch size (n_batch) | Integer | [32,512] | `-b` flag |
| Context length (n_ctx) | Integer | [128,1024] | via `-p` |
| KV-cache type | Categorical | {f16, q8_0, q4_0} | `-ctk`/`-ctv` |

### C. Objective Function

We minimize a scalarized, constrained multi-objective function:

**f(x) = α·L̂(x) + β·(1−Q̂(x)) + γ·M̂(x) + δ·Ê(x) + Penalty(x)**  (1)

where L̂, Q̂, M̂, Ê are min–max normalized latency, quality, peak memory, and energy per token, and Penalty(x) = +∞ (hard reject, not a finite soft penalty) if peak RSS exceeds the RAM ceiling or the process crashes/OOMs/times out — a configuration that will actually OOM on this hardware must never be numerically outweighed by a good quality score. Weights α=0.35, β=0.35, γ=0.2, δ=0.1 are illustrative defaults, not tuned; Section IV-D's Grid-vs-DE energy/latency split on Q4_K_M shows the scalarization is doing real, non-trivial work, since different weights could plausibly favor Grid's low-energy point over DE's lower overall cost.

### D. Metaheuristics and Evaluation-Cost Argument

We use PSO [8], GWO [9], and DE [10] (Table II gives parameters), unchanged mechanically from [11]'s classical-ML setting — what changes is the cost of the fitness function. In [11], one evaluation (fit + cross-validate a KNN/SVM) costs milliseconds, permitting population×iteration budgets of thousands of evaluations. On the Pi 5, one Phase-2 evaluation measured 21–60s depending on variant — three to four orders of magnitude slower. A flat search at [11]'s budget would require 17–33 hours of device time for a *single* algorithm on a *single* variant; this is why Phase 2 uses a deliberately time-boxed, per-variant-adaptive budget (population 4, iterations 5–8, 2 runs — an evaluation floor chosen to keep each algorithm within a 12-minute time box per variant, confirmed against real per-variant totals in Section IV-C).

Before trusting them on this real, noisy objective, PSO/GWO/DE are validated on three closed-form benchmark functions following [11]'s protocol (population 20, 200 iterations, 10 runs): Sphere (dim 10, unimodal), Goldstein–Price (dim 2, multimodal, global min. 3), and FindValues (dim 5, asymmetric-bounds squared-error function, global min. 0) — the closest synthetic proxy to the real Phase 2 objective. Grid search, random search, PSO, GWO, and DE are also compared on these functions at a matched budget of 4,000 evaluations, establishing the baseline implementations are correct before the real hardware experiment.

**Table II: Metaheuristic parameters for Phase 2 (real hardware run)**

| Algorithm | Pop. | Iter. | Runs | Key parameters |
|---|---|---|---|---|
| PSO | 4 | 5–8 | 2 | w=1.0, w_damp=0.99, c1=c2=2.0 |
| GWO | 4 | 5–8 | 2 | a linear decay 2→0 |
| DE | 4 | 5–8 | 2 | β_min=0.2, β_max=0.9, p_CR=0.1 |

### E. Measurement and Baselines

Latency, quality, memory, and energy are all measured on physical hardware rather than simulated. Energy is measured via the Pi 5's on-board PMIC (`vcgencmd pmic_read_adc`, not external instrumentation): total power is the sum of V×I across all reported rails, sampled concurrently with each benchmark (1.0s interval) and integrated over wall-clock generation time. Thermal throttling (`vcgencmd get_throttled` bits set, SoC ≈83–84.5°C under sustained load) and OS/background jitter were observed but not actively mitigated in the reported run (`bench_repetitions=1`, one measurement per configuration, not repetition-averaged) — both are disclosed as limitations (Section VI) rather than corrected for after the fact. Grid search and random search, evaluation-budget-matched to the metaheuristics, are implemented as baselines to directly test whether metaheuristics outperform naive search under an equal budget — a claim that must not be assumed without a control.

## IV. Results and Discussion

### A. Phase 1: Quantization Variant Impact

**Table III: Phase 1 quantization variant impact (real, Pi 5)**

| Variant | PPL | Peak RSS (MB) | Gen tok/s | Prompt tok/s | Size (MB) |
|---|---|---|---|---|---|
| Q2_K | 28.8148 | 495.4 | 25.62 | 34.15 | 413.0 |
| Q3_K_M | 21.6011 | 793.4 | 19.18 | 34.47 | 523.9 |
| Q4_0 | 20.6162 | 1197.7 | 18.06 | 116.85 | 608.2 |
| Q4_K_M | 20.4490 | 1257.2 | 17.02 | 70.59 | 637.8 |
| Q5_K_M | 20.1640 | 1465.9 | 14.38 | 60.73 | 746.7 |
| Q6_K | 20.0001 | 1687.6 | 12.26 | 61.31 | 862.5 |
| Q8_0 | 19.9734 | 2180.6 | 8.47 | 54.97 | 1116.5 |

**All 7 variants are Pareto-nondominated, and that is itself a finding**: quality, speed, and memory trade off perfectly monotonically with quantization level (every step toward Q8_0 strictly improves perplexity while strictly increasing memory and decreasing throughput, zero exceptions), so no variant beats another on every axis simultaneously. Marginal quality return shrinks sharply past Q4_K_M (20.62→19.97 PPL, a 3.2% gain, while RSS nearly doubles and speed drops by more than half) — a poor trade for most deployment goals. Because each Phase 2 evaluation costs real seconds (Section III-D), the real search began on a 3-variant subset (Q2_K, Q4_K_M, Q8_0) and was extended to the remaining 4 (Q3_K_M, Q4_0, Q5_K_M, Q6_K) once the pipeline proved stable across an unattended overnight run; all 7 are complete below.

### B. Metaheuristic Validation and Baseline Rehearsal (Synthetic)

All three optimizers reach values effectively indistinguishable from each function's known global minimum in their best run, confirming correct implementations before trusting them on the real objective. The comparative pattern echoes [11]: PSO is uniformly strong and low-variance across all three functions; GWO is the most fragile on the multimodal/asymmetric problems (its FindValues median is over 30x worse than its best run, plateauing around iteration 30 while PSO/DE continue descending for 150+ more — consistent with premature convergence once its alpha/beta/delta leaders cluster); DE is competitive but slower to fully converge on Sphere and FindValues. This premature-convergence pattern reproduces on the real Phase 2 objective for some variants but not others (Section IV-C).

At a matched budget of 4,000 evaluations, grid search collapses on Sphere (dim 10): a shared-resolution grid can only afford 2 points per axis, so no grid point can land near the interior optimum (best cost 1000, exactly the corner-stuck value); random search does far better (median 50.63) but still loses by 8–11 orders of magnitude to any metaheuristic. On FindValues (dim 5, asymmetric bounds — the closest synthetic proxy to the real Phase 2 space), grid search's best (4554) is worse than random search's *median* (1523) — a direct empirical warning against grid search on Phase 2's real, similarly asymmetric parameter space.

### C. Phase 2: Convergence and Robustness

Figure 2 plots median best-cost-so-far (2 runs) for PSO/GWO/DE on the real hardware-measured objective across all 7 variants; Figure 3 and Table IV give per-run spread. Patterns vary substantially by variant: on Q4_K_M and Q3_K_M, GWO/DE respectively plateau early (the same premature-convergence behavior seen on synthetic functions); on Q2_K and Q8_0, GWO instead converges *later* but to the lowest cost, so this behavior is variant-dependent, not fixed. Q4_0's much lower per-evaluation cost (20.7s) let the adaptive budget allocate 8 iterations instead of the 5-iteration floor, and all three algorithms show genuine gradual descent rather than early plateaus — this is also, as Section IV-D shows, the variant with the most trustworthy recorded costs. Run-to-run spread (Table IV) is large and inconsistent across variants (1.03x–99x), including one striking inversion of the synthetic-validation expectation: on Q4_K_M, GWO — flagged as fragile on synthetic functions — is the *most* consistent algorithm here (1.03x spread) while PSO and DE swing 4–5x. With only N_runs=2, this spread is the gap between two samples, not a statistically grounded estimate (Section VI).

*Figure 2: Real Phase 2 convergence, all 7 completed variants (median of 2 runs).* (`results/figures/real_phase2_convergence_*.png`)

*Figure 3: Real Phase 2 robustness, all 7 completed variants (2 runs each).* (`results/figures/real_phase2_boxplot_*.png`)

**Table IV: Per-run best cost and run-to-run spread, all 7 variants**

| Variant | PSO Run 1 | PSO Run 2 | PSO Spread | GWO Run 1 | GWO Run 2 | GWO Spread | DE Run 1 | DE Run 2 | DE Spread |
|---|---|---|---|---|---|---|---|---|---|
| Q4_K_M | 0.0350 | 0.1414 | 4.0x | 0.1094 | 0.1066 | 1.03x | **0.0205** | 0.1000 | 4.9x |
| Q2_K | 0.0247 | 0.0729 | 3.0x | **0.0000** | 0.0477 | n/a (floor) | 0.0189 | 0.0658 | 3.5x |
| Q8_0 | 0.0497 | 0.1042 | 2.1x | **0.0010** | 0.0988 | 99x | 0.1144 | 0.0755 | 1.5x |
| Q3_K_M | 0.1231 | 0.0802 | 1.5x | 0.0210 | 0.0685 | 3.3x | **0.0026** | 0.0864 | 33x |
| Q4_0 | 0.0667 | 0.1019 | 1.5x | 0.0120 | 0.1536 | 12.8x | 0.1262 | 0.1043 | 1.2x |
| Q5_K_M | 0.0878 | 0.1312 | 1.5x | **0.0073** | 0.1477 | 20.2x | 0.0680 | 0.3491 | 5.1x |
| Q6_K | 0.0198 | 0.1011 | 5.1x | **0.0168** | 0.1048 | 6.2x | 0.0467 | 0.1638 | 3.5x |

### D. The Optimizer's Curse: Best Configurations and Reconciliation

Q2_K's exact-zero GWO cost (Table IV) and Q8_0's near-zero GWO cost prompted a systematic check, run across all 7 variants: recompute each method's scalarized cost from its own reported raw metrics (`best_configs.csv`, run through the objective's actual weights and calibrated ranges) and compare against the recorded search-time cost (Table V).

**Table V: Reconstructed-cost / recorded-cost ratio, all 7 variants**

| Variant | PSO | GWO | DE | Random | Grid |
|---|---|---|---|---|---|
| Q2_K | 0.00x (clipped) | inf (clipped) | 0.00x (clipped) | 1.01x | inf (clipped) |
| Q3_K_M | 2.89x | 8.45x | 61.51x | 2.60x | 0.93x |
| Q4_0 | 0.89x | 1.16x | 1.02x | 1.33x | 1.00x |
| Q4_K_M | 6.06x | 1.98x | 7.01x | 0.96x | 0.80x |
| Q5_K_M | 1.40x | 22.50x | 1.54x | 2.91x | 1.32x |
| Q6_K | 0.72x | 1.12x | 0.00x (clipped) | 0.20x | 1.33x |
| Q8_0 | 1.14x | 179.89x | 3.16x | 2.33x | 8.20x |

The pattern is consistent with the **optimizer's curse**: a population-based metaheuristic's iterative convergence toward a promising region repeatedly samples nearby-but-distinct configurations there, and the single lowest cost observed across those noisy real-hardware evaluations is a biased, overly optimistic estimator of that configuration's *true* cost. Grid and Random search are far less exposed (ratios mostly near 1x) because they do not iteratively concentrate sampling around a converging optimum — every ratio above 8x anywhere in this dataset belongs to a metaheuristic, never to Grid or Random. The recorded cost remains a fair basis for ranking methods against each other within a variant (every method is subject to the same evaluation regime), but the raw-metric columns come from a separate, later re-evaluation and should be read as illustrative of the plausible trade-off space, not confirmed causal fact.

Table VI gives the best configuration found by each method for every variant with its full raw re-measurement. **Bold** marks this paper's best-supported recommendation per variant; † marks a row excluded because its recorded cost does not reconcile with its own raw metrics. Two variants (Q4_0, Q6_K) give fully trustworthy, mechanistically interpretable metaheuristic wins (GWO, ratios 1.16x/1.12x): GWO's Q4_0 win comes from threads=2 (not the maximum 4) with low memory and energy, while Grid's threads=1 corner is unambiguously worst. Two variants invert the paper's central motivating claim: on **Q5_K_M**, excluding GWO's unreliable 0.0073, Random search (0.0157) is the lowest recorded cost, beating every metaheuristic; on **Q8_0**, excluding GWO's unreliable 0.0010, Grid search (0.0239) beats two of the three metaheuristics outright — a genuinely uncomfortable result for this paper's motivating claim, and the largest, most expensive variant, where Grid's minimal corner is disproportionately attractive. **Q3_K_M** shows the sharpest single warning: DE's recorded win (0.0026, the lowest cost anywhere in the dataset) reconstructs to ≈0.159 from its own raw metrics — a 61.5x discrepancy, the least trustworthy number in the entire study, with Grid's reconcilable 0.1596 arguably a more honest "known" result for that variant. **Q2_K** shows all of PSO/GWO/DE/Grid reconstructing to a clipped ~0 cost from a calibration-sample-size artifact (N_calib=8, Section VI); only Random's number is trustworthy, and it is also the worst of the five.

**Table VI: Best configurations found, all 7 variants (raw re-measurement of each method's search-time winner)**

| Variant | Method | Threads | Batch | Ctx | KV | f(x) | Latency (ms/tok) | Quality (−PPL) | Peak RSS (MB) | Energy (J/tok) |
|---|---|---|---|---|---|---|---|---|---|---|
| Q2_K | **GWO** | 4 | 153 | 128 | f16 | **0.0000** | 42.99 | −31.3272 | 468.8 | 1.257 |
| Q2_K | Grid | 4 | 512 | 128 | f16 | 0.0000 | 44.83 | −31.3272 | 468.0 | 1.193 |
| Q2_K | DE | 4 | 167 | 387 | f16 | 0.0189 | 42.83 | −31.3272 | 467.9 | 3.052 |
| Q2_K | PSO | 4 | 221 | 409 | f16 | 0.0247 | 42.54 | −31.3272 | 468.9 | 3.173 |
| Q2_K | Random | 4 | 318 | 387 | f16 | 0.0482 | 44.01 | −31.3272 | 476.9 | 3.158 |
| Q3_K_M | **DE** | 4 | 228 | 128 | f16 | **0.0026** | 49.27 | −23.2585 | 785.7 | 1.115 |
| Q3_K_M | GWO | 4 | 244 | 465 | f16 | 0.0210 | 49.77 | −23.2585 | 785.3 | 3.269 |
| Q3_K_M | PSO | 4 | 205 | 831 | q8_0 | 0.0802 | 49.61 | −23.2639 | 784.2 | 6.250 |
| Q3_K_M | Random | 4 | 273 | 819 | q8_0 | 0.1049 | 51.07 | −23.2639 | 784.6 | 6.308 |
| Q3_K_M | Grid | 4 | 32 | 128 | f16 | 0.1596 | 49.12 | −23.2585 | 784.5 | 1.138 |
| Q4_0 | **GWO** | 2 | 179 | 294 | q8_0 | **0.0120** | 48.83 | −22.3364 | 1159.5 | 0.964 |
| Q4_0 | Random | 2 | 181 | 182 | f16 | 0.0157 | 48.57 | −22.3562 | 1160.6 | 0.663 |
| Q4_0 | PSO | 2 | 299 | 380 | q8_0 | 0.0667 | 48.70 | −22.3364 | 1170.0 | 1.238 |
| Q4_0 | DE | 2 | 47 | 930 | q8_0 | 0.1043 | 48.62 | −22.3364 | 1154.1 | 3.564 |
| Q4_0 | Grid | 1 | 32 | 128 | f16 | 0.3098 | 57.03 | −22.3562 | 1146.4 | 0.718 |
| Q4_K_M | **DE** | 3 | 318 | 387 | f16 | **0.0205** | 53.86 | −22.3879 | 1248.5 | 1.639 |
| Q4_K_M | PSO | 3 | 308 | 567 | q8_0 | 0.0350 | 54.61 | −22.3722 | 1252.5 | 2.689 |
| Q4_K_M | GWO | 3 | 217 | 764 | q8_0 | 0.1066 | 54.97 | −22.3722 | 1249.6 | 3.568 |
| Q4_K_M | Random | 3 | 311 | 663 | f16 | 0.1238 | 54.65 | −22.3879 | 1243.7 | 2.725 |
| Q4_K_M | Grid | 4 | 512 | 128 | f16 | 0.3311 | 63.02 | −22.3879 | 1239.5 | 0.805 |
| Q5_K_M | GWO† | 3 | 313 | 323 | f16 | *0.0073* | 64.19 | −21.9608 | 1451.0 | 1.731 |
| Q5_K_M | DE | 3 | 167 | 387 | f16 | 0.0680 | 63.17 | −21.9608 | 1450.8 | 2.086 |
| Q5_K_M | PSO | 3 | 310 | 697 | q8_0 | 0.0878 | 62.52 | −21.9742 | 1449.2 | 3.753 |
| Q5_K_M | **Random** | 3 | 66 | 419 | f16 | **0.0157** | 62.30 | −21.9608 | 1449.6 | 2.232 |
| Q5_K_M | Grid | 4 | 32 | 128 | f16 | 0.2486 | 67.22 | −21.9608 | 1450.4 | 0.936 |
| Q6_K | **GWO** | 3 | 202 | 359 | f16 | **0.0168** | 72.96 | −21.7199 | 1669.2 | 2.076 |
| Q6_K | PSO | 3 | 309 | 455 | f16 | 0.0198 | 72.92 | −21.7199 | 1668.1 | 2.382 |
| Q6_K | Random | 3 | 66 | 419 | f16 | 0.0371 | 72.87 | −21.7199 | 1667.0 | 2.355 |
| Q6_K | DE | 3 | 48 | 387 | f16 | 0.0467 | 72.65 | −21.7199 | 1664.6 | 2.059 |
| Q6_K | Grid | 4 | 32 | 128 | f16 | 0.2455 | 77.80 | −21.7199 | 1671.1 | 1.163 |
| Q8_0 | GWO† | 2 | 108 | 128 | f16 | *0.0010* | 85.86 | −21.7013 | 2179.6 | 0.841 |
| Q8_0 | **Grid** | 1 | 32 | 128 | f16 | **0.0239** | 87.51 | −21.7013 | 2179.3 | 0.987 |
| Q8_0 | Random | 2 | 95 | 688 | f16 | 0.0340 | 86.04 | −21.7013 | 2164.8 | 2.916 |
| Q8_0 | PSO | 2 | 279 | 797 | f16 | 0.0497 | 86.08 | −21.7013 | 2161.0 | 3.218 |
| Q8_0 | DE | 2 | 47 | 867 | f16 | 0.0755 | 85.75 | −21.7013 | 2177.9 | 4.009 |

### E. Discussion

**Table VII: Cross-variant summary**

| Variant | Best trustworthy method | Margin over grid | Reconciliation |
|---|---|---|---|
| Q4_K_M | DE (ratio 7.0x, weak) | 16.1x recorded, not fully trustworthy | poor |
| Q2_K | unresolved | not numerically resolvable | poor (4/5 clipped) |
| Q3_K_M | DE recorded win, ratio 61.5x (worst) | ~61x recorded; do not trust | very poor |
| Q4_0 | GWO, ratio 1.16x | 25.8x, trustworthy | **excellent** |
| Q5_K_M | Random search wins over all metaheuristics | — (baseline wins) | Random/DE/Grid fair; GWO 22.5x |
| Q6_K | GWO, ratio 1.12x | 14.6x, trustworthy | good |
| Q8_0 | Grid itself wins | — | Grid fair; GWO 180x |

**The paper's central empirical finding, supported by all 7 variants, is not "how much do metaheuristics beat baselines" but "how reliably can this be measured at all"** — and the answer depends heavily on per-variant noise conditions that the two-phase design surfaces rather than averages away. Two variants give fully trustworthy metaheuristic wins; two see a naive baseline outright beat every metaheuristic once unreliable numbers are excluded; the remaining three range from partially confounded to actively misleading if taken at face value. This is, to our knowledge, the first concrete, measured account of the optimizer's curse as a failure mode specific to hardware-in-the-loop metaheuristic-tuned LLM deployment, rather than a generic risk asserted in the abstract. The mechanistic reason traces back to Section III-D's cost argument: real hardware evaluation is expensive, so Phase 2's necessarily small budget is far more exposed to per-variant noise, calibration undersampling, and the optimizer's curse than the millisecond-scale, thousands-of-evaluations budget used in synthetic validation or in [11]. A single quantization variant chosen up front, without the two-phase structure, would have reported only one of seven qualitatively different outcomes as if it were the general finding — the two-phase decomposition's full 7-variant coverage is what makes this cross-variant picture visible at all.

## VI. Limitations and Threats to Validity

- **Single model family and hardware target**: TinyLlama-1.1B on Pi 5 only; generalization untested.
- **Scalarized objective with fixed, untuned weights** collapses a genuine multi-objective trade-off into one number; Section IV-D's Q4_K_M split (Grid wins on energy alone but loses overall) shows this is doing real work, not cosmetic aggregation. A Pareto-front analysis would be more informative but more expensive.
- **Small calibration sample** (N_calib=8) measurably changed a result: on Q2_K, two independent optima landed outside the calibrated min/max window and clipped to an exact cost of 0.0000 — a normalization artifact, not a true global optimum, that directly narrowed the apparent margin on that variant.
- **The optimizer's curse is general and pervasive**, not an isolated anomaly: `bench_repetitions=1` means each evaluation is one noisy hardware measurement; reconstructing costs from raw metrics (Section IV-D) disagrees with the recorded cost by up to 180x, and every ratio above 8x belongs to a metaheuristic, never Grid/Random. A concrete fix — `bench_repetitions`>1 with an automated consistency check before reporting any "winner" — is the top priority in future work.
- **N_runs reduced from 10 to 2** for real Phase 2: reported spread is the gap between two samples, not a statistically grounded estimate.
- **Hardware measurement noise** (thermal throttling at 83–84.5°C) may dominate small differences between top configurations; a mid-run Pi crash, likely thermal, lost no completed data but exposed the runner's lack of checkpoint/resume capability.
- **Quality metric is perplexity, not task accuracy**, which is why temperature/top-p were dropped from the search space; a downstream task benchmark might rank configurations differently.
- **Novelty claim** rests on four targeted web-search passes, not a full database-backed citation-graph review; a Google Scholar/Semantic Scholar/Scopus traversal from [19,20,21,22] could still surface closer prior art.

## VII. Conclusion and Future Work

This paper set out to answer whether a two-phase decomposition — exhaustive discrete quantization selection followed by metaheuristic runtime tuning — is a feasible, worthwhile way to structure edge-LLM configuration search on physical hardware, and how PSO/GWO/DE compare for this task under real hardware-in-the-loop measurement rather than synthetic validation. Phase 1's exhaustive sweep found all 7 candidate variants Pareto-nondominated, and Phase 2's real search across all 7 retained variants showed that the best achievable configuration, its cost, and even *how reliably that cost can be trusted at all* differ substantially by variant. A systematic reconciliation check — recomputing every reported "best" configuration's cost from an independent re-measurement — revealed the **optimizer's curse**: population-based metaheuristics' iterative convergence repeatedly samples near a promising region, so their reported best is a biased, overly optimistic estimator of true cost, far more so than grid or random search's more spread-out sampling. This directly explains why a naive baseline outright wins on 2 of 7 variants and why one metaheuristic's apparent 61x margin should not be trusted: not because metaheuristics are worse optimizers, but because their own search dynamics expose their reported bests to a specific, nameable, now-measured statistical bias that cheap synthetic validation never surfaces. To our knowledge, this is the first time this failure mode has been concretely measured for metaheuristic-tuned LLM deployment specifically.

Future work, in priority order: (1) `bench_repetitions`>1 with an automated recorded-vs-reconstructed consistency check built into the pipeline, plus a larger calibration sample, directly motivated by the scale of the optimizer's-curse finding; (2) checkpoint/resume capability for the hardware-in-the-loop runner, motivated by the mid-run Pi crash; (3) multi-objective Pareto-front search (NSGA-II/MOEA-D) instead of scalarization; (4) a downstream task-accuracy quality metric, reopening the temperature/top-p dimensions dropped here; (5) extension to other edge platforms and small models to test whether the variant-to-variant instability found here is hardware-specific or general; (6) hybrid metaheuristics and additional Phase 1 compression axes (structured pruning, LoRA-merged distillation).

## Data Availability

Code, configuration, and all raw measurement data (Phase 1 sweep results, Phase 2 convergence/robustness/best-configuration CSVs, and figures) referenced in this paper, including full repository structure and pseudocode, are maintained at https://github.com/Koraiche/phd-hypetune-service, currently a private repository that will be made public upon publication. The WikiText-2 corpus used for perplexity measurement is a standard, separately-licensed public dataset and is not redistributed in this repository.

## References

[1] P. Zhang, G. Zeng, T. Wang, and W. Lu, "TinyLlama: An Open-Source Small Language Model," arXiv:2401.02385, 2024.

[2] ggml-org, "GGUF File Format Specification," ggml [GitHub repository], https://github.com/ggml-org/ggml/blob/master/docs/gguf.md.

[3] G. Gerganov et al., llama.cpp [Software]. https://github.com/ggml-org/llama.cpp.

[4] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers," arXiv:2210.17323, 2022.

[5] J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G. Xiao, X. Dang, C. Gan, and S. Han, "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration," arXiv:2306.00978, 2023.

[6] T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale," arXiv:2208.07339, 2022.

[7] S. Kim, C. Hooper, A. Gholami, Z. Dong, X. Li, S. Shen, M. W. Mahoney, and K. Keutzer, "SqueezeLLM: Dense-and-Sparse Quantization," arXiv:2306.07629, 2023.

[8] J. Kennedy and R. Eberhart, "Particle swarm optimization," in Proc. ICNN'95 – Int. Conf. Neural Networks, vol. 4, 1995, pp. 1942–1948.

[9] S. Mirjalili, S. M. Mirjalili, and A. Lewis, "Grey wolf optimizer," Advances in Engineering Software, vol. 69, pp. 46–61, 2014.

[10] R. Storn and K. Price, "Differential evolution – a simple and efficient heuristic for global optimization over continuous spaces," Journal of Global Optimization, vol. 11, no. 4, pp. 341–359, 1997.

[11] Z. Nouna, H. Bouyghf, M. Nahid, and I. Sabiri, "Heart disease prediction optimization using metaheuristic algorithms," IAES International Journal of Artificial Intelligence, vol. 14, no. 5, pp. 4332–4341, 2025.

[12] K. Wang, Z. Liu, Y. Lin, J. Lin, and S. Han, "HAQ: Hardware-Aware Automated Quantization with Mixed Precision," in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2019, pp. 8612–8620, doi: 10.1109/CVPR.2019.00881.

[13] J. B. P. Matos Jr., I. Bessa, E. Manino, X. Song, and L. C. Cordeiro, "CEG4N: Counter-Example Guided Neural Network Quantization Refinement," in Proc. Int. Workshop on Numerical Software Verification / Formal Methods for ML-Enabled Autonomous Systems (NSV/FoMLAS), Springer LNCS vol. 13466, 2022, pp. 29–45, doi: 10.1007/978-3-031-21222-2_3.

[14] P. Dong, L. Li, Z. Wei, X.-Y. Niu, Z. Tian, and H. Pan, "EMQ: Evolving Training-free Proxies for Automated Mixed Precision Quantization," in Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV), 2023.

[15] J. Tmamna, E. Ben Ayed, and M. Ben Ayed, "Automatic Quantization of Convolutional Neural Networks Based on Enhanced Bare-Bones Particle Swarm Optimization for Chest X-Ray Image Classification," in Computational Collective Intelligence (ICCCI 2023), Springer, 2023, pp. 125–137, doi: 10.1007/978-3-031-41456-5_10.

[16] S. Hameed, B. Qolomany, S. B. Belhaouari, M. Abdallah, J. Qadir, and A. Al-Fuqaha, "Large Language Model Enhanced Particle Swarm Optimization for Hyperparameter Tuning for Deep Learning Models," arXiv:2504.14126, 2025.

[17] Y. Zheng, Y. Chen, B. Qian, X. Shi, Y. Shu, and J. Chen, "A Review on Edge Large Language Models: Design, Execution, and Applications," ACM Computing Surveys, vol. 57, no. 8, Art. 209, Mar. 2025.

[18] D. Xu, H. Zhang, L. Yang, R. Liu, G. Huang, M. Xu, and X. Liu, "Fast On-device LLM Inference with NPUs," in Proc. 30th ACM Int. Conf. on Architectural Support for Programming Languages and Operating Systems (ASPLOS), 2025.

[19] M. Ardakani, J. Malekar, and R. Zand, "LLMPi: Optimizing LLMs for High-Throughput on Raspberry Pi," in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW), EDGE Workshop, 2025, pp. 6379–6388.

[20] A. A. Chowdhury, M. A. Hossen, M. A. Azam, and M. H. Rahman, "DeepQGHO: Quantized Greedy Hyperparameter Optimization in Deep Neural Networks for On-the-Fly Learning," IEEE Access, vol. 10, pp. 6407–6416, 2022.

[21] Z. Zhang, A. Le Metzger, J. Lyu, C.-C. Chang, J. Shao, Y. Liu, E. A. Mensah, E. Wang, K. Heimerl, G. D. Abowd, S. Patel, N. Jaques, and V. Iyer, "Embedded Arena: Iterative Optimization via Hardware Feedback," arXiv:2606.16190, 2026.

[22] S. Sadheerthan, M.-J. Hsu, C.-H. Huang, and Y.-T. Wang, "Attention Distribution-Aware Softmax for NPU-Accelerated On-Device Inference of LLMs: An Edge-Oriented Approximation Design," Electronics, vol. 15, no. 6, Art. 1312, 2026, doi: 10.3390/electronics15061312.
