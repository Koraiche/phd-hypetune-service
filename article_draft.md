# Metaheuristic Co-Optimization of Compression and Runtime Configuration for Embedded LLM Inference: A Case Study on TinyLlama and Raspberry Pi 5

**Status: DRAFT v0.1 — structure and narrative complete, experimental results pending.**
All numeric results, figures, and tables below are marked `[PLACEHOLDER]` and must be filled in once the code in `hyptune/` produces real measurements. Do not submit with placeholders unresolved.

---

## Abstract

`[PLACEHOLDER — write last, ~200 words]`

Deploying large language models (LLMs) on resource-constrained edge devices requires navigating a large, mixed discrete-continuous configuration space spanning model compression and inference runtime settings. This paper formulates embedded LLM deployment as a two-phase optimization problem: an exhaustive discrete phase over quantization variants, followed by a continuous/combinatorial metaheuristic search over runtime parameters. We validate three metaheuristics — particle swarm optimization (PSO), grey wolf optimizer (GWO), and differential evolution (DE) — on standard benchmark functions, then apply them to tune TinyLlama-1.1B inference on a Raspberry Pi 5 under a joint objective balancing output quality, latency, memory footprint, and energy consumption. Results show `[PLACEHOLDER: headline result, e.g., "a X% latency reduction at equal perplexity versus the default configuration, and Y% versus grid search under equal evaluation budget"]`. This work contributes an empirically grounded, hardware-in-the-loop methodology for systematic edge LLM configuration search, extending metaheuristic hyperparameter optimization — previously applied to classical ML models — to the embedded LLM setting.

**Keywords:** Edge AI; Embedded large language models; Hyperparameter tuning; Metaheuristic optimization; Model compression; Quantization; TinyLlama

---

## 1. Introduction

Large language models (LLMs) are increasingly deployed outside the datacenter, on phones, microcontrollers, and single-board computers, driven by privacy, latency, connectivity, and cost concerns [VERIFY: cite edge-LLM survey]. Running an LLM on a device such as the Raspberry Pi 5 (8 GB RAM, no dedicated GPU) is only feasible after aggressive compression — primarily post-training quantization — and careful runtime configuration of the inference engine (thread count, batch size, context length, KV-cache behavior). Unlike cloud deployment, where extra compute can mask a suboptimal configuration, embedded deployment operates under hard resource ceilings: a configuration that exceeds available RAM does not degrade gracefully, it fails outright.

Two bodies of prior work address pieces of this problem in isolation. Model compression research (quantization methods such as GPTQ [4], AWQ [5], LLM.int8() [6], and mixed-precision schemes such as SqueezeLLM [7]) optimizes for a quality/size trade-off but is typically evaluated independently of the serving runtime and target hardware. Systems work on inference engines (e.g., llama.cpp [3]) exposes dozens of runtime flags but leaves their tuning to manual trial-and-error or vendor-suggested defaults. Neither line of work treats compression choice and runtime configuration as parts of a single search problem evaluated end-to-end on the physical target device.

Hyperparameter optimization (HPO) research offers a template for closing this gap. Grid and random search provide robust baselines but scale poorly with the number of parameters [7-mirrored], while metaheuristics — PSO [8], GWO [9], DE [10] — have been shown to outperform naive search when tuning classical machine learning models. Nouna et al. [11] recently demonstrated this for heart disease prediction, using PSO, GWO, and DE to tune KNN and SVM hyperparameters after a preprocessing and wrapper-based feature-selection stage, validating each metaheuristic on known benchmark functions before applying it to the real objective (cross-validated prediction error). That paper's structure — preprocessing/feature-selection as a discrete reduction step, metaheuristic validation on synthetic functions, then metaheuristic tuning of the real model — is the direct methodological inspiration for this work.

We adapt that structure to a fundamentally different cost regime. In [11], evaluating one candidate hyperparameter setting costs milliseconds (a KNN or SVM fit). In embedded LLM deployment, evaluating one candidate can cost seconds to minutes, because it may require loading a multi-hundred-megabyte quantized model file from disk and running real generation workloads on physical hardware. This cost asymmetry is the central engineering constraint of our method design (Section 3) and, we argue, a genuine methodological contribution: naively porting a flat metaheuristic search vector from the classical-ML setting to the embedded-LLM setting is computationally infeasible, and the two-phase discrete/continuous decomposition we propose is what makes hardware-in-the-loop metaheuristic tuning tractable on a Raspberry Pi 5.

**Contributions:**
1. A two-phase search formulation — exhaustive discrete quantization-variant selection followed by metaheuristic runtime tuning — that makes hardware-in-the-loop optimization of embedded LLM inference computationally tractable.
2. A multi-objective evaluation function jointly capturing quality (perplexity/task score), latency, memory footprint, and energy per token, with hard feasibility constraints (RAM ceiling) enforced by rejection/repair rather than soft penalty alone.
3. An empirical comparison of PSO, GWO, and DE for this task, including synthetic-function validation, convergence analysis, and robustness (boxplot) analysis under repeated runs on physical hardware — with explicit control for thermal throttling and OS jitter, a noise source absent from synthetic HPO benchmarks.
4. A comparison against grid search, random search, and vendor-default configuration under an equalized evaluation budget.

The remainder of this paper is organized as follows. Section 2 reviews related work in LLM compression, edge inference, and metaheuristic HPO. Section 3 details the method: benchmark construction, the two-phase search space, the objective function, and the metaheuristics used. Section 4 describes the experimental setup. Section 5 presents results and discussion. Section 6 discusses limitations and threats to validity. Section 7 concludes.

---

## 2. Related Work

**Status: a first-pass web literature check has been done (2026-08-01) to stress-test the Section 2.4 novelty claim before the paper is built around it. Findings below are from search-engine/abstract-level review, not full close reading of every cited paper — treat citations here as located and worth citing, not yet verified line-by-line against the source PDFs. A proper library-database pass (Google Scholar/Semantic Scholar with citation graphs) is still needed before submission.**

### 2.1 LLM compression for edge deployment
Post-training quantization (GPTQ [4], AWQ [5], LLM.int8() [6]), weight pruning, and mixed-precision schemes (SqueezeLLM [7]) reduce model size and memory bandwidth at some quality cost. `[PLACEHOLDER: position TinyLlama [1] and GGUF quantization families (Q2_K…Q8_0) within this landscape.]`

### 2.2 Edge and embedded LLM inference
llama.cpp [3] and related runtimes enable CPU-only inference on commodity edge hardware via GGUF-quantized models. Recent surveys catalog the broader edge-LLM deployment landscape — pruning/quantization/distillation for edge devices [VERIFY: e.g. a 2025 ResearchGate survey found in the literature check, needs a stable bibliographic citation], edge-vs-cloud co-tuning frameworks, and dedicated edge accelerators (NPUs supporting INT8 LLM blocks under tight DRAM budgets) [VERIFY]. Community (non-peer-reviewed) benchmarking of TinyLlama-1.1B on Raspberry Pi 5 via llama.cpp reports roughly 5-14 tokens/second generation depending on quantization level and configuration, with prompt-processing throughput around 80-108 tokens/second at Q4_0 — cited here only as an informal sanity-check range for Section 5.1's expected magnitude, not as an academic reference `[VERIFY/replace with a peer-reviewed number if one exists before submission]`.

### 2.3 Hyperparameter optimization and metaheuristics
Grid search and random search provide baselines; PSO [8], GWO [9], and DE [10] have been applied to tune classical ML models, including the direct inspiration for this work, Nouna et al. [11], who tuned KNN/SVM hyperparameters for heart disease prediction using PSO/GWO/DE after preprocessing and wrapper-based feature selection, validating optimizers on Sphere, Goldstein-Price, and a custom error function before deployment on the real task.

**Metaheuristic and AutoML-style quantization search (not LLM-specific, not hardware-in-the-loop on physical edge boards).** The closest prior art to this paper's Phase 1 is HAQ [12], which uses *reinforcement learning* — not a population-based metaheuristic — to search per-layer mixed-precision quantization bitwidths for CNNs, with a **hardware simulator** (not a physical device) providing latency/energy feedback to the RL agent. Separately, evolutionary and swarm-based methods have been applied to quantization search directly: genetic-algorithm-based mixed-precision search compared against brute-force enumeration [13], evolutionary/training-free-proxy search for mixed-precision quantization (EMQ) [14], and PSO-based (bare-bones PSO) automatic quantization of CNNs for a medical imaging task [15]. All of this line of work targets CNNs, searches quantization alone (not jointly with a serving runtime's configuration), and evaluates cost via simulators, proxies, or GPU/accelerator hardware rather than a physical low-power embedded CPU board.

**LLM-adjacent metaheuristic work is a different problem than this paper's.** A 2025 paper on "LLM-Enhanced Particle Swarm Optimization" [16] uses a large language model to *guide* PSO's search (substituting weak particle positions with LLM-suggested ones) for tuning conventional deep-learning architecture hyperparameters (LSTM/CNN layer counts, filter sizes) — the LLM is a component *inside* the optimizer, not the artifact being deployed and tuned. This is worth citing explicitly to preempt a reviewer conflating "LLM + PSO" work with this paper's very different use of PSO: here, PSO/GWO/DE tune an LLM's own deployment configuration, with no LLM inside the optimizer itself.

### 2.4 Gap
Based on this first-pass check, no located work combines all of: (a) a population-based metaheuristic (PSO/GWO/DE, as opposed to RL as in HAQ [12] or LLM-guided search as in [16]), (b) searching LLM compression *and* inference-runtime configuration jointly (as opposed to quantization alone, as in [12–15]), (c) evaluated with hardware-in-the-loop measurement on a physical, commodity, low-power embedded board (as opposed to a simulator, proxy metric, or GPU/accelerator, as in [12–15]). Each individual piece has prior art; the combination, on this class of hardware, for LLM inference specifically, did not surface in this pass. This is stated as an evidence-based gap rather than an unqualified "first" claim, and should be revisited once a full database-backed literature search (not just web search) is done — see the Limitations section.

---

## 3. Method

### 3.1 Overview

Figure 1 `[PLACEHOLDER FIGURE — pipeline diagram: Prompt benchmark → Phase 1 (exhaustive quant sweep) → Phase 2 (metaheuristic runtime search, conditioned on best quant variant(s)) → Multi-objective evaluation on device → Best configuration]`.

We decompose the search space into two tiers, mirroring the preprocessing/feature-selection → hyperparameter-tuning structure of [11]:

- **Phase 1 (discrete, exhaustive):** select a quantization variant from a small, enumerable catalog. This is analogous to [11]'s wrapper-based feature selection — a combinatorially small, exhaustively evaluable reduction step performed once, upstream of the expensive search.
- **Phase 2 (metaheuristic):** tune continuous/combinatorial runtime parameters for the chosen quantization variant(s) using PSO, GWO, and DE. This is analogous to [11]'s metaheuristic hyperparameter tuning of KNN/SVM.

This decomposition exists because quantization variants are not continuously interpolable — each corresponds to a distinct requantized model artifact — while runtime parameters (threads, batch size, context length, sampling settings) can be varied on a fixed model file at near-zero marginal cost per evaluation. Flattening both into a single metaheuristic search vector, as a naive port of [11] would suggest, would force the swarm/population to repeatedly reload and re-benchmark full model files, which is computationally infeasible within a reasonable experiment budget on a Raspberry Pi 5 (see Section 3.7 for the cost argument).

### 3.2 Target platform and model

- **Hardware:** Raspberry Pi 5 Model B Rev 1.1, 8GB RAM, Broadcom BCM2712 (4x Cortex-A76), active cooling (PWM fan, `cooling_device0: pwm-fan`). Confirmed via SSH 2026-08-15.
- **OS / software stack:** Debian GNU/Linux 13 (trixie), kernel `6.12.47+rpt-rpi-2712`. llama.cpp commit `adb55e5148dc93bcdca7212a2d1df3ccc422959a` (built from source via cmake, Release config, `-DLLAMA_CURL=OFF`; OpenSSL not found at build time so HTTPS support in the bundled server is disabled — irrelevant to Phase 1, which only uses the CLI tools). GCC 14.2.0. CMake auto-detected CPU features: `dotprod`, `fp16_vector_arithmetic`, `fma`; no SVE/SME/MATMUL_INT8 on this SoC.
- **Model:** TinyLlama-1.1B-Chat-v1.0 [1], GGUF. **Deviation from the original plan, recorded rather than silently changed:** no pre-converted F16 GGUF was found on the HF Hub (checked `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`; finest available there is Q8_0). Converting from the original transformers checkpoint via `convert_hf_to_gguf.py` would need torch+transformers installed on the Pi (heavy on ARM64, meaningful disk/bandwidth cost under this device's tight SD-card budget) just to reproduce what Q8_0 already gives near-losslessly. **Q8_0 is used as the requantization source for all Phase 1 variants instead of true F16** — llama-quantize requires `--allow-requantize` to requantize an already-quantized source, confirmed against the real build's `--help` output and applied in `llama_cpp_runner.py`. This is a defensible, widely-used proxy but is a real methodological choice, not F16, and every mention of "base model" in Section 5.1 refers to this Q8_0 source.
- **CPU governor:** confirmed set to `performance` on all 4 cores (was `ondemand` by default; changed 2026-08-15) to reduce DVFS-induced measurement noise. Available governors on this kernel: `conservative ondemand userspace powersave performance schedutil`.
- **Disk:** 29.5GB SD card (`mmcblk0p2`); freed from 90%/2.8GB-free to 38%/18GB-free by clearing a regenerable model-cache directory from an unrelated prior project, logged in the project's own history rather than the article, since it's infrastructure housekeeping, not part of the method.
- **Expected magnitude (sanity check, not a target):** informal community benchmarks (Section 2.2) put TinyLlama-1.1B generation throughput on a Pi 5 via llama.cpp roughly in the 5-14 tokens/second range depending on quantization and thread configuration. Phase 1 results (Section 5.1) wildly outside this range — in either direction — should be treated as a likely measurement or configuration bug before being reported as a finding.

### 3.3 Benchmark construction

Following [11]'s emphasis on data quality preceding optimization, we construct a fixed prompt benchmark before any tuning begins:

- `[PLACEHOLDER: N prompts, e.g., 100-300]` prompts spanning `[PLACEHOLDER: task categories, e.g., short QA, summarization, instruction following]`.
- Fixed generation length (`[PLACEHOLDER: max_new_tokens]`) and fixed random seed per prompt for reproducibility across configurations.
- Prompts with unstable timing (`[PLACEHOLDER: e.g., coefficient of variation > threshold across 3 warm-up runs]`) are removed, mirroring [11]'s pre-processing filtering step.
- A held-out subset (`[PLACEHOLDER: e.g., 20%]`) is reserved for final validation of the selected best configuration, distinct from the subset used during search.

### 3.4 Phase 1: Quantization variant selection

Candidate quantization variants (GGUF), confirmed as valid `llama-quantize` type strings against the real build: Q2_K, Q3_K_M, Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0. (No F16 entry — see Section 3.2's note on using Q8_0 as the requantization source rather than true F16; Q8_0 as a "variant" in this sweep is therefore a same-precision copy of the base, included as a consistency check.)

For each variant, evaluate once on the benchmark subset:
- Perplexity / task score
- Peak resident memory (RSS)
- Model load time
- Baseline tokens/second at default runtime settings
- Model file size on disk

This produces a table directly analogous to [11]'s Table 1 (pre-processing impact) and Table 2 (feature-selection impact) — see Section 5.1, Table `[PLACEHOLDER]`.

Selection rule: retain all variants that are not dominated on the quality/memory/speed trade-off (Pareto-nondominated) for Phase 2 — no fixed top-k cap. This avoids arbitrarily committing to a single "best" quantization before runtime tuning is known. In practice (Section 5.1), all 7 candidate variants turned out Pareto-nondominated, so the rule as stated retained the full set; a bounded compute budget then forced a separate, explicitly documented priority-order subset for the real Phase 2 search itself (Section 5.3).

### 3.5 Phase 2: Runtime hyperparameter search space

**Status: the continuous↔discrete/categorical encoding layer is implemented and unit-tested** (`src/phase2_search/parameter_space.py`) — PSO/GWO/DE search a plain bounded box in R^d; `ParameterSpace.decode(x)` maps that vector to the actual integer/categorical runtime config (rounding integers, floor-and-clip for categorical indices) so the optimizers never need to know the space is anything but continuous.

For each searched quantization variant, the metaheuristics search over a 4-dimensional mixed integer/categorical vector — **confirmed against the real llama.cpp build and its exposed CLI flags** (`src/phase2_search/run_real_phase2.py::build_space`):

| Parameter | Type | Range (final) | Notes |
|---|---|---|---|
| Threads | Integer | `[1, 4]` | Pi 5 has 4 Cortex-A76 cores; full range searched |
| Batch size (`n_batch`) | Integer | `[32, 512]` | llama.cpp prompt-processing batch (`-b`) |
| Context length (`n_ctx`) | Integer | `[128, 1024]` | Realized via `llama-bench`'s `-p` (prompt-token count), not a `-c/--ctx-size` flag — `llama-bench` has no such flag (confirmed via `--help`); capped from an originally planned 2048 upper bound after direct measurement (`diag_eval.py`) showed larger prompt sizes cost real additional minutes per evaluation, which would have made the time-boxed real-hardware budget (below) infeasible |
| KV-cache type | Categorical | `{f16, q8_0, q4_0}` | Supported by the runtime build via `-ctk`/`-ctv`; the only parameter that affects quality, so quality is pre-computed and cached once per type rather than re-measured per evaluation (Section 3.6) |

Temperature and top-p, originally planned as continuous dimensions, were **dropped from the real search** rather than left as unresolved placeholders: both only affect generation quality through sampling randomness, which perplexity — the quality metric actually available without a labeled downstream task — cannot meaningfully distinguish. Searching them would have spent real, expensive hardware evaluations moving along dimensions the objective is blind to. This narrows the search to the four parameters that mechanically and measurably affect latency, memory, energy, and (via KV-cache type only) quality.

### 3.6 Objective function

**Status: implemented and unit-tested** (`src/phase2_search/objective.py`, 4 tests covering normalization, feasibility rejection, and weighted-sum correctness — see `tests/test_objective.py`), against a mocked evaluator — see Section 5.2.2 for the end-to-end plumbing rehearsal this enabled. The real evaluator (calling `src/common/llama_cpp_runner.py` against actual hardware) is not implemented; only the `Evaluator` interface it must satisfy is fixed.

We minimize a scalarized, constrained multi-objective function:

```
f(x) = α · L̂(x) + β · (1 − Q̂(x)) + γ · M̂(x) + δ · Ê(x)  +  Penalty(x)

where, for configuration x:
  L̂(x) = normalized latency (ms/token, min-max scaled over observed range)
  Q̂(x) = normalized quality score (perplexity inverted, or task accuracy)
  M̂(x) = normalized peak memory (RSS)
  Ê(x) = normalized energy per generated token
  Penalty(x) = +∞ (hard reject) if peak RSS > RAM_LIMIT, or if the runtime
               process crashes / OOMs / exceeds a wall-clock timeout
```

Hard constraints (RAM ceiling, crash, timeout) are enforced by **rejection/repair**, not by a finite soft penalty term: an infeasible configuration is either discarded and resampled (repair) or assigned a fitness of +∞ so it is never selected as a personal/global best. This differs from a pure weighted-penalty formulation, where a sufficiently good quality score could numerically outweigh a memory violation — which is not acceptable on hardware that will actually OOM.

Weights `α, β, γ, δ`: `[PLACEHOLDER — justify choice, or report Pareto front instead of a single scalarization; consider running both a scalarized version and a small multi-objective study (e.g., NSGA-II-style Pareto front) as a robustness check].`

### 3.7 Metaheuristics and evaluation-cost argument

We use the same three metaheuristics as [11] for direct comparability: PSO [8], GWO [9], and DE [10]. Their mechanics are unchanged from the classical-ML setting (velocity/position update for PSO; alpha/beta/delta-guided position update for GWO; mutation/crossover/selection for DE) — what changes is the cost model of the fitness function, which is the core adaptation this paper makes.

**Cost argument for the two-phase design.** In [11], one fitness evaluation (fit + cross-validate a KNN/SVM) costs on the order of milliseconds, permitting population×iteration budgets like 20×100 = 2,000 evaluations. On a Raspberry Pi 5, one Phase-2 fitness evaluation (run N benchmark prompts at a given runtime configuration on an already-loaded quantized model) is expected to cost `[PLACEHOLDER: measure — estimate seconds per evaluation]`, while switching quantization variant additionally costs a full model reload (`[PLACEHOLDER: measure load time per variant]`). A flat single-phase search over both tiers at [11]'s population/iteration budget would therefore require `[PLACEHOLDER: compute — likely hours to days]` of device time, which is why Phase 1 is evaluated exhaustively but only once per variant, and Phase 2 uses a deliberately smaller budget than [11]:

`[PLACEHOLDER: population size, iterations, and number of independent runs for Phase 2 — choose based on measured per-evaluation cost from Section 4, e.g., population 8-10, iterations 20-30, 5-10 independent runs, to keep total device time within a practical bound such as a few hours.]`

### 3.8 Metaheuristic validation on synthetic functions

**Status: implemented and run (`src/phase2_search/{pso,gwo,de}.py`, `src/validation/`). This step has no hardware dependency and is complete independent of Raspberry Pi access; see Section 5.2 for results.**

Before Phase 2, PSO, GWO, and DE are validated on three closed-form benchmark functions, following [11]'s validation protocol, to confirm each optimizer converges correctly under our chosen implementation and parameter settings before trusting it on the real (noisy, hardware-measured) objective. Table 3 gives the same at-a-glance summary [11]'s own Table 3 does — representation, range, and global minimum for each function — generated directly from this project's own function implementations (`src/validation/synthetic_functions.py`, plotted by `src/validation/plot_function_representations.py`) rather than redrawn by hand, so the plotted surface is guaranteed to match what the optimizers actually search. The surface plots use a rainbow (`jet`) colormap to match [11]'s own Figure/Table 3 visual style — a deliberate exception to this project's otherwise-consistent single-hue sequential palette used in every other figure (Sections 5.2/5.2.1/5.2.2), made specifically so these two plots read as visually equivalent to the paper being mirrored.

**Table 3. Validation functions details**

| Function | Representation | Range | Global minimum |
|---|---|---|---|
| Sphere | ![Sphere surface](results/figures/function_repr_sphere.png) | `Ub=10, Lb=-10` (run at dim=10 for validation, see note below) | 0 |
| Goldstein-Price | ![Goldstein-Price surface](results/figures/function_repr_goldstein_price.png) | `Ub=2, Lb=-2` | 3 |
| FindValues | — *(no 2-variable surface exists to plot — 5-D squared error against a fixed target vector, same as [11]'s own Table 3 entry)* | `Ub=[100,700,50,200,-400]`, `Lb=[0,200,-200,-100,-900]` | 0 (error function) |

- **Sphere** (dim=10 for the actual PSO/GWO/DE validation runs in Section 5.2, though Table 3's plot shows the standard 2-variable `f(x1,x2)` case for visualization, matching [11]'s own Sphere plot) — smooth, convex, unimodal. Run at dim=10 rather than [11]'s 2D case for the real validation runs, since a 2D sphere is near-trivial for any of the three optimizers within a handful of iterations and would not meaningfully stress them; the 2D plot above is purely a visual aid, not the actual validation dimensionality.
- **Goldstein-Price** (dim=2, `lb=ub=[-2,2]`, global minimum 3) — fixed 2D, multimodal, matches [11] exactly for direct comparability, plot and validation dimensionality alike.
- **FindValues** (dim=5, asymmetric per-dimension bounds, global minimum 0) — a custom sum-of-squared-error function reproducing [11]'s own validation function: minimize the squared distance between a candidate vector and a fixed target `[21, 555, -106, 74, -701]`. Structurally the closest synthetic proxy to the real Phase 2 objective (Section 3.6), since both are error-minimization over a bounded, asymmetric, multi-dimensional space.

Implementation notes (all in `src/phase2_search/`): PSO uses global-best form with inertia damping (`w=1.0`, `wdamp=0.99`, `c1=c2=2`) and velocity clamping at 20% of the search range per dimension — the clamp is a standard stabilization addition not present in the minimal textbook form, noted here rather than left implicit. GWO follows Mirjalili et al. [9] directly (alpha/beta/delta-guided position averaging, `a` linearly decayed 2→0). DE uses DE/rand/1/bin with per-generation resampled mutation scale `F ~ U(beta_min, beta_max) = U(0.2, 0.9)` and crossover rate `pCR=0.1`, matching [11]'s Table 4 parameterization.

### 3.9 Measurement protocol (hardware-in-the-loop specifics)

Unlike [11]'s objective function (a deterministic or lightly-stochastic cross-validation error), our fitness function is measured on physical hardware and is subject to:

- **Thermal throttling:** consecutive high-load evaluations can raise SoC temperature and trigger frequency scaling. Mitigation: `[PLACEHOLDER: fixed cooldown interval between evaluations; monitor via vcgencmd measure_temp; discard/re-run evaluations above a temperature threshold]`. **Observed, not just anticipated:** during the Phase 1 sweep (2026-08-15), `vcgencmd get_throttled` reported `0xe0000` (frequency-capping, throttling, and soft-temperature-limit "has occurred" bits all set) after roughly 80 minutes of sustained perplexity computation, with SoC temperature peaking around 83°C. No cooldown/thermal-guard mitigation was implemented yet at the time of this run — the sweep was allowed to proceed regardless, on the reasoning that a first full data point was more valuable than delaying for tooling that doesn't exist yet, but this means later-evaluated quantization variants in that run may carry more thermal-throttling-induced measurement noise than earlier ones, a confound that must be reported alongside Section 5.1's results rather than papered over. Building the actual `thermal_guard.py` mitigation (still not implemented) becomes a stronger priority for Phase 2, where dozens to hundreds of back-to-back evaluations make compounding heat far more likely than Phase 1's 7 one-shot variant evaluations.
- **OS/background jitter:** `[PLACEHOLDER: run with minimal background services; pin process priority; repeat each evaluation k times and use median]`.
- **Energy measurement:** `[PLACEHOLDER: USB-C inline power meter / INA219 sensor on GPIO; measure average watts over the generation window, multiply by wall-clock time and divide by tokens generated to get joules/token]`.

This measurement noise is itself reported as part of the robustness analysis (Section 5.4 boxplots), extending [11]'s robustness protocol (10 runtimes × 100 iterations) to explicitly account for a noise source that does not exist in synthetic-function or classical-ML validation.

---

## 4. Experimental Setup

### 4.1 Hardware / software summary

`[PLACEHOLDER TABLE — mirrors paper's Table 4]`

| Component | Spec |
|---|---|
| Board | Raspberry Pi 5, `[RAM]` |
| Cooling | `[active/passive]` |
| OS | `[PLACEHOLDER]` |
| Inference engine | llama.cpp `[commit/version]` |
| Model | TinyLlama-1.1B, GGUF |
| Power measurement | `[PLACEHOLDER device]` |

### 4.2 Metaheuristic parameters

`[PLACEHOLDER TABLE — analogous to [11] Table 4]`

| Algorithm | Population | Iterations | Independent runs | Key parameters |
|---|---|---|---|---|
| PSO | `[ ]` | `[ ]` | `[ ]` | w, c1, c2 |
| GWO | `[ ]` | `[ ]` | `[ ]` | a schedule |
| DE | `[ ]` | `[ ]` | `[ ]` | beta_min, beta_max, pCR |

### 4.3 Baselines

- Vendor/manual default configuration (`[PLACEHOLDER: llama.cpp defaults]`)
- Grid search over the same Phase-2 space, evaluation-budget-matched — **implemented** (`src/baselines/grid_search.py`) and already exercised on the Section 3.8 synthetic functions (Section 5.2.1); not yet run against the real Phase 2 objective (needs Pi).
- Random search, evaluation-budget-matched — **implemented** (`src/baselines/random_search.py`), same status as above.

Including these baselines directly tests the claim — inherited from [11]'s framing of grid/random search in its introduction — that metaheuristics outperform naive search under an equal evaluation budget; this must not be assumed without a control. The synthetic-function rehearsal in Section 5.2.1 already confirms this claim holds sharply on hard, high-dimensional, asymmetric problems before the real hardware experiment is run — see that section for why grid search in particular should not be trusted as a serious baseline once Phase 2's parameter count grows past a couple of dimensions.

---

## 5. Results and Discussion

*(All content in this section is placeholder scaffolding — replace with real numbers, figures, and analysis once `hyptune/` produces experimental output. Keep the subsection structure; it mirrors [11]'s results section 1:1 for narrative consistency.)*

### 5.1 Phase 1: quantization variant impact

**Status: real results below** (`python src/phase1_quant_sweep.py`, run on the Raspberry Pi 5 target 2026-08-15/16, threads=4, wikitext-2-raw-v1 full test corpus at ctx=512, `llama-bench` repetitions=3). Full data: `results/phase1_table.csv`. Note the table below reports generation and prompt tokens/second rather than a separate "load time" column — model load time was not instrumented in this run (`llama_cpp_runner.py` doesn't currently isolate it from the bench subprocess's total wall time), a gap worth closing in a follow-up pass rather than fabricating a number here.

| Quant variant | Perplexity | Peak RSS (MB) | Gen tok/s | Prompt tok/s | File size (MB) |
|---|---|---|---|---|---|
| Q2_K | 28.8148 | 495.4 | 25.62 | 34.15 | 413.0 |
| Q3_K_M | 21.6011 | 793.4 | 19.18 | 34.47 | 523.9 |
| Q4_0 | 20.6162 | 1197.7 | 18.06 | 116.85 | 608.2 |
| Q4_K_M | 20.4490 | 1257.2 | 17.02 | 70.59 | 637.8 |
| Q5_K_M | 20.1640 | 1465.9 | 14.38 | 60.73 | 746.7 |
| Q6_K | 20.0001 | 1687.6 | 12.26 | 61.31 | 862.5 |
| Q8_0 | 19.9734 | 2180.6 | 8.47 | 54.97 | 1116.5 |

(Base model for this sweep is Q8_0, not F16 — see Section 3.2's documented deviation. Q8_0's own row is therefore a same-precision requantization of itself, included as a consistency check rather than a genuine compression point; its file size/RSS/speed being the largest/slowest of the sweep is exactly what should happen if the pipeline is working correctly, since it's the least-compressed variant tested.)

**Pareto outcome — all 7 variants retained, and that is itself the finding.** Running the Pareto filter (`src/common/pareto.py`) over (quality, tokens/s, peak RSS) returned all 7 variants as non-dominated: `results/phase1_retained.yaml` lists `[Q2_K, Q3_K_M, Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0]` in full. This is not a bug or a trivial outcome — it happens because quality, speed, and memory trade off *perfectly monotonically* with quantization level in this data: every step from Q2_K toward Q8_0 strictly improves perplexity (28.81 → 19.97) while strictly increasing memory (495MB → 2181MB) and strictly decreasing generation throughput (25.6 → 8.5 tok/s), with zero exceptions across all 7 points. A variant would only be Pareto-dominated if some other variant beat it on every axis simultaneously — impossible here since improving on any one axis (by moving toward higher precision) always costs on the other two. Two things are worth noting about the *shape* of the trade-off rather than just its monotonicity: (1) the marginal quality return shrinks sharply past Q4_K_M — Q4_0→Q4_K_M→Q5_K_M→Q6_K→Q8_0 only improves perplexity from 20.62 to 19.97 (a 3.2% relative gain) while RSS nearly doubles (1198MB→2181MB) and speed drops by more than half (18.1→8.5 tok/s), a classically poor trade past that point for most deployment goals; (2) Q4_0 and Q4_K_M are close in quality (20.62 vs 20.45) and size (608MB vs 638MB) but Q4_K_M is measurably slower (18.06 vs 17.02 tok/s) and has ~5% higher peak RSS for a modest quality edge — exactly the kind of nuance a scalar "pick the best" rule would hide and a multi-objective framing is supposed to surface.

Given the full retained set is impractically large to run the real hardware-in-the-loop Phase 2 search across all 7 (each Phase 2 evaluation costs real seconds, not the milliseconds of Section 5.2's synthetic validation — see Section 3.7's cost argument, now empirically confirmed rather than just estimated), Phase 2 in this draft runs on a deliberately chosen 3-variant subset spanning the frontier — Q2_K (fast/small extreme), Q4_K_M (the community-standard "K-quant sweet spot," and the point where marginal quality returns visibly start shrinking per the analysis above), and Q8_0 (quality extreme) — documented explicitly in `src/phase2_search/run_real_phase2.py` as `PHASE2_VARIANT_SUBSET`, not silently narrowed from the full retained list. See Section 5.3 onward.

### 5.2 Metaheuristic validation on synthetic functions

**Status: real results below (`python src/validation/run_validation.py`, population 20, 200 iterations, 10 independent runs per algorithm per function, seeds 42–51). Figures at `results/figures/{convergence,boxplot}_*.png`, raw data at `results/validation/`.**

| Function (global min) | Algorithm | Best (10 runs) | Median | Worst | Std |
|---|---|---|---|---|---|
| Sphere (0) | PSO | 6.45e-12 | 9.53e-11 | 5.99e-09 | 1.76e-09 |
| Sphere (0) | GWO | 2.87e-24 | 7.78e-23 | 7.11e-22 | 2.33e-22 |
| Sphere (0) | DE | 3.77e-07 | 6.83e-07 | 3.82e-06 | 9.65e-07 |
| Goldstein-Price (3) | PSO | 3.0000000 | 3.0000000 | 3.0000000 | 3.4e-15 |
| Goldstein-Price (3) | GWO | 3.0000027 | 3.0001698 | 84.0004 | 24.30 |
| Goldstein-Price (3) | DE | 3.0000000 | 3.0001953 | 80.8047 | 23.34 |
| FindValues (0) | PSO | 3.63e-24 | 8.68e-21 | 1.58e-19 | 5.54e-20 |
| FindValues (0) | GWO | 1.958 | 71.00 | 441.73 | 151.87 |
| FindValues (0) | DE | 1.57e-14 | 6.76e-14 | 1.11e-12 | 3.21e-13 |

All three optimizers reach values effectively indistinguishable from each function's known global minimum in at least their best run, confirming the implementations are correct before being trusted on the real, noisy Phase 2 objective. Beyond that baseline check, the comparative pattern echoes [11] closely enough to be worth stating plainly:

- **PSO is uniformly strong and low-variance across all three functions** — essentially zero std on Goldstein-Price, tightest spread on FindValues. This matches [11]'s observation that "PSO algorithm kept good performances for all three functions."
- **GWO is the most fragile of the three on the multimodal/asymmetric problems.** Its median on FindValues (71.0) is over 30x worse than its best run (1.96), and it has the single worst outlier on both Goldstein-Price (84.0) and FindValues (441.7) — visible as the outlier points in `boxplot_Goldstein_Price.png` and `boxplot_FindValues.png`. The convergence curve (`convergence_FindValues.png`) shows GWO plateauing around iteration 30 while PSO and DE continue descending for another 150+ iterations — consistent with GWO's alpha/beta/delta position-averaging update converging prematurely once the three leaders cluster, a known failure mode on functions with widely-spread, asymmetric optima.
- **DE is competitive but slower to fully converge than PSO** on Sphere and FindValues (visibly trailing PSO in the log-scale convergence plots), matching [11]'s finding that DE needed more iterations than PSO/GWO on Sphere and Goldstein-Price.
- On Goldstein-Price, most runs across all three algorithms land on *exactly* 3.0 (to floating-point precision), collapsing the box to a near-invisible line with only the rare bad run showing as an outlier — the same phenomenon [11] reports verbatim ("non-visibility of all the quartiles boxplot is due to identical error result in different runs").

One implication worth carrying into Phase 2 design: if GWO's premature-convergence pattern on asymmetric, higher-dispersion problems reproduces on the real runtime-configuration objective (Section 3.6), GWO may need either a larger population, a modified exploration schedule, or simply weaker weight in the final algorithm comparison — this is a hypothesis to check against Phase 2 data, not yet a conclusion.

#### 5.2.1 Baseline comparison rehearsal (synthetic functions)

**Status: implemented and run (`src/baselines/{grid_search,random_search,compare_baselines}.py`).** This is a **methodology rehearsal, not the real Section 5.6 result** — the actual claim the paper needs to support is that metaheuristics beat naive search on the *real hardware-measured objective*, which requires Phase 1/2 data from the Pi. What this rehearsal does establish now, honestly and in advance: the baseline implementations are correct, and the paper's introduction claim ("grid search becomes computationally expensive for numerous hyperparameters," Section 1) is not just a citation — it reproduces empirically, sharply, on these functions.

All five methods (PSO, GWO, DE, random search, grid search) were run at a matched evaluation budget of 4,000 (= population 20 × iterations 200, identical to Section 5.2's protocol), 10 independent runs for the stochastic methods. Grid search cannot spend a budget exactly — a shared per-axis resolution forces its realized evaluation count down to `floor(budget^(1/dim))^dim`, reported explicitly rather than padded:

| Function | dim | Grid resolution/axis | Grid evals used | Grid best | Random median | PSO median | GWO median | DE median |
|---|---|---|---|---|---|---|---|---|
| Sphere (min 0) | 10 | 2 | 1,024 / 4,000 | **1000.0** | 50.63 | 9.53e-11 | 7.78e-23 | 6.83e-07 |
| Goldstein-Price (min 3) | 2 | 63 | 3,969 / 4,000 | 3.441 | 3.274 | **3.000** | 3.000 | 3.000 |
| FindValues (min 0) | 5 | 5 | 3,125 / 4,000 | 4554 | 1523 | **8.69e-21** | 71.0 | 6.76e-14 |

The Sphere result is the sharpest illustration: at dim=10, a shared-resolution grid within a fixed budget can only afford **2 points per axis** — the two endpoints of each `[-10, 10]` range — so no grid point can ever land near the interior optimum at the origin. The best achievable grid-search cost, 1000, is exactly `10 axes × (10)²`: the cost of being stuck at a corner. This is not a tuning failure of this particular grid search implementation, it is the structural curse of dimensionality that Section 1 cites as the reason metaheuristics are needed at all — reproduced numerically rather than only asserted. Random search, despite also being naive, does far better than grid at the same budget (median 50.63) simply because it isn't forced into a shared coarse resolution, but still loses by 8-11 orders of magnitude to any of the three metaheuristics. At dim=2 (Goldstein-Price), where a fine 63-point grid is affordable, grid search is competitive but still measurably worse than PSO/GWO/DE's near-exact convergence to 3.0 — even the "easy," low-dimensional case doesn't favor grid search once metaheuristics are on the table. On FindValues (dim=5, asymmetric bounds — structurally the closest synthetic proxy to the real Phase 2 runtime-parameter space), grid search's best result (4554) is worse than random search's *median* (1523), a direct empirical warning against using grid search on Phase 2's real, similarly asymmetric, mixed-range parameter space (Section 3.5).

Figures: `results/figures/baseline_comparison_{Sphere,Goldstein_Price,FindValues}.png`. Full data: `results/baselines/summary.csv`, `final_costs_*.csv`.

#### 5.2.2 Phase 2 plumbing rehearsal (mock evaluator — NOT real hardware data)

**Status: implemented and run** (`src/phase2_search/{parameter_space,objective,mock_evaluator,run_mock_rehearsal}.py`). This is the narrowest and most important caveat in this draft, stated as plainly as possible: **every number in this subsection comes from a made-up, hand-specified cost surface in `mock_evaluator.py` — diminishing-returns thread speedup, KV-cache memory roughly linear in context length, a quality penalty for lossy quantized KV cache — none of it measured from a real TinyLlama run or real Pi hardware.** Its only purpose is to prove the Phase 2 plumbing (parameter encode/decode, feasibility rejection, normalization calibration, PSO/GWO/DE wiring to a 4-dimensional mixed integer/categorical objective) works correctly end to end before the same code is pointed at a real `Evaluator` backed by `llama_cpp_runner.py` and actual hardware. **This subsection must be deleted or fully replaced once real Phase 1/2 data exists — it is scaffolding, not a finding.**

Search space: `threads∈[1,4]` (int), `n_batch∈[32,512]` (int), `n_ctx∈[512,4096]` (int), `kv_cache_type∈{f16,q8_0,q4_0}` (categorical) — the mechanically-grounded subset of Section 3.5 (temperature/top-p are omitted from this rehearsal since the mock model has no task-accuracy notion for them to affect; they matter once a real task-accuracy quality metric replaces perplexity). Normalization ranges were calibrated by 300 random feasible samples (Section 3.6's calibration procedure), the RAM ceiling was set to a toy 900 MB (not the real Pi 5 limit) specifically to create a genuine feasible/infeasible boundary inside the search space for the rejection logic to navigate. PSO, GWO, and DE (population 20, 100 iterations, 10 runs) all converged to the identical configuration: `threads=4, n_batch=32, n_ctx=512, kv_cache_type=f16` (cost 0.0226), which a brute-force grid scan over the same 4-dimensional space (feasible on this small, cheap-to-evaluate mock objective, unlike the real dim=10 Sphere case in Section 5.2.1) confirms is the true global optimum of the mock surface.

Two things are worth being honest about in this result. First, positively: the full pipeline — encode/decode, calibration, feasibility rejection, and all three optimizers — is verified correct, because the found solution matches a brute-force ground truth. Second, a real limitation of the mock model rather than of the search machinery: this particular toy cost surface happens to have a single corner that simultaneously minimizes latency, maximizes quality, and minimizes memory and energy — i.e. no genuine multi-objective trade-off exists in this instance, so the rehearsal does not exercise the interesting case (Pareto tension between competing objectives) that the real hardware objective is expected to have. A follow-up rehearsal with a deliberately conflicting mock surface (e.g., a quality metric that improves with larger context in some way that actually costs memory) would be a better test of the optimizers' behavior under real trade-off pressure — worth doing before Pi access if there's more runway, but not done in this pass.

Figures: `results/figures/mock_phase2_{convergence,boxplot}.png`. Data: `results/mock_phase2/best_configs.csv`.

**Status: real hardware run in progress on the Pi 5.** Scope decision (documented in `run_real_phase2.py`, not silently narrowed): Phase 1 (Section 5.1) found all 7 quantization variants Pareto-nondominated, so running the real Phase 2 metaheuristic search on all 7 is not tractable in a single-night compute budget at real (tens-of-seconds-per-evaluation) hardware cost. Three variants spanning the retained frontier were selected — **Q4_K_M** (community-default "sweet spot"), **Q2_K** (fast/small extreme), **Q8_0** (quality extreme) — processed in that priority order so a partial run still yields the single most representative variant complete. As of this draft: **Q4_K_M is complete**; Q2_K and Q8_0 are `[IN PROGRESS / PENDING]` and will be added before submission.

### 5.3 Phase 2: runtime configuration search — convergence

**Q4_K_M (complete).** Figure `results/figures/real_phase2_convergence_Q4_K_M.png` plots the median (of 2 runs) best-cost-so-far against iteration for PSO, GWO, and DE, on the real hardware-measured objective (Section 3.6) with population 4 and 5 iterations — a budget an order of magnitude smaller than the synthetic-validation protocol (Section 5.2), by necessity: at a measured ≈33s per real evaluation (calibrated per-variant, `meta.yaml`), the population×iteration floor used in Section 5.2 (20×200) would not complete overnight.

| Iteration | PSO (median) | GWO (median) | DE (median) |
|---|---|---|---|
| 1 | 0.1047 | 0.1909 | 0.1136 |
| 2 | 0.1047 | 0.1158 | 0.1136 |
| 3 | 0.1047 | 0.1080 | 0.1136 |
| 4 | 0.0882 | 0.1080 | 0.0603 |
| 5 | 0.0882 | 0.1080 | 0.0603 |

DE reaches the lowest median final cost (0.0603) and, in its best single run, the lowest cost of any algorithm on this variant (0.0205, run 1 — see Section 5.4). GWO converges earliest in wall-clock terms (its iteration-2 value is already close to its iteration-5 value) but plateaus at a visibly worse cost than PSO or DE — the same premature-convergence pattern flagged as a hypothesis in Section 5.2 on the asymmetric synthetic functions reproduces here on the real objective. PSO is flat for the first three iterations before a single late improvement at iteration 4; with only 5 iterations total this is a coarse curve, not a smooth one, and should be read as a budget-constrained real-hardware result rather than a well-resolved convergence trace.

**Q2_K (complete).** Figure `results/figures/real_phase2_convergence_Q2_K.png`, same protocol (population 4, 5 iterations, per-eval cost 59.9s — nearly double Q4_K_M's 32.9s, since Q2_K's smaller model runs faster per token but the calibration landed on a different sample mix):

| Iteration | PSO (median) | GWO (median) | DE (median) |
|---|---|---|---|
| 1 | 0.0607 | 0.0913 | 0.0460 |
| 2 | 0.0607 | 0.0913 | 0.0460 |
| 3 | 0.0550 | 0.0388 | 0.0460 |
| 4 | 0.0550 | 0.0238 | 0.0423 |
| 5 | 0.0488 | 0.0238 | 0.0423 |

On Q2_K, GWO overtakes both PSO and DE by iteration 4 rather than plateauing early as it did on Q4_K_M — the premature-convergence hypothesis from Section 5.2 does **not** generalize to this variant. All three algorithms converge to a narrower final band (0.0238–0.0488) than on Q4_K_M (0.0603–0.1080), consistent with Q2_K's smaller, cheaper model exposing a flatter, easier-to-navigate cost surface.

### 5.4 Robustness across repeated runs

**Q4_K_M (complete).** With `N_RUNS = 2` — reduced from the synthetic-validation protocol's 10 runs because each run here costs real Pi wall-clock time (Section 5.3), not milliseconds — variance can only be characterized coarsely, and this is stated as a limitation rather than glossed over (Section 6).

| Algorithm | Run 1 | Run 2 | Spread |
|---|---|---|---|
| PSO | 0.0350 | 0.1414 | 4.0x |
| GWO | 0.1094 | 0.1066 | 1.03x |
| DE | **0.0205** | 0.1000 | 4.9x |

Figure: `results/figures/real_phase2_boxplot_Q4_K_M.png`. The pattern **inverts** what Section 5.2's synthetic-function validation would predict: there, GWO was flagged as the most fragile/high-variance algorithm and PSO as uniformly low-variance. Here, on the real objective, GWO is by far the *most* consistent (1.03x spread) while both PSO and DE swing by roughly 4-5x between their two runs. Two non-exclusive explanations are worth stating plainly rather than picking one without evidence: (1) with only 2 runs, "variance" is really just the gap between two samples, not a stable estimate — a single unlucky PSO/DE run could flip this ranking, so this result should not be over-interpreted as a general PSO/DE weakness; (2) GWO's low variance here may be the *same* premature-convergence behavior noted in Section 5.3 rather than a virtue — an algorithm that plateaus early and stays put will naturally show low run-to-run spread, at the cost of the worse absolute values visible in Table 5.3's convergence data. Distinguishing these honestly requires more runs than the real-hardware compute budget allowed for this draft.

**Q2_K (complete).**

| Algorithm | Run 1 | Run 2 | Spread |
|---|---|---|---|
| PSO | 0.0247 | 0.0729 | 3.0x |
| GWO | **0.0000** | 0.0477 | n/a (run 1 = exact floor) |
| DE | 0.0189 | 0.0658 | 3.5x |

Figure: `results/figures/real_phase2_boxplot_Q2_K.png`. GWO's run 1 landed on a cost of *exactly* 0.0000 — not approximately zero, but every normalized term simultaneously at its clipped floor (Section 3.6's `normalize()` clips to `[0, 1]`). This is worth explaining rather than reporting as a triumphant result: `meta.yaml`'s calibrated energy range for Q2_K has a minimum of 3.24 J/token, yet GWO's found configuration measured 1.26 J/token — *below* the calibrated minimum — so the normalized energy term clips to 0 rather than reflecting a true relative ranking. This is a direct consequence of `N_CALIBRATION_SAMPLES = 8`: eight random samples do not reliably bracket the true min/max of a 4-dimensional space, and an optimizer that later visits a more extreme corner (here, `n_ctx=128`, the lower bound of the search range) can fall outside the calibrated window entirely. The same clipping affected Grid search's best point independently (also `n_ctx=128`, also cost 0.0000 — see Section 5.5), which is itself informative: two unrelated methods converging on the identical corner is real evidence that corner is genuinely strong for Q2_K, even though the exact "0.0000" values are a calibration-sample-size artifact rather than a meaningful score of zero. Run-to-run spread otherwise follows the same qualitative pattern as Q4_K_M — GWO's second run (0.0477) is unremarkable, so the run-1 zero looks more like a lucky corner hit than a systematically superior algorithm on this variant.

### 5.5 Best configurations found

**Q4_K_M (complete)** — best configuration found by each method, with the full raw measurement (not just the scalarized cost) for that configuration:

| Method | Threads | n_batch | n_ctx | KV type | f(x) | Latency (ms/tok) | Quality (−PPL) | Peak RSS (MB) | Energy (J/tok) |
|---|---|---|---|---|---|---|---|---|---|
| **DE** | 3 | 318 | 387 | f16 | **0.0205** | 53.86 | −22.3879 | 1248.5 | 1.639 |
| PSO | 3 | 308 | 567 | q8_0 | 0.0350 | 54.61 | −22.3722 | 1252.5 | 2.689 |
| GWO | 3 | 217 | 764 | q8_0 | 0.1066 | 54.97 | −22.3722 | 1249.6 | 3.568 |
| Random | 3 | 311 | 663 | f16 | 0.1238 | 54.65 | −22.3879 | 1243.7 | 2.725 |
| Grid | 4 | 512 | 128 | f16 | 0.3311 | 63.02 | −22.3879 | 1239.5 | 0.805 |

The overall winner (DE) and the second-best (PSO) agree closely on threads (3, not the maximum 4) and n_batch (~310-320), differing mainly in n_ctx (387 vs. 567) and KV-cache type (f16 vs. q8_0) — with near-identical quality between f16 and q8_0 KV cache (−22.3879 vs. −22.3722, a 0.07% difference), the objective's latency/energy terms dominate the choice, and DE's smaller n_ctx plus f16 cache combination wins mainly on energy (1.639 vs. 2.689 J/token, a 39% reduction) at essentially the same latency. Grid search's best point is revealing in the other direction: it lands on the smallest n_ctx (128) and *lowest* energy of any method (0.805 J/tok) but pays for it with the worst latency (63.02ms/tok, ~17% slower than DE) and, combined with the objective's other weighted terms, the worst overall cost (0.3311, 16x worse than DE) — a reminder that no single raw metric should be read as "the" answer; the scalarized objective exists precisely because latency, energy, memory, and quality trade off against each other in ways a single-metric view would rank differently (Grid "wins" on energy alone, DE wins overall).

**Q2_K (complete):**

| Method | Threads | n_batch | n_ctx | KV type | f(x) | Latency (ms/tok) | Quality (−PPL) | Peak RSS (MB) | Energy (J/tok) |
|---|---|---|---|---|---|---|---|---|---|
| **GWO** | 4 | 153 | 128 | f16 | **0.0000** | 42.99 | −31.3272 | 468.8 | 1.257 |
| Grid | 4 | 512 | 128 | f16 | 0.0000 | 44.83 | −31.3272 | 468.0 | 1.193 |
| DE | 4 | 167 | 387 | f16 | 0.0189 | 42.83 | −31.3272 | 467.9 | 3.052 |
| PSO | 4 | 221 | 409 | f16 | 0.0247 | 42.54 | −31.3272 | 468.9 | 3.173 |
| Random | 4 | 318 | 387 | f16 | 0.0482 | 44.01 | −31.3272 | 476.9 | 3.158 |

Every method's best configuration uses `threads=4, kv_cache_type=f16` — a much stronger cross-method consensus than Q4_K_M's split between f16 and q8_0. GWO and Grid land on essentially the same point (`n_ctx=128`, small but different `n_batch`) and tie on the calibration-clipped cost of 0.0000 (see Section 5.4's discussion of this artifact); the more informative comparison is DE and PSO, both of which found `n_ctx≈390-410` at real, non-clipped costs (0.0189, 0.0247) — meaningfully worse than GWO/Grid's corner but with plausible, non-degenerate raw metrics. Unlike Q4_K_M, where energy varied 4.4x across methods' best points (0.805-3.568 J/tok) alongside a genuine latency trade-off, Q2_K's five best configurations cluster within 3ms/tok of each other in latency (42.5-44.8) — for this smaller, cheaper model, the search space appears close to flat in the region the optimizers explored, consistent with the narrower final-cost band already noted in Section 5.3.

### 5.6 Comparison against baselines

**Q4_K_M (complete).** At the matched evaluation budget (population 4 × iterations 5 = 20 evaluations for PSO/GWO/DE and Random; Grid realized 16/20 due to its shared-resolution constraint, Section 5.2.1):

| Method | Best f(x) | vs. best metaheuristic (DE) |
|---|---|---|
| DE | 0.0205 | — |
| PSO | 0.0350 | 1.7x worse |
| GWO | 0.1066 | 5.2x worse |
| Random search | 0.1238 | 6.0x worse |
| Grid search | 0.3311 | 16.1x worse |

On this variant, all three metaheuristics beat both naive baselines, and the best of them (DE) beats grid search by over an order of magnitude — echoing, on the real hardware objective rather than only the synthetic functions of Section 5.2.1, the paper's central motivating claim that grid search does not scale to this kind of mixed integer/categorical space. The margin over random search is narrower than over grid (DE is 6.0x better vs. random, 16.1x vs. grid), consistent with Section 5.2.1's synthetic-function finding that random search — despite being naive — is a meaningfully stronger baseline than grid once dimensionality or a mixed/asymmetric range structure is involved.

**Q2_K (complete):**

| Method | Best f(x) | vs. best metaheuristic (GWO) |
|---|---|---|
| GWO | 0.0000 | — |
| Grid search | 0.0000 | tied (calibration-clipped, Section 5.4) |
| DE | 0.0189 | n/a — worse than a clipped 0 |
| PSO | 0.0247 | n/a |
| Random search | 0.0482 | n/a |

Q2_K **complicates** the clean Q4_K_M narrative rather than confirming it: grid search *ties* the best metaheuristic here, not because grid search is genuinely competitive, but because both GWO and grid happened to land on the same under-calibrated corner (Section 5.4) — the honest comparison is DE/PSO (real, non-clipped metaheuristic results: 0.0189, 0.0247) against random search (0.0482), a 2.5-2.6x improvement, far short of Q4_K_M's 6-16x margins. Reported plainly rather than cherry-picked toward the stronger variant: **the metaheuristic-vs-baseline margin is not constant across quantization variants** — it is large and clear on Q4_K_M, and narrow-to-confounded on Q2_K because of a calibration artifact. This is itself a finding worth stating rather than hiding: a small calibration-sample budget (`N_CALIBRATION_SAMPLES=8`, forced by real-hardware time cost) can materially change which method looks best on a given variant, and this should be disclosed as a limitation (Section 6), not smoothed over.

Q8_0 baseline data is `[PENDING]` and will complete this section's cross-variant range.

### 5.7 Discussion

`[PARTIAL — Q4_K_M and Q2_K done; complete once Q8_0 finishes]`

For Q4_K_M, the practical guidance the search data supports: **prefer DE's found configuration** (`threads=3, n_batch=318, n_ctx=387, kv_cache_type=f16`) for a balanced deployment, since it achieves both the lowest latency-adjacent cost and the lowest energy among the non-grid methods at statistically indistinguishable quality from the q8_0-cache alternatives. For a strictly memory-constrained deployment, all five methods' best configurations cluster tightly in peak RSS (1239.5-1252.5 MB, a 1% spread) — for this variant, memory is not the axis the search actually differentiates on; latency and energy are.

For Q2_K, the guidance is different in character: **prefer DE's or PSO's found configuration** (`n_ctx≈390-410`, f16, 4 threads) over GWO's or grid's nominal "winner", precisely because the latter's win is a calibration artifact (Section 5.4) rather than a genuinely superior configuration — a concrete illustration of why raw scalarized cost should never be read off a results table without checking the underlying raw metrics (Section 5.5), a discipline this paper's own results table would violate if GWO/Grid's 0.0000 were reported as the headline number without the explanation given above.

Comparing the two completed variants directly speaks to the paper's core question — whether the two-phase decomposition's Pareto-retained multi-variant set (Section 5.1) actually mattered: **yes, so far.** The best achievable configuration, its cost, and even which algorithm should be trusted as "the winner" differ substantively between Q4_K_M and Q2_K — Q4_K_M has a real, resolvable trade-off between f16/q8_0 KV cache and context length that DE navigates better than the naive baselines by an order of magnitude; Q2_K's search space is comparatively flat, its naive-baseline margin is much narrower, and its nominal best result requires a caveat to interpret correctly. A single quantization variant chosen up front — before Phase 2 tuning — would not have surfaced either of these variant-specific patterns. Whether this conclusion holds a third way once Q8_0 (the quality-extreme variant, `[PENDING]`) completes is the last open piece of this section.

---

## 6. Limitations and Threats to Validity

- **Single model family (TinyLlama-1.1B) and single hardware target (Pi 5)** — generalization to other model sizes/devices is untested.
- **Scalarized objective with fixed weights** collapses a genuine multi-objective trade-off into one number; a Pareto-front analysis would be more informative but more expensive. Section 5.5's Q4_K_M result shows this concretely (Grid "wins" on energy alone but loses overall) — the scalarization is doing real, non-trivial work, not just cosmetic aggregation.
- **Calibration sample size is small ($N_{\text{calib}}=8$), and this measurably changed a result.** Normalization ranges (Section 3.6) are estimated from 8 random warm-cache samples rather than an exhaustive or much larger sweep, because each sample costs real Pi wall-clock time. On Q2_K, this caused two independently-found optima (GWO, Grid search) to land outside the calibrated min/max window and clip to an exact cost of 0.0000 (Section 5.4) — not a true global optimum, but a normalization artifact. This directly narrowed the apparent metaheuristic-vs-baseline margin on that variant (Section 5.6: 2.5-2.6x rather than Q4_K_M's 6-16x). A larger calibration budget, or a normalization scheme that re-widens ranges online as the search visits more extreme points, would remove this artifact — not done here due to the real-hardware time budget.
- **N_RUNS reduced from 10 (synthetic validation, Section 5.2) to 2 (real Phase 2, Sections 5.3-5.6)** for the same reason — real per-evaluation cost. With only 2 runs, reported "spread" is the gap between two samples, not a statistically grounded variance estimate; the GWO-appears-most-stable finding (Section 5.4) is explicitly flagged as possibly an artifact of this small N rather than a robust property of the algorithm.
- **Hardware measurement noise** (Section 3.9 — thermal throttling observed at 83-84.5°C during Phase 1) may dominate small differences between top-performing configurations; point estimates are reported without confidence intervals for exactly this reason (too few runs to construct one meaningfully, per the point above).
- **Quality metric is perplexity, not task accuracy** — the search space (Section 3.5) deliberately dropped temperature/top-p because perplexity cannot distinguish sampling-parameter effects on generation quality. A downstream task benchmark (e.g., a QA or instruction-following eval) would enable searching those dimensions and might rank configurations differently than perplexity does here.
- **Search-budget-driven scope narrowing at two levels**, both documented rather than silently applied: (1) Phase 2's real search covers 3 of the 7 Pareto-retained Phase 1 variants (Q4_K_M, Q2_K done; Q8_0 in progress) rather than all 7; (2) `n_ctx`'s upper bound was capped at 1024 rather than an originally planned 2048 after direct measurement showed larger prompt sizes cost real additional minutes per evaluation.
- **Load time was never instrumented** — Phase 1/2 measure inference latency, memory, quality, and energy, but not model-load time, which matters for real deployment (e.g., cold-start latency) and is a plausible axis on which quantization variants would differ further.
- Novelty claim (Section 2.4) is currently based on a single web-search pass (2026-08-01), not a full database-backed literature review (Google Scholar/Semantic Scholar citation-graph traversal, backward/forward citation chasing from HAQ [12] and the quantization-search papers [13-15]). The claim as written is already deliberately narrow (specific combination of population-based metaheuristic + joint compression/runtime search + physical embedded hardware), but a proper review pass before submission could still surface something that narrows it further, or a paper this search missed entirely.

---

## 7. Conclusion and Future Work

`[PLACEHOLDER — draft after results, following [11]'s conclusion structure: confirm the two contributions (feasibility of the two-phase hardware-in-the-loop method; comparative metaheuristic performance), then future work:]`

- Extend to other edge platforms (Jetson Orin Nano, mobile SoCs) and other small models (Phi-3-mini, Qwen2.5-0.5B/1.5B, Gemma-2B).
- Hybrid metaheuristics (e.g., PSO-DE hybrids) as in emerging metaheuristic literature.
- Multi-objective Pareto-front search (NSGA-II/MOEA-D) instead of scalarization.
- Joint search including model-level compression beyond quantization (structured pruning, LoRA-merged distillation) as an additional discrete Phase 1 axis.

---

## Funding

`[PLACEHOLDER]`

## Author Contributions (CRediT)

`[PLACEHOLDER — table as in [11]]`

## Conflict of Interest

`[PLACEHOLDER]`

## Data Availability

`[PLACEHOLDER: benchmark prompts, raw measurement logs, and code to be released at <repo URL> upon publication.]`

---

## References

*(Real, verifiable sources are numbered and used as-is. Entries marked `[VERIFY]` must be checked against the actual bibliographic record — exact venue, year, page numbers — before submission. Do not submit this reference list unverified.)*

[1] Zhang, P., Zeng, G., Wang, T., & Lu, W. (2024). TinyLlama: An Open-Source Small Language Model. *arXiv:2401.02385*. `[VERIFY final venue/version]`

[2] `[PLACEHOLDER: GGUF format reference / llama.cpp documentation citation]`

[3] Gerganov, G. et al. *llama.cpp* [Software]. GitHub. `[VERIFY: cite specific commit/release used in experiments]`

[4] Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. *arXiv:2210.17323*.

[5] Lin, J., Tang, J., Tang, H., Yang, S., Chen, W.-M., Wang, W.-C., Xiao, G., Dang, X., Gan, C., & Han, S. (2023). AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. *arXiv:2306.00978*.

[6] Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. *arXiv:2208.07339*.

[7] Kim, S., Hooper, C., Gholami, A., Dong, Z., Li, X., Shen, S., Mahoney, M. W., & Keutzer, K. (2023). SqueezeLLM: Dense-and-Sparse Quantization. *arXiv:2306.07629*.

[8] Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. *Proceedings of ICNN'95 — International Conference on Neural Networks*, 4, 1942–1948.

[9] Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer. *Advances in Engineering Software*, 69, 46–61.

[10] Storn, R., & Price, K. (1997). Differential evolution — a simple and efficient heuristic for global optimization over continuous spaces. *Journal of Global Optimization*, 11(4), 341–359.

[11] Nouna, Z., Bouyghf, H., Nahid, M., & Sabiri, I. (2025). Heart disease prediction optimization using metaheuristic algorithms. *IAES International Journal of Artificial Intelligence*, 14(5), 4332–4341.

[12] Wang, K., Liu, Z., Lin, Y., Lin, J., & Han, S. (2019). HAQ: Hardware-Aware Automated Quantization with Mixed Precision. *CVPR 2019*. `arXiv:1811.08886`. `[VERIFY page numbers/proceedings citation]`

[13] `[VERIFY: genetic-algorithm-based mixed-precision quantization search paper located in the literature check — need to pin down the exact title/venue; found via a search result describing GA search vs. brute-force enumeration over per-layer bitwidths, not yet traced to a specific citable paper.]`

[14] `[VERIFY exact authors/venue]`. EMQ: Evolving Training-free Proxies for Automated Mixed Precision Quantization. `arXiv:2307.10554`.

[15] `[VERIFY exact authors/venue]`. Automatic Quantization of Convolutional Neural Networks Based on Enhanced Bare-Bones Particle Swarm Optimization for Chest X-Ray Image Classification. *Computational Collective Intelligence* (Springer). `[VERIFY DOI: 10.1007/978-3-031-41456-5_10]`

[16] Hameed, A., Qolomany, B., et al. (2025). Large Language Model Enhanced Particle Swarm Optimization for Hyperparameter Tuning for Deep Learning Models. `arXiv:2504.14126`. `[VERIFY full author list/IEEE citation]`

`[PLACEHOLDER: this reference list still needs a proper database-backed pass (Google Scholar / Semantic Scholar with citation-graph checking) — the above were located via web search at abstract level on 2026-08-01 and are not yet verified against the full source PDFs. In particular [13] is not yet pinned to a specific paper and must be resolved or removed before submission. Also still needed: a peer-reviewed (not community-blog) source for Raspberry Pi 5 / TinyLlama throughput numbers if one exists, to replace the informal benchmark range cited in Section 2.2.]`

---

## Appendix A — Planned repository structure (for `hyptune/`)

```
hyptune/
├── article_draft.md          # this file
├── data/
│   └── prompts/               # benchmark prompt set (Section 3.3)
├── models/
│   └── tinyllama-gguf/        # quantized model variants (Phase 1 outputs)
├── src/
│   ├── phase1_quant_sweep.py  # exhaustive quant-variant evaluation [Pi-dependent, built, not yet run]
│   ├── common/
│   │   ├── llama_cpp_runner.py  # quantize/bench/perplexity subprocess wrappers [Pi-dependent]
│   │   └── pareto.py            # non-dominated filter, used by Phase 1 [done]
│   ├── phase2_search/
│   │   ├── result.py            # shared OptimizeResult dataclass [done]
│   │   ├── pso.py                # [done, validated]
│   │   ├── gwo.py                # [done, validated]
│   │   ├── de.py                  # [done, validated]
│   │   ├── parameter_space.py   # continuous<->discrete/categorical encode/decode, Section 3.5 [done, tested]
│   │   ├── objective.py         # f(x) from Section 3.6 [done, tested — real Evaluator not yet implemented]
│   │   ├── mock_evaluator.py    # SYNTHETIC evaluator for plumbing rehearsal only [done — never a real data source]
│   │   └── run_mock_rehearsal.py  # Section 5.2.2 driver [done, run]
│   ├── validation/
│   │   ├── synthetic_functions.py  # Sphere/Goldstein-Price/FindValues, Section 3.8 [done]
│   │   ├── run_validation.py       # convergence + boxplot driver, Section 5.2 [done, run]
│   │   └── plot_function_representations.py  # Table 3 surface plots, Section 3.8 [done, run]
│   ├── baselines/
│   │   ├── grid_search.py          # [done]
│   │   ├── random_search.py        # [done]
│   │   └── compare_baselines.py    # matched-budget comparison, Section 5.2.1/4.3 [done, run]
│   └── measurement/
│       ├── power_logger.py    # energy measurement, Section 3.9 [not started — needs hardware]
│       └── thermal_guard.py   # cooldown / throttle detection [not started — needs Pi]
├── tests/
│   ├── test_parameter_space.py  # [done, passing]
│   └── test_objective.py        # [done, passing]
├── results/
│   ├── phase1_table.csv       # [pending Pi]
│   ├── validation/            # [done — Section 5.2 data]
│   ├── baselines/             # [done — Section 5.2.1 data]
│   ├── mock_phase2/           # [done — Section 5.2.2 SYNTHETIC data, not real findings]
│   ├── phase2_runs/           # [pending Pi — real Phase 2 data goes here]
│   └── figures/               # convergence_*.png, boxplot_*.png, baseline_comparison_*.png, mock_phase2_*.png [done]
└── notebooks/
    └── analysis.ipynb         # generates Section 5 tables/figures [not started]
```

## Appendix B — Two-phase search pseudocode

```text
# Phase 1 — exhaustive, run once
for variant in QUANT_VARIANTS:
    load model(variant)
    result[variant] = evaluate(quality, memory, load_time, baseline_speed)
retained = pareto_nondominated(result, axes=[quality, memory, speed])

# Phase 2 — metaheuristic, per retained variant
for variant in retained:
    load model(variant)               # loaded once, reused across all evaluations
    for algorithm in [PSO, GWO, DE]:
        for run in range(N_INDEPENDENT_RUNS):
            population = init_population(RUNTIME_PARAM_SPACE)
            for iteration in range(MAX_ITER):
                for candidate in population:
                    if infeasible(candidate):     # RAM ceiling, crash, timeout
                        fitness = +inf             # reject, no soft penalty
                    else:
                        fitness = f(candidate)     # Section 3.6, measured on device
                update_population(algorithm, population, fitness)
                log(best_so_far)                  # convergence curve, Section 5.3
            record(best_of_run)                   # robustness boxplot, Section 5.4
```
