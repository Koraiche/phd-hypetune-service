# hyptune

Code companion to `article_draft.md` (working draft) / `latex/article.tex`
(IEEEtran journal-format LaTeX version, kept in sync with the markdown
draft): metaheuristic co-optimization of compression and runtime
configuration for TinyLlama inference on a Raspberry Pi 5.

## Building the LaTeX version

```
cd latex
pdflatex article.tex
pdflatex article.tex   # second pass to resolve cross-references/citations
```

Requires `IEEEtran.cls`, `booktabs`, `subcaption`, `algorithm`/`algpseudocode`
(all in a standard TeX Live install). Figures are pulled from
`../results/figures/` via `\graphicspath`, so Phase 1/2 results regenerated
there show up on the next compile without editing the `.tex` file. All
incomplete parts render as colored `[PLACEHOLDER: ...]` / `[VERIFY: ...]`
text and blue `Status:` boxes — search the source for `\ph{`, `\verifynote{`,
`\status{` to find everything that still needs resolving before submission.

Implemented so far:
- **Phase 1 — exhaustive quantization-variant sweep** (article Section 3.4). Requires the Raspberry Pi 5 / llama.cpp toolchain — not runnable until hardware is available.
- **Metaheuristic implementations + synthetic-function validation** (article Section 3.8/5.2) — `src/phase2_search/{pso,gwo,de}.py` and `src/validation/`. Pure Python/numpy, no hardware dependency, already run — see "Running the metaheuristic validation" below.
- **Grid/random search baselines + matched-budget comparison** (article Section 4.3/5.2.1) — `src/baselines/`. Also pure Python/numpy, already run against the synthetic functions (not yet against the real Phase 2 objective, which needs Pi data) — see "Running the baseline comparison" below.
- **Phase 2 objective-function plumbing** (article Section 3.5/3.6/5.2.2) — `src/phase2_search/{parameter_space,objective,mock_evaluator,run_mock_rehearsal}.py` plus `tests/`. The parameter encode/decode layer and the scoring/normalization/feasibility-rejection logic are implemented, unit-tested (`pytest tests/`), and exercised end to end with PSO/GWO/DE against a **synthetic mock evaluator** — not real hardware data, see the disclaimer in "Running the mock Phase 2 rehearsal" below.

Not built yet: the *real* Evaluator (`src/common/llama_cpp_runner.py` is the hardware-facing half; nothing yet implements the `Evaluator` protocol against it) — swapping `MockLlamaCppEvaluator` for a real one in `run_mock_rehearsal.py`'s structure is the actual Phase 2 integration step once Pi + Phase 1 data exist.

## Setup (run on the Raspberry Pi 5 target — Phase 1 measures real hardware behavior)

1. **Build llama.cpp** on the Pi, if not already:
   ```
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp && cmake -B build && cmake --build build --config Release -j4
   ```
   Confirm the resulting binaries' names and flags match what
   `src/common/llama_cpp_runner.py` expects — llama.cpp's CLI has changed
   flag names across releases. Run `./build/bin/llama-quantize --help`,
   `llama-bench --help`, `llama-perplexity --help` and diff against the
   assumptions documented at the top of `llama_cpp_runner.py` before trusting
   any numbers.

2. **Install Python dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Get a base F16 GGUF for TinyLlama-1.1B-Chat-v1.0.** Two options:
   - If a repo on the HF Hub already hosts a converted `.gguf` file, set
     `base_model.hf_repo` / `hf_filename` in `config/phase1.yaml` and the
     script will download it automatically.
   - Otherwise, convert from the original transformers checkpoint yourself
     using llama.cpp's `convert_hf_to_gguf.py`, and point
     `base_model.local_path` at the resulting file. This repo does not
     script that conversion step yet — it's a one-time manual step, add it
     to `phase1_quant_sweep.py::ensure_base_model` if you need it automated.

4. **Get a real perplexity corpus.** The bundled
   `data/prompts/perplexity_smoke_sample.txt` is only for confirming the
   pipeline runs end to end — it is too short and repetitive to produce a
   meaningful perplexity number. For actual results, download
   wikitext-2-raw's `wiki.test.raw` (the standard corpus for GGUF
   perplexity comparisons) and point `perplexity.corpus_path` at it.

5. **Edit `config/phase1.yaml`** — in particular `llama_cpp.bin_dir` (path to
   your build's `bin/` directory) and the perplexity corpus path.

## Running Phase 1

```
python src/phase1_quant_sweep.py --config config/phase1.yaml
```

Outputs:
- `results/phase1_table.csv` — per-variant perplexity, tokens/s, peak RSS,
  file size. This is the data source for article_draft.md Section 5.1.
- `results/phase1_retained.yaml` — the Pareto-nondominated quant variants
  (on quality / speed / memory) to carry into Phase 2.

## Running the metaheuristic validation (no hardware needed)

```
python src/validation/run_validation.py
python src/validation/plot_function_representations.py   # Table 3-style surface plots
```

Runs PSO, GWO, and DE (`src/phase2_search/`) against Sphere, Goldstein-Price,
and a custom FindValues error function — 20 population, 200 iterations, 10
independent runs each. Outputs:
- `results/validation/summary.csv`, `convergence_*.csv`, `final_costs_*.csv`
- `results/figures/convergence_*.png`, `boxplot_*.png`

Findings are written up in `article_draft.md` Section 5.2. Headline: all
three converge correctly (sanity-checked against each function's known
global minimum); PSO is uniformly strong and low-variance; GWO is fragile on
the asymmetric/multimodal functions (large outliers, premature convergence
on FindValues); DE is competitive but slower to fully converge than PSO.

## Running the baseline comparison (no hardware needed)

```
python src/baselines/compare_baselines.py
```

Runs grid search and random search (`src/baselines/`) alongside PSO/GWO/DE
at a matched evaluation budget (4,000 = population 20 × iterations 200) on
the same three synthetic functions. Outputs:
- `results/baselines/summary.csv`, `final_costs_*.csv`
- `results/figures/baseline_comparison_*.png`

Findings are written up in `article_draft.md` Section 5.2.1 (explicitly
flagged there as a methodology rehearsal, not the real hardware comparison).
Headline: grid search collapses hard under dimensionality at a fixed budget
— on the 10-D Sphere function it can only afford 2 points per axis, so its
best achievable result (1000) is literally the cost of being stuck at a
grid corner, 10+ orders of magnitude worse than any metaheuristic. Even
random search beats grid search by a wide margin at every dimensionality
tested. This reproduces the paper's Section 1 motivating claim numerically
rather than only citing it.

## Running the mock Phase 2 rehearsal (no hardware needed)

```
pytest tests/                              # 8 unit tests: encode/decode + objective scoring
python src/phase2_search/run_mock_rehearsal.py
```

**Disclaimer, stated as bluntly as the script itself states it: every number
this produces comes from `mock_evaluator.py`'s made-up cost surface, not a
real TinyLlama run or real Pi hardware.** It exists to prove the Phase 2
plumbing — parameter encode/decode, calibration, feasibility rejection,
PSO/GWO/DE wiring — works end to end before the same code is pointed at a
real evaluator. Outputs:
- `results/mock_phase2/best_configs.csv`
- `results/figures/mock_phase2_{convergence,boxplot}.png`

Findings are written up in `article_draft.md` Section 5.2.2, explicitly
labeled as non-hardware scaffolding to be deleted once real data exists.
Headline: all three optimizers converge to the same configuration, verified
against a brute-force grid scan to be the true optimum of the mock surface
— confirming the pipeline is wired correctly — but the mock model happens
to have no genuine multi-objective trade-off (one corner dominates on
every axis at once), so it doesn't yet rehearse the more interesting
Pareto-tension case the real hardware objective is expected to have.

## Known gaps (not yet implemented)

- The *real* Evaluator — an implementation of the `Evaluator` protocol
  (`src/phase2_search/objective.py`) backed by `src/common/llama_cpp_runner.py`
  against actual hardware, replacing `MockLlamaCppEvaluator` in
  `run_mock_rehearsal.py`'s structure. This is the actual Phase 2
  integration step once Pi + Phase 1 data exist — everything it plugs into
  is already built and tested.
- Energy measurement (`src/measurement/power_logger.py`) — needs a physical
  power meter wired up; not started.
- Thermal-throttle guard (`src/measurement/thermal_guard.py`) between
  repeated evaluations — not started; relevant once Phase 2 runs many
  evaluations back-to-back.
- `convert_hf_to_gguf.py` automation for the Phase 1 base-model step.
- A mock evaluator variant with a genuine multi-objective trade-off, to
  rehearse Pareto-tension behavior before real data exists (see Section
  5.2.2's limitation note).
