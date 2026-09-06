# hyptune

Code and data for **"Metaheuristic Co-Optimization of Compression and Runtime
Configuration for Embedded LLM Inference: A Case Study on TinyLlama and
Raspberry Pi 5"** — `article_draft.md` (markdown working draft) and
`latex/article.tex` (IEEEtran journal-format LaTeX version, kept in sync with
the markdown draft) are the two article files; this README documents the
code, the experimental status, and — in full detail — everything that still
needs a human before this is submission-ready.

**Project status as of this writing: all experimental work is complete.**
Phase 1 (exhaustive quantization sweep) and Phase 2 (metaheuristic runtime
search) have both run to completion on real Raspberry Pi 5 hardware across
all 7 Pareto-retained quantization variants. The article's central finding —
the **optimizer's curse** as a pervasive, measurable bias in hardware-in-the-
loop metaheuristic search — is fully written up and the LaTeX compiles
cleanly. What remains is enumerated exhaustively in
["Remaining Work Before Submission"](#remaining-work-before-submission) below.

---

## What's actually implemented and run

Everything below has been run for real, not simulated or mocked:

- **Phase 1 — exhaustive quantization-variant sweep** (`src/phase1_quant_sweep.py`,
  article Section 3.4/5.1). Run on the real Pi 5 target. Output:
  `results/phase1_table.csv` (7 variants: Q2_K, Q3_K_M, Q4_0, Q4_K_M, Q5_K_M,
  Q6_K, Q8_0 — perplexity, peak RSS, tokens/s, file size) and
  `results/phase1_retained.yaml` (all 7 variants Pareto-nondominated — see
  the article for why that's itself a finding).
- **Real Phase 2 hardware-in-the-loop search** (`src/phase2_search/run_real_phase2.py`,
  article Section 5.3 onward). Runs PSO, GWO, DE, grid search, and random
  search against the real objective — real latency, real memory, real
  on-device energy (via the Pi 5's PMIC, see below), real perplexity — for
  all 7 variants. Includes a thermal-cooldown safety check
  (`wait_for_cooldown()` in the same file) added after a mid-run Pi crash.
  Output: `results/phase2_runs/<variant>/{convergence,final_costs,best_configs}.csv`
  and `meta.yaml` per variant, plus figures in `results/figures/`.
- **Real energy measurement** (`src/measurement/power_logger.py`) — reads the
  Raspberry Pi 5's on-board PMIC directly via `vcgencmd pmic_read_adc`
  (per-rail voltage/current), no external power meter needed. This is a
  genuinely working, validated solution (sanity-tested against a `sleep 3`
  baseline), not a placeholder — an earlier draft of this README incorrectly
  listed this as "needs a physical power meter, not started," which was
  stale even before this rewrite.
- **The real `Evaluator`** (`src/phase2_search/llama_cpp_evaluator.py`) —
  `LlamaCppEvaluator`, backed by `src/common/llama_cpp_runner.py`, driving
  `llama-bench`/`llama-quantize`/`llama-perplexity` against the actual
  hardware. This produced every real number in the article's Results section.
- **Metaheuristic implementations + synthetic-function validation**
  (`src/phase2_search/{pso,gwo,de}.py`, `src/validation/`, article Section
  3.8/5.2). Pure Python/numpy, no hardware dependency.
- **Grid/random search baselines** (`src/baselines/`, article Section
  4.3/5.2.1 and, on the real objective, Section 5.6) — run both against
  synthetic functions and against the real hardware objective for all 7
  variants.
- **The optimizer's-curse reconciliation check** (`src/analysis/check_reconciliation.py`)
  — recomputes every reported "best" configuration's scalarized cost from
  its own independently re-measured raw metrics and compares against the
  recorded search-time cost. This is what surfaced the article's central
  finding (Section 5.5): metaheuristics' iterative convergence makes their
  recorded costs a biased, overly optimistic estimator of true cost, badly
  so on some variants (up to 180× on Q8_0's GWO) and negligibly on others
  (Q4_0, Q6_K).
- **Objective-function plumbing + unit tests** (`src/phase2_search/{parameter_space,objective}.py`,
  `tests/`) — the parameter encode/decode layer and the scoring/
  normalization/feasibility-rejection logic, unit-tested (`pytest tests/`).
- `src/phase2_search/{mock_evaluator,run_mock_rehearsal}.py` — an early,
  synthetic-cost-surface evaluator used **before real hardware access** to
  validate the Phase 2 plumbing end to end. This is no longer referenced in
  the article (its own scaffolding note said it should be removed once real
  data existed, and real data now exists for all 7 variants) but the code is
  left in the repo as a genuine historical/development artifact and because
  it's still a fast, hardware-free way to smoke-test changes to the
  optimizer wiring.

## Reproducing the results

### Prerequisites confirmed to work (documented here since llama.cpp's CLI
flags have changed across releases and will likely change again)

- Raspberry Pi 5, 8 GB RAM, active (fan) cooling. Confirmed working on
  Debian GNU/Linux 13 (trixie), kernel `6.12.47+rpt-rpi-2712`.
- llama.cpp built from source at commit `adb55e5148dc93bcdca7212a2d1df3ccc422959a`
  (`cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF && cmake --build build -j4`).
  Newer commits may rename flags `src/common/llama_cpp_runner.py` depends on
  — re-run `llama-quantize --help`, `llama-bench --help`,
  `llama-perplexity --help` and diff against that file's assumptions before
  trusting new numbers.
- CPU governor set to `performance` on all 4 cores (was `ondemand` by
  default) to reduce DVFS-induced measurement noise.
- Base model: `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`'s `Q8_0` file, used as
  the requantization *source* for all 7 variants (no F16 GGUF was available
  on the HF Hub for this model at the time — see article Section 3.2 for the
  full deviation note; this is a real, disclosed methodological choice, not
  an oversight).
- Perplexity corpus: WikiText-2 raw test split (`data/prompts/wikitext2_raw_test.txt`
  for Phase 1, a smaller `wikitext2_quality_slice.txt` for Phase 2's
  per-KV-cache-type quality precomputation). These corpus files are pulled
  onto the Pi directly and are not committed to this repo (large, standard,
  separately licensed).

```
pip install -r requirements.txt
```

### Phase 1 (run on the Pi)

```
python src/phase1_quant_sweep.py --config config/phase1.yaml
```

### Real Phase 2 (run on the Pi; takes several hours for all 7 variants)

```
python3 -u src/phase2_search/run_real_phase2.py
```

Edit `PHASE2_VARIANT_SUBSET` at the top of the file to change which variants
run (currently set to the 4 variants that were added in the second batch —
`Q3_K_M, Q4_0, Q5_K_M, Q6_K` — since `Q4_K_M, Q2_K, Q8_0` were run in an
earlier invocation; set it to all 7 for a from-scratch run). Writes
incrementally after every algorithm (not just at the end of a variant), so a
crash loses at most one algorithm's progress, not the whole run — this was
added after a real mid-run Pi crash, see the article's Limitations section.

### Everything else (no hardware needed, all pure Python/numpy)

```
python src/validation/run_validation.py                  # metaheuristic validation, article Sec. 5.2
python src/validation/plot_function_representations.py   # Table-3-style 3D surface plots
python src/baselines/compare_baselines.py                 # baseline rehearsal on synthetic functions, Sec. 5.2.1
pytest tests/                                              # 8 unit tests: encode/decode + objective scoring
python src/analysis/check_reconciliation.py                # the optimizer's-curse check, Sec. 5.5 — needs results/phase2_runs/ populated first
```

### Building the LaTeX article

```
cd latex
pdflatex article.tex
pdflatex article.tex   # second pass to resolve cross-references/citations
```

Requires `IEEEtran.cls`, `booktabs`, `subcaption`, `algorithm`/`algpseudocode`
(all in a standard TeX Live install). Figures are pulled from
`../results/figures/` via `\graphicspath`. As of this writing it compiles
cleanly to 24 pages with no errors and no undefined references.

---

## Remaining Work Before Submission

This section is meant to be exhaustive. Nothing below is blocking the
*technical* content of the paper — every experimental result, every figure,
every table, and the central optimizer's-curse finding are done, written up,
and verified in both `article_draft.md` and `latex/article.tex`. Everything
below is editorial, verification, or a genuine judgment call that only the
authors (or someone with tooling/access a coding assistant doesn't have) can
make.

### 1. Author-facing declarations (currently absent from the article entirely)

The article previously had **Funding**, **Author Contributions (CRediT)**,
and **Conflict of Interest** sections filled with defensible *defaults* (no
external funding; a conventional PhD-candidate-plus-supervisors CRediT
split; no known conflicts) rather than blank placeholders. These were
**removed entirely at the corresponding author's explicit request** (they
were still unverified claims about real people's real circumstances). They
now need to be **written from scratch with real information** before
submission — most IEEE and IAES-family venues require all three as a hard
submission requirement, so the paper cannot go out without them.

- **Funding**: was there any grant, institutional funding line, or hardware
  donation (e.g., did LIDSI or the university provide the Raspberry Pi 5, or
  was it personally purchased)? If genuinely none, the standard sentence is
  *"This research received no specific grant from any funding agency in the
  public, commercial, or not-for-profit sectors,"* but this should be a
  confirmed fact, not an assumption.
- **CRediT (Contributor Roles Taxonomy)**: needs sign-off from Fahd
  Koraiche, Rachid Dehbi, and Amine Dehbi on who did what, using the
  standard 14 CRediT roles (Conceptualization, Methodology, Software,
  Validation, Formal analysis, Investigation, Resources, Data Curation,
  Writing – Original Draft, Writing – Review & Editing, Visualization,
  Supervision, Project Administration, Funding Acquisition). A previous
  draft assigned Fahd Koraiche essentially all hands-on roles and Rachid/
  Amine Dehbi supervision + review — but whether that's actually accurate
  (e.g., did either co-author write code, run analysis, or contribute
  specific prose beyond review?) is something only the three of you know.
- **Conflict of Interest**: standard boilerplate (*"The authors declare no
  known competing financial interests or personal relationships that could
  have influenced the work reported in this paper"*) is almost certainly
  fine here, but should be explicitly confirmed rather than assumed —
  e.g., does any author have a consulting relationship, equity, or
  employment tie to a company whose products are discussed (Raspberry Pi
  Foundation, any cloud/LLM vendor, etc.)?

### 2. A full database-backed literature review

What exists now: **three passes of general web search** (2026-08-01,
2026-08-17, 2026-08-18), each time finding candidate papers by keyword
search and then verifying author names / venue / year / pages against at
least one independent source before citing. This located and verified 20
real references, including the two closest pieces of prior art found —
LLMPi (`ref19`, CVPRW 2025 EDGE workshop) and DeepQGHO (`ref20`, IEEE Access
2022) — and used DeepQGHO specifically to stress-test the novelty claim
(Section 2.4 names exactly which of the claim's 4 defining criteria it does
and doesn't satisfy).

What this is **not** a substitute for, and what still needs to happen:

- **Citation-graph traversal** using Google Scholar, Semantic Scholar, Web
  of Science, or Scopus — tools this assistant does not have API/browser
  access to. Specifically:
  - **Forward citation chasing**: who has cited DeepQGHO (2022), LLMPi
    (2025), or HAQ (2019) *since* they were published? A 2023–2025 paper
    that cites DeepQGHO and extends it toward population-based metaheuristics
    or LLM inference specifically would be the single most important thing
    to find before submission, since it would directly bear on the novelty
    claim. A keyword web search will not reliably surface this unless the
    paper's title happens to contain the right words; a citation-graph tool
    will surface it directly via DeepQGHO's "Cited by" list.
  - **Backward citation chasing**: what does DeepQGHO itself cite? Working
    backward through its references could surface earlier foundational work
    in the same niche (physical-hardware-in-the-loop + hyperparameter
    search) that never appeared in keyword search results at all.
  - **Structured/field-limited search** (title-only, abstract-only, venue
    filters, citation-count sorting) — general web search ranks by textual
    relevance to the query, not by academic influence, and can bury a
    highly-cited but plainly-titled paper under blog posts and tutorials.
- **Estimated effort**: roughly 30–60 minutes for someone with Google
  Scholar access, specifically: pull up DeepQGHO and LLMPi, check each
  paper's "Cited by" list, skim DeepQGHO's own reference list, and check
  specifically for anything combining (a) a population-based metaheuristic
  (PSO/GWO/DE or similar) with (b) LLM (not CNN) inference-time (not
  training-time) tuning on (c) physical (not simulated) low-power hardware.
  If nothing closer than DeepQGHO turns up, the novelty claim as written
  stands unmodified. If something does, Section 2.4 (`sec:gap` in LaTeX)
  needs one more paragraph in the same style as the DeepQGHO discussion.
- **Two specific bibliography entries worth a final independent check**,
  even though both are already resolved from multiple sources:
  - `ref12` (HAQ, Wang et al., CVPR 2019): page range confirmed as
    8612–8620 via DBLP, which itself resolved a discrepancy between two
    other secondary sources (one gave 8604–8612). DBLP is generally
    reliable but a direct look at IEEE Xplore's own record would be the
    fully authoritative source if there's any remaining doubt.
  - `ref13` (CEG4N, Matos Jr. et al., NSV/FoMLAS 2022): this replaced an
    earlier candidate (Miccini et al., tinyML 2024) that was checked and
    found *not* to match the originally-intended description (a
    genetic-algorithm-vs-brute-force-enumeration comparison). CEG4N was
    confirmed via its full HTML text (not just its abstract) to use an
    NSGA-II genetic algorithm and to explicitly frame this against
    brute-force combinatorics (52⁷ combinations for a 7-layer network) — a
    real match, but worth a final skim of the actual paper if time allows,
    since it was found via a second, more targeted search rather than the
    original search that identified this citation gap in the first place.

### 3. Target-venue-specific formatting

The LaTeX is currently `\documentclass[journal]{IEEEtran}` — a generic IEEE
journal template, not tailored to any specific venue's house style. Before
submission:

- **Confirm the actual target venue** (IAES International Journal of
  Artificial Intelligence, given this paper explicitly mirrors Nouna et
  al. 2025's structure and is cited throughout as `ref11`? A different IEEE
  Transactions/journal? A conference?) and re-typeset against that venue's
  actual template/class file if it differs from generic IEEEtran (column
  widths, reference style, abstract length limits, keyword count limits all
  vary by venue).
- **Page/word limits**: the paper is currently 24 pages in IEEEtran journal
  two-column format. Check the target venue's page limit — some journals cap
  at 8–12 pages for this format, which could require condensing the very
  extensive Phase 2 per-variant results tables (Sections 5.3–5.6 currently
  give full tables for all 7 variants; a page-limited venue might need these
  moved to supplementary material with only the cross-variant summary table
  kept in the main text).
- **Running header**: the `\markboth{Draft, \today}{}` line that put "DRAFT,
  [date]" on every page has been removed entirely (per the "no draft
  scaffolding" cleanup). IEEEtran's default running header is now in effect
  (typically blank or auto-generated); most venues want a short-author-list
  / short-title running head set explicitly via `\markboth{short authors}{short title}`
  once the paper is in near-final form — trivial to add back with real
  content, but currently just absent.
- **ORCID iDs**: not currently included in the author block; many IEEE and
  Elsevier-family venues now require or strongly encourage them.
- **IEEE copyright notice / conference-vs-journal formatting differences**:
  if the eventual target is an IEEE conference rather than a journal, the
  document class and several formatting conventions (two-column from page 1,
  copyright footer, page limit) differ from the current journal template.

### 4. Repository visibility

`https://github.com/Koraiche/phd-hypetune-service.git` is currently
**private** (confirmed via an unauthenticated fetch returning 404, noted in
the article's Data Availability section). The article states it "will be
made public upon publication" — this needs to actually happen at the
appropriate point in the submission/publication process (some venues want
the repository public *at submission time* for review, not just after
acceptance; check the target venue's data/code availability policy).

### 5. Final human read-through

No matter how much verification has been done at each step, a full,
uninterrupted read-through by a human co-author before submission is
standard practice and has not substituted for anything here — this document
was built incrementally, section by section, over an extended session, and
even with careful internal consistency checks at each stage, a cover-to-cover
read is the only way to catch the kind of small drift (a stale
cross-reference, a sentence that made sense in an earlier draft's context
but reads oddly now) that a tool built to check specific things can miss
between checks.

### 6. Optional but genuinely strengthening: acting on the paper's own Future Work list

These are not required for submission — the paper is complete and honest
without them — but each one is directly motivated by a specific finding in
this paper (not generic future work), and doing any of them before
submission would make the paper stronger rather than just longer. In the
priority order the paper itself gives (Conclusion, Section 7):

1. **Fix the optimizer's-curse artifact directly** rather than only
   disclosing it: set `bench_repetitions > 1` in the real Phase 2 evaluator
   calls (currently 1, i.e. every hardware measurement is a single noisy
   sample) and add an automated check that re-verifies a recorded "best"
   cost against a repetitions-averaged re-measurement before it's reported
   as a winner. `src/analysis/check_reconciliation.py`'s logic could be
   adapted to run this check live inside `run_real_phase2.py` rather than as
   a manual post-hoc script. This is explicitly named as the single
   highest-priority follow-up in the paper's own Conclusion.
2. **Checkpoint/resume capability** for `run_real_phase2.py`, motivated
   directly by the real mid-run Pi crash this project experienced — right
   now, a crash mid-algorithm loses that algorithm's progress even though
   per-algorithm incremental CSVs already protect against losing a whole
   variant.
3. **Multi-objective Pareto-front search** (NSGA-II/MOEA-D) instead of the
   current fixed-weight scalarization (`ObjectiveWeights`: α=0.35 latency,
   β=0.35 quality, γ=0.2 memory, δ=0.1 energy — explicitly flagged in the
   code as illustrative defaults, not tuned or empirically justified).
4. **A downstream task-accuracy quality metric** instead of perplexity,
   which would re-open the temperature/top-p search dimensions that were
   dropped from Phase 2's search space specifically because perplexity
   cannot distinguish their effect on generation quality.
5. **A larger or online-adaptive calibration sample size** — `N_CALIBRATION_SAMPLES=8`
   in `run_real_phase2.py` is small enough that it directly caused a
   real, disclosed artifact on Q2_K (two methods' costs clipped to an exact
   calibration floor of 0.0000).
6. Extending to other edge platforms (Jetson Orin Nano, mobile SoCs) or
   other small models (Phi-3-mini, Qwen2.5-0.5B/1.5B, Gemma-2B), to test
   whether the variant-to-variant instability found here is specific to
   this hardware/model pair or a more general property of hardware-in-the-
   loop metaheuristic search under tight evaluation budgets.
7. Hybrid metaheuristics (e.g., PSO–DE hybrids), and extending the discrete
   Phase 1 axis beyond quantization to structured pruning or LoRA-merged
   distillation.
8. A full `thermal_guard.py` module — what exists today
   (`wait_for_cooldown()` inside `run_real_phase2.py`) only gates *between*
   variants (checks temperature before starting a new one); it does not
   guard between individual evaluations within a variant, which is where
   most of the thermal load actually accumulates during a long run.

---

## Known code-level gaps (not required for the paper, listed for completeness)

- `convert_hf_to_gguf.py` automation for the Phase 1 base-model step was
  never built — a real F16 GGUF for TinyLlama-1.1B-Chat-v1.0 was never
  found on the HF Hub, so Q8_0 was used as the requantization source
  instead (a disclosed, deliberate methodological choice, not a bug — see
  article Section 3.2). If a real F16 source is ever needed, this
  conversion step would need to be scripted; it currently isn't.
- `src/phase2_search/{mock_evaluator,run_mock_rehearsal}.py` and
  `results/mock_phase2/` still exist in the repo (kept as a real
  development artifact and fast hardware-free smoke test) but are no
  longer referenced anywhere in the article text, per the "remove
  scaffolding once real data exists" note that shipped with that code from
  the start.
- Model load time was never instrumented separately from total benchmark
  wall time (`llama_cpp_runner.py` doesn't isolate it) — disclosed as a
  limitation in the article rather than silently omitted, but the
  instrumentation itself was never added.
