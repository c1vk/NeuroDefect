# NeuroDefect

**A Hybrid Neuro-Fuzzy-GA Decision Support System for Software Defect Prediction and Bug-Severity Assessment**

NeuroDefect is an explainable research MVP. It loads a NASA PROMISE-style software-metric dataset, performs leakage-safe preprocessing, selects features with a genetic algorithm, trains a feed-forward neural defect predictor, and derives a **risk severity assessment** with a small, documented fuzzy rule base.

NASA PC5 / JM1 / CM1 provide **binary defect labels**. They do not provide reliable four-class ground-truth severity. This system therefore reports:

- Defect Prediction (defective / non-defective)
- Defect probability from the trained neural model
- Risk Severity Assessment (derived, not ground-truth bug severity)
- Testing priority and the fuzzy rules that fired

## Architecture

```
Dataset → Validate → Clean → Stratified split
      → Fit preprocessing on training data only
      → SMOTE on training data only
      → GA feature selection on training data only
      → BPN neural predictor
      → Fuzzy risk engine
      → Streamlit decision dashboard
```

Data leakage is treated as a defect. Scalers, SMOTE, GA, and early-stopping validation never see the test set.

## Dataset requirements

A CSV (or XLSX) with numeric software metrics and a binary defect column. Common NASA names such as `defects`, `bug`, or `label` are detected automatically. If detection is ambiguous, the UI asks you to choose the column.

A bundled **synthetic CM1-like sample** lives at `data/sample/NASA_CM1_sample.csv` so the app runs without a download. For research experiments, place a real NASA file (PC5, JM1, CM1, …) in `data/raw/`.

The sample is generated from published CM1-like marginal statistics. It is **not** the official NASA distribution.

## Methodology

- **GA feature selection:** binary mask chromosome; fitness = 3-fold CV F1 of a logistic probe on SMOTE-resampled *training* data minus `λ·selected/total`. Selection frequency per feature is recorded across all generations.
- **Softmax BPN:** Dense(64)-ReLU-Dropout-Dense(32)-ReLU-Dropout-**Dense(2)-softmax**, class-weighted cross-entropy, Adam, early stopping on a train-only validation split. Defect probability = softmax class-1 output; the decision threshold is tuned on that validation split (`f1`, `accuracy_floor`, or `cost_sensitive`). PyTorch and NumPy backends implement the same model.
- **Mamdani fuzzy risk:** complexity, LOC, branch count, Halstead volume/difficulty/effort and defect probability are fuzzified (Low/Medium/High; Unlikely/Possible/Likely); 15 documented rules (metric-only escalations to HIGH/CRITICAL require `defect_prob` ≥ Possible); centroid defuzzification to 0–100 → LOW/MEDIUM/HIGH/CRITICAL.
- **GA-tuned memberships (optional):** 4 unit genes per variable, sorted and decoded into ordered overlapping triangles (always valid). Fitness = 0.6·F1(tier ≥ HIGH) + 0.4·corr(risk, label) on a train-only validation split, using **3-fold out-of-fold BPN probabilities** so the GA never sees in-sample probabilities or test data.
- **Feature selection alternatives:** binary PSO (sigmoid transfer, same fitness and evaluation budget as the GA) and a GA fitness that uses a small fast BPN (8-4 hidden units, 15 epochs, ≤1500-row holdout).
- **Optional BPN+RF ensemble:** `p = w·p_BPN + (1−w)·p_RF`; w and the threshold are tuned on 20% held-out *real* (non-SMOTE) training rows.
- **SHAP:** KernelExplainer on P(defect) with a 50-row training background, shown alongside the fired rules.
- **Leakage guards:** imputer/scaler, SMOTE, both GAs, early stopping and threshold tuning see training data only; metrics are computed once on the untouched stratified test split.

## Installation

Python 3.11+ (tested on 3.13). PyTorch (CPU) is preferred for the BPN; an equivalent NumPy implementation is used when `torch` is absent.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional
pip install -e .                 # installs the `neurodefect` CLI
```

Real NASA PROMISE data (CM1, PC1, JM1) is in `data/raw/` (see `data/raw/README.md`). PC5 was not obtainable from public mirrors.

## Dashboard

```bash
streamlit run app.py
```

Pages: Dashboard (incl. **GA diagnostics**: fitness by generation, feature-selection frequency, saved ablation figures), Dataset Analysis, Model Training, Predictions, **Review code** (scan a Python folder, or upload a scanner CSV, and read the top functions with the fired rule), Model Comparison, Explainability, Settings.

## Review a Python tree

```bash
neurodefect review src --top 10
neurodefect review src --changed
neurodefect review src --format md
```

Each row is `file:line`, the network's defective or clean call, a derived tier, and the fuzzy rule that fired. Severity is a testing priority, not a NASA bug label. Python scores are structural analogues of the NASA C/C++ metrics, not measured Python defects. The command exits 0 unless `--fail-on TIER` is set and that tier is present. `--top 0` prints every matched function.

## CLI

```bash
neurodefect scan src -o metrics.csv                      # or: python -m neurodefect scan ...
neurodefect train-shared --dataset data/raw/cm1.csv      # -> models/shared/<name>_seed42 + latest.json
neurodefect predict metrics.csv --top 20 --csv-out ranked.csv --json-out ranked.json
```

`scan` and `predict` remain for a metrics CSV you already have. `predict` ranks by the network call, then by defect probability. The tier is printed beside that call.

`scan` computes per-function metrics for Python code (radon for McCabe complexity and raw line counts; classical token-based Halstead counts by default, `--halstead radon` for radon's AST counts). Columns use NASA names:

| NASA column | Scanner source |
|---|---|
| `loc` | function physical lines |
| `v(g)` | radon cyclomatic complexity |
| `lOCode`, `lOComment`, `lOBlank` | radon raw SLOC / `#` comments / blank lines (scanned, **not used by the model**) |
| `uniq_Op`, `uniq_Opnd`, `total_Op`, `total_Opnd` | Halstead h1, h2, N1, N2 (Python tokens: operators+keywords / names, numbers, strings) |
| `n`, `v`, `l`, `d`, `i`, `e`, `b`, `t` | Halstead length, volume, level 1/d, difficulty, intelligence v/d, effort, bugs v/3000, time e/18 |
| `branchCount` | **approximation** `2·(v(g)−1)+1` |
| `ev(g)`, `iv(g)`, `locCodeAndComment` | **not computable** (need control-flow reduction / call-graph analysis / per-line code+comment counts) |

The shared-subset model trains on 15 columns: `loc, v(g), n, v, l, d, i, e, b, t, uniq_Op, uniq_Opnd, total_Op, total_Opnd, branchCount`.

Demo output (scan of this repo's `src/`, model trained on CM1) is in `reports/demo/`.

## Experiments

```bash
python scripts/evaluate.py data/raw/cm1.csv --seeds 42 7 13     # BPN vs LR / RF / SVM
python scripts/ablation.py                                       # reports/ablation.md + figures
python scripts/compare_shared.py                                 # reports/shared_vs_full.md
python scripts/cv_eval.py main --dataset CM1 --folds 5 --repeats 5   # leakage-safe repeated CV -> reports/cv/
python scripts/cv_eval.py selectors --dataset CM1                    # none vs GA(logistic) vs GA(BPN) vs PSO
python scripts/cv_report.py                                          # mean±std, Wilcoxon, Friedman -> reports/cv_results.md
```

Key configuration (`config.yaml`): `ga.method` (`ga` | `pso`), `ga.fitness_model` (`logistic` | `bpn`), `training.ensemble` (BPN+RF weighted ensemble feeding the fuzzy layer, default false), `training.threshold_mode` (`f1` | `accuracy_floor` | `cost_sensitive`), `fuzzy.ga_tune` (GA-tuned membership triangles, default **false** — see results), `ga.enabled`, `smote.enabled`.

## Cross-validated results (5×5 repeated stratified CV; JM1 2×5)

| Dataset | Hybrid F1 | BPN F1 | Ensemble F1 | LR F1 | RF F1 | SVM F1 | Friedman p |
|---|---|---|---|---|---|---|---|
| Synthetic | .710±.084 | .690±.100 | .706±.108 | .733±.095 | .732±.108 | .763±.084 | .0077 |
| CM1 | .335±.081 | .289±.116 | .335±.103 | .372±.091 | .242±.098 | .345±.084 | .00031 |
| PC1 | .310±.055 | .354±.076 | .286±.085 | .360±.055 | .361±.054 | .329±.054 | .00040 |
| JM1 | .404±.011 | .436±.010 | .428±.022 | .440±.022 | .446±.017 | .447±.018 | .00026 |

Wilcoxon p-values, the selector comparison (GA vs PSO vs GA with BPN fitness vs none) and MCC are in `reports/cv_results.md`. Defaults chosen from these results: logistic-probe GA fitness, GA selector, ensemble off.

## Single-split results (held-out 30% test, mean over seeds 42/7/13)

Softmax BPN vs baselines (all features for BPN/LR/RF/SVM rows):

| Dataset | Model | Acc | Prec | Rec | F1 | ROC-AUC | MCC |
|---|---|---|---|---|---|---|---|
| Synthetic sample | LR / RF / SVM F1 | | | | .670 / .702 / .710 | | |
| Synthetic sample | BPN (before: sigmoid) | .388 | .264 | .933 | .375 | .881 | .204 |
| Synthetic sample | BPN (softmax) | .913 | .700 | .711 | .698 | .945 | .653 |
| CM1 | LR / RF / SVM F1 | | | | .377 / .352 / .344 | | |
| CM1 | BPN (before) | .358 | .139 | .690 | .192 | .664 | .001 |
| CM1 | BPN (softmax) | .815 | .262 | .405 | .316 | .772 | .223 |
| PC1 | LR / RF / SVM F1 | | | | .349 / .393 / .362 | | |
| PC1 | BPN (before) | .483 | .128 | .778 | .212 | .741 | .137 |
| PC1 | BPN (softmax) | .861 | .253 | .444 | .314 | .795 | .262 |
| JM1 | LR / RF / SVM F1 | | | | .444 / .452 / .448 | | |
| JM1 | BPN (before) | .709 | .377 | .451 | .411 | .677 | .221 |
| JM1 | BPN (softmax) | .560 | .303 | .729 | .427 | .675 | .202 |

Ablations, fixed vs GA-tuned fuzzy and full vs shared-subset tables: `reports/ablation.md`, `reports/shared_vs_full.md`, summary in `reports/SUMMARY.md`.

## Known limitations

- Risk severity is **derived**, not labelled by NASA.
- **Language gap:** NASA metrics come from C/C++ (CM1, PC1) and C (JM1) code; the scanner measures Python. Equal complexity does not imply equal risk across languages, so predictions on Python are indicative, not validated.
- **Metric counting differences:** radon and the NASA MDP extractor count differently. Halstead token rules differ (radon's AST mode counts only arithmetic/boolean operators and gives ~15–20× smaller values; the default token mode matches CM1 medians closely, e.g. n 59 vs 67.5, v 281 vs 330). `branchCount` is approximated from `v(g)`. PROMISE line columns (`lOCode` median 1 vs `loc` 17 on CM1) are inconsistent, so they are excluded from the shared model.
- Dropping `ev(g)`, `iv(g)` and the line columns costs F1 on CM1 (.325 → .242) but not on PC1/JM1 (see `reports/shared_vs_full.md`).
- Python functions much larger than anything in CM1 can get extreme probabilities (the network extrapolates).
- GA-tuned fuzzy memberships help on the synthetic sample but not reliably on real NASA data, so they are off by default.
- On real NASA data the BPN and Hybrid are still at or below LR / RF / SVM F1 in cross-validation (Friedman p < .001 on CM1/PC1/JM1; the best mean rank goes to LR or RF). The Hybrid is significantly better than RF only on CM1.
- Repeated-CV Wilcoxon p-values are optimistic (overlapping training folds). JM1 used 2×5 CV for time.
- SHAP values are computed on scaled, selected features with KernelExplainer (approximate, 200 samples).
- GA fitness uses a logistic probe, not the BPN (computational safety). PC5 not included.

## Project layout

See `src/` for data, models, GA, fuzzy, evaluation, and the training pipeline. `app.py` only handles Streamlit navigation, session state, and calls into those modules. Styles live in `assets/styles.css`.

## Tests

```bash
pytest -q
```

The suite covers loading, cleaning, leakage-safe scaling and SMOTE, GA chromosome validity, softmax BPN (both backends, gradient check, threshold modes), portable model loading, fuzzy membership validity and GA tuning, pipeline ablation variants, plots, the radon scanner, and the CLI end to end.

## Research roadmap

Phase 2: PSO selector, GA vs PSO, multi-seed ablation.  
Phase 3: PC5 ↔ JM1 cross-project validation.  
Phase 4: Eclipse JDT/PDE ground-truth severity.  
Phase 5: Friedman / Wilcoxon significance tests.  
Phase 6: SHAP and feature stability.  
Phase 7: FastAPI + React production split.

## Citation / references

- NASA PROMISE software defect datasets (CM1, JM1, PC5, …)
- Halstead, McCabe software science / complexity metrics
- Chawla et al., SMOTE
- DEAP genetic algorithms
- Mamdani fuzzy inference

Faculty question this MVP is built to answer: *Where did this prediction come from?* The UI can show the selected features, the neural probability, the high risk metrics, the fuzzy rules that fired, why the severity band was assigned, and which configuration produced the run.
