# Ensemble — a stacking movie recommender

A working reconstruction of a heterogeneous **stacking ensemble** for movie rating
prediction, plus the recommender built on top of it. Rate a handful of films, get
a ranked Top-K list, and inspect exactly which model produced each number.

**Research basis.** Nisha Sharma and Dr. Mala Dutta, *“An Ensemble Movie
Recommender System Based on Stacking”*, Journal of Theoretical and Applied
Information Technology, Vol. 101, No. 18, 30 September 2023.

**Paper claim, reproduced or not:**
`KNN + XGBoost + Gradient Boosting → Linear Regression`

The project is deliberately split into two modes:

| Mode | Feature set | What it is |
| --- | --- | --- |
| **Paper reproduction** | `user_id`, `movie_id`, `year` | the paper's stated features, nothing else |
| **Enhanced** (labelled everywhere) | + movie popularity, movie mean rating, user activity, user mean rating | our documented improvement, never mixed into the reproduction tables |

Published values are reference-only. They live in one read-only file
(`src/ml/reference.ts`) and render only under the heading *“Published Paper
Reference Results”*. Every other number in the app was computed by this code.

---

## What version 1 does

Scope: **rate films, get Top-K picks, and demo the paper honestly.**

* `/` — landing: architecture, published reference results, integrity rules.
* `/rate` — build a taste profile from a starter set or any of the 1,682 titles.
* `/recommendations` — Top-K (5/10/20/50), accuracy vs diversity ranking, per-item
  model decomposition and evidence-based explanations.
* `/browse` — catalogue explorer with genre filters and predicted ratings.
* `/analytics` — run the experiment, compare published vs measured results, read
  the charts, inspect meta-learner weights and feature importance.
* `/dashboard` — account overview, persona match, Top-3 preview.

Deferred to version 2 (and documented rather than stubbed): the Python/FastAPI
service layer, PostgreSQL port, Docker/compose deployment files, MLflow tracking,
and full TreeSHAP. See `docs/ARCHITECTURE.md` §10 for the phase table.

## Dataset

MovieLens 100K (October 1998 release), bundled at `public/data/ml-100k/`:
100,000 ratings · 943 users · 1,682 movies · 19 genres · global mean rating 3.53.
The data ships with the app so runs are reproducible without network access.

## Quick start

```bash
bun install
bun run typecheck        # tsc -b --noEmit
bun test tests/          # 32 tests
bun run experiment --scope full          # offline paper-mode reproduction (~51 s)
bun run experiment --mode enhanced       # enhanced features (~92 s)
```

In the app: sign in → **Rate films** → **Models & experiments → Run experiment**
→ **Top-K picks**. Training runs in a web worker; nothing is retrained per
request.

## Architecture

```
MovieLens files ──► web worker (training + serving)  ◄──typed messages──► React UI
                          │                                                  │
                          └────────── Convex: ratings, prefs, model runs ◄───┘
```

| Layer | Implementation |
| --- | --- |
| ML | TypeScript from scratch — histogram CART, KD-tree KNN, Random Forest, AdaBoost.R2, Gradient Boosting, XGBoost-style 2nd-order booster, Linear Regression |
| Orchestration | `src/ml/experiments.ts` (standalone + 7 stacking configs), `src/ml/stacking.ts` (k-fold OOF) |
| Offline job | `scripts/run-experiment.ts` → `experiments/results/*.json` |
| Serving | worker `recommend` / `score-movies` handlers over resident fitted models |
| Backend | Convex: `ratings`, `profiles`, `modelRuns`, `experimentResults` |
| UI | React 19, React Router 7, Tailwind v4, shadcn/ui, Framer Motion, Recharts |

Full detail: `docs/ARCHITECTURE.md`.

### Data preprocessing

Merge `u.data` ⨝ `u.item` ⨝ `u.user` into one table, then audit: no duplicate
(user, movie) pairs, no out-of-range ratings, no dropped rows; **9** rows lack a
usable release year and are imputed with the *training-split median*. User
demographics are merged and then excluded from modelling, as the paper states.

### Feature engineering

Paper mode encodes `user_id`, `movie_id` and `year` (label encoding for the
categorical ids; unseen categories → bucket 0). Encoders and every statistic are
fitted on training rows only. Enhanced mode adds four supervised aggregates with
leave-one-out encoding for training rows and a per-fold refit of the feature
space, so a row never contributes to its own feature. Genre flags are implemented
and off by default (`--genres`).

### Evaluation

`MAE = (1/n)Σ|y − ŷ|`, `MSE = (1/n)Σ(y − ŷ)²`, `RMSE = √MSE`, reported for every
standalone learner, all seven stacking configurations, the proposed model, and the
enhanced model. Also produced per run: OOF RMSE, residuals, error histogram,
actual-vs-predicted scatter, base-prediction correlation matrix, meta-learner
weights and split-gain feature importance.

## Stacking methodology

```
1. split training rows into K folds (default 5, seeded)
2. for each fold: fit each base learner on K−1 folds, predict the held-out fold
   → out-of-fold (OOF) prediction matrix          ← no in-sample predictions
3. train the meta learner on [OOF_knn, OOF_xgb, OOF_gb, …] against the target
4. refit each base learner on the complete training split  ← serving models
5. run the test set through those, assemble the meta input matrix
6. predict with the trained meta learner and evaluate against the test target
```

The OOF matrix is computed once per run and every candidate meta configuration
reuses it, so all seven configurations are trained identically but evaluated
cheaply.

### Base learners and meta learner

| Base learner | Notes |
| --- | --- |
| Linear Regression | OLS on standardized features; also the meta learner |
| K-Nearest Neighbours | kd-tree, k=40, standardized, `k` unpublished by the paper |
| Random Forest | 80 bagged CART trees, bootstrap sampling |
| AdaBoost.R2 | 50 weighted regression trees, weighted-median combination |
| Gradient Boosting | 120 stages, depth 3, learning rate 0.1, squared loss |
| XGBoost | **reimplementation of the algorithm**, not the official library: 180 rounds, depth 6, `λ=1`, subsampling |

Meta learners: Linear Regression or XGBoost, per configuration.
Hyperparameters are unpublished by the paper; ours are recorded per run.

## Results

### Published paper reference results (not ours)

| Model | Paper RMSE |
| --- | --- |
| Linear Regression | 1.08 |
| XGBoost | 0.92 |
| Random Forest | 0.98 |
| AdaBoost | 0.96 |
| Gradient Boosting | 0.99 |
| KNN | 1.07 |

Stacking (paper): `XGB+KNN→LR` 0.91 · `XGB+KNN→XGB` 0.92 · `KNN+GB+RF→XGB` 0.94 ·
`KNN+XGB+GB→XGB` 0.91 · **`KNN+XGB+GB→LR` 0.90** · `KNN+RF→XGB` 0.95 ·
`KNN+XGB+GB+AB→LR` 0.91. Proposed model (paper): MAE 0.69 · MSE 0.82 · RMSE 0.90.

### Our reproduction results (paper mode, full 100K, seed 42, 5-fold, 51 s)

| Model | MAE | MSE | RMSE | Paper | Δ |
| --- | --- | --- | --- | --- | --- |
| Linear Regression | 0.8928 | 1.1783 | **1.0855** | 1.08 | +0.0055 |
| K-Nearest Neighbours | 0.8660 | 1.1465 | **1.0708** | 1.07 | +0.0008 |
| Random Forest | 0.8435 | 1.0970 | **1.0474** | 0.98 | +0.0674 |
| AdaBoost.R2 | 0.8884 | 1.1604 | **1.0772** | 0.96 | +0.1172 |
| Gradient Boosting | 0.8544 | 1.1102 | **1.0537** | 0.99 | +0.0637 |
| XGBoost | 0.8326 | 1.0657 | **1.0323** | 0.92 | +0.1123 |

| # | Configuration | MAE | MSE | RMSE | Paper | Δ |
| --- | --- | --- | --- | --- | --- | --- |
| A | XGB + KNN → LR | 0.8323 | 1.0652 | 1.0321 | 0.91 | +0.1221 |
| B | XGB + KNN → XGB | 0.8319 | 1.0684 | 1.0336 | 0.92 | +0.1136 |
| C | KNN + GB + RF → XGB | 0.8396 | 1.0869 | 1.0425 | 0.94 | +0.1025 |
| D | KNN + XGB + GB → XGB | 0.8311 | 1.0681 | 1.0335 | 0.91 | +0.1235 |
| **E** | **KNN + XGB + GB → LR (proposed)** | 0.8323 | 1.0652 | **1.0321** | 0.90 | +0.1321 |
| F | KNN + RF → XGB | 0.8421 | 1.0948 | 1.0463 | 0.95 | +0.0963 |
| G | KNN + XGB + GB + AB → LR | 0.8323 | 1.0652 | 1.0321 | 0.91 | +0.1221 |

Lowest-RMSE configuration in this experiment: **G**, tied with A and E to four
decimals. That is an experimental ordering, not a claim of superiority.

**So the architecture reproduces and the headline numbers do not.** Linear
Regression and KNN land within 0.01 RMSE of the published values; the four tuned
ensembles and the stack land 0.06–0.13 above. Error analysis explains the
mechanism: with only ids and a year the model compresses toward the mean
(prediction SD 0.42 against a target SD 1.09, correlation 0.376), which is why
the meta learner collapses onto XGBoost (weight 1.055) with a slightly negative
KNN weight. Full discussion in `docs/RESEARCH-REPORT.md` §15.

### Enhanced results (full 100K, 7 features, 92 s — a labelled deviation)

| Model | RMSE | Paper | Δ |
| --- | --- | --- | --- |
| Linear Regression | 0.9536 | 1.08 | −0.1264 |
| K-Nearest Neighbours | 0.9679 | 1.07 | −0.1021 |
| Random Forest | 0.9519 | 0.98 | −0.0281 |
| AdaBoost.R2 | 0.9724 | 0.96 | +0.0124 |
| Gradient Boosting | 0.9472 | 0.99 | −0.0428 |
| XGBoost | 0.9578 | 0.92 | +0.0378 |
| **Proposed stack E** | **0.9450** | 0.90 | +0.0450 |

With informative features the same code reaches within 0.045 RMSE of the paper's
proposed model, the meta weights spread across all three learners
(KNN +0.194, XGBoost +0.264, Gradient Boosting +0.599), and residual SD falls from
1.0106 to 0.9224. That is evidence the reproduction gap is an information gap,
not a broken implementation — and it is *not* a like-for-like reproduction, since
the aggregates are target-derived.

## Recommendation engine

1. Candidates = the catalogue minus everything **you** rated and everything the
   matched MovieLens **persona** rated.
2. Your ratings map onto the closest MovieLens user by cosine similarity over
   co-rated films (≥ 8 shared films required). The paper has no new-user story, so
   this bridge is ours and is named in every explanation that uses it.
3. KNN, XGBoost and Gradient Boosting score the candidates; the Linear Regression
   meta learner combines them.
4. Optional content-affinity blend from your own genre history
   (`affinityWeight`, default 0.30, `0.00` = pure paper model).
5. **Accuracy mode** ranks by relevance; **diversity mode** re-ranks with Maximum
   Marginal Relevance (`λ` exposed in the UI, default 0.65).
6. Each item reports its predicted rating, base-learner breakdown, learner spread
   (a disagreement measure, not a probability), dataset support, and an
   explanation assembled only from evidence that exists.

**Cold start is labelled, not faked.** No ratings → popularity + Bayesian average
rating, source `cold_start_fallback`, `predictedRating: null`, and a caveat on
every card. Ratings but no persona match → `no_persona_fallback`. The paper does
not address cold start, and no such mechanism is attributed to it.

## Tests

`bun test tests/` — 32 tests, 1,115 assertions:

* **dataset** — parsers, genre order, year extraction, merge counts, missing-year audit, popularity stats
* **metrics / splitting** — MAE/MSE/RMSE definitions, seeded reproducibility, k-fold disjointness, temporal ordering
* **features** — paper-mode column set, train-median imputation, leave-one-out encoding for enhanced mode
* **stacking** — out-of-fold predictions are *not* in-sample (1-NN on noise targets), meta input naming, missing-member errors
* **models** — tree ensembles beat the mean baseline, importance normalisation, tree output bounds
* **recommender** — rated-film and persona-history exclusion, explanation content, MMR lowering intra-list similarity, cold-start labelling
* **reference integrity** — published constants stay reference-only, configurations match the paper's combinations

## Known limitations

* Hyperparameters, seeds, split ratio and fold count are unpublished, so this is a
  methodology reproduction, not a numeric one.
* XGBoost is reimplemented from scratch; Random Forest, Gradient Boosting and
  AdaBoost use quantile-binned (64-bin) split search rather than exact splits.
* The paper publishes RMSE only for most rows; MAE/MSE comparisons are one-sided.
* RMSE is not a ranking metric — no published precision@k/NDCG baseline exists to
  compare the Top-K lists against, so list quality is reported qualitatively
  (genre coverage, intra-list similarity, support).
* Enhanced mode's aggregates are target-derived and thus not comparable to the
  paper's feature set.
* Model artifacts are not persisted: reloading the page requires a retrain
  (~51 s paper mode, ~92 s enhanced).
* KNN degrades past six features; its neighbour pool is capped and the cap is
  recorded in the model parameters.

## Future work

Hyperparameter search modes (`quick` / `research` / `production` — only `research`
is used in the reported run) · ranking metrics with a temporal protocol · exact
TreeSHAP · an embedding or matrix-factorisation base learner to test the
information-gap hypothesis · artifact registry so serving survives a reload · the
FastAPI + PostgreSQL service layer from the original brief.

## Project layout

```
public/data/ml-100k/   dataset
scripts/               offline experiment CLI
src/ml/                dataset, features, models, stacking, experiments, recommender, worker
src/components/        AppShell, MlProvider, MoviePoster, StarRating, ui/*
src/pages/             Landing, Auth, Dashboard, Rate, Recommendations, Browse, Analytics
src/convex/            schema + recommender functions
tests/                 bun:test suites
docs/                  ARCHITECTURE.md, RESEARCH-REPORT.md, LEAKAGE-AUDIT.md
experiments/results/   JSON output of CLI runs
```

## Reproducing a number from this README

```bash
bun run experiment --scope full
# → experiments/results/<timestamp>_paper_full_random_seed42.json
```

Same seed, same dataset, same code ⇒ same numbers. If a number in this README
does not match your run, the JSON in `experiments/results/` is the ground truth
for what this repository actually measured.
