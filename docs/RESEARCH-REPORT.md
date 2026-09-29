# Research report — reproducing a stacking ensemble movie recommender

**Reproduction target.** Nisha Sharma and Dr. Mala Dutta, *“An Ensemble Movie
Recommender System Based on Stacking”*, Journal of Theoretical and Applied
Information Technology, Vol. 101, No. 18, 30 September 2023.

**Artifact.** `experiments/results/2026-09-29T14-50-53-115Z_paper_full_random_seed42.json`
(raw run output). Reproduce with `bun run experiment --scope full`.

**Integrity statement.** Every number in the “our results” tables was produced by
this repository on the MovieLens 100K files bundled in `public/data/ml-100k/`.
Published values appear only in tables explicitly headed as reference, come from
`src/ml/reference.ts`, and are never substituted into a computed column. No
citation is asserted that is not the paper named above; no claim about the paper's
content is made beyond what the brief states it publishes.

---

## 1. Abstract

We reconstruct the paper's stacking ensemble for movie rating prediction: six
heterogeneous base regressors evaluated standalone, then combined through a
meta learner over out-of-fold predictions. On MovieLens 100K with the paper's
stated feature set — user id, movie id and release year — our Linear Regression
(RMSE 1.0855 vs 1.08) and K-Nearest Neighbours (1.0708 vs 1.07) reproduce the
paper's standalone results to within 0.01 RMSE. The four ensemble learners do
not: Random Forest (1.0474 vs 0.98), Gradient Boosting (1.0537 vs 0.99), AdaBoost
(1.0772 vs 0.96) and XGBoost (1.0323 vs 0.92) all land above the published
values. The paper's proposed stack (KNN + XGBoost + Gradient Boosting → Linear
Regression) reaches RMSE 1.0321 here, against a published 0.90. We report the
gap, quantify the error structure that explains it, and identify the most likely
sources without asserting them as fact. Adding four supervised aggregate columns
in an explicitly labelled enhanced mode brings the same stack to RMSE 0.9450 —
within 0.045 of the published value — which is the strongest evidence available
here that the paper's ensembles had access to information the three id/year
columns do not carry.

## 2. Introduction

Rating prediction is a regression problem: given a user and a film, estimate the
score the user would give. The paper's contribution is architectural rather than
algorithmic — heterogeneous learners are stacked so that a meta learner can
exploit their complementary errors. This report asks a narrow question: with the
paper's stated dataset and feature set, and no access to its code, what does that
architecture actually achieve?

## 3. Problem statement

Given `D = {(user, movie, year, rating)}`, learn `f: (user, movie, year) → rating`,
evaluate with MAE/MSE/RMSE on a held-out split, then rank unseen films by
predicted rating to produce a Top-K list. Two constraints dominate the difficulty:
the feature set carries no content and no collaborative structure beyond two
categorical ids and a year, and the meta learner must be trained without leakage
from in-sample base predictions.

## 4. Objectives

1. Reproduce the paper's preprocessing, feature set and learner catalog.
2. Evaluate all six standalone learners independently.
3. Implement genuine, leakage-free two-level stacking with k-fold out-of-fold
   meta features.
4. Re-run all seven meta-learner configurations the paper reports.
5. Separate published values from measured ones, and explain divergences.
6. Ship a working recommender: Top-K lists with model-derived explanations,
   diversification and documented cold-start behaviour.

## 5. Literature context

Only the paper named above is referenced, because only that paper's methodology
is being reproduced. Where the brief attributes a claim to the paper — base
learner catalog, proposed stack, reported RMSE values, exclusion of demographic
information, the accuracy/diversity balance — we follow it and record the
attribution in code comments. We do not synthesise additional citations, and we
do not attribute cold-start handling, diversification, or feature engineering to
a paper that does not describe them.

## 6. Dataset

MovieLens 100K (October 1998 release), bundled at `public/data/ml-100k/`.

| Property | Value |
| --- | --- |
| Ratings | 100,000 |
| Users | 943 |
| Movies | 1,682 |
| Rating range | 1–5 (integers) |
| Global mean rating | 3.53 |
| Files used | `u.data`, `u.item`, `u.user`, `u.genre` |

Merging (step 2 of the pipeline) joins ratings to movie metadata and to user
demographics; the demographics are carried through the table and then excluded
from the model, matching the paper's statement that demographic information is
not used in the final modeling.

## 7. Data preprocessing

Missing-value audit on the merged table:

| Check | Result | Handling |
| --- | --- | --- |
| Duplicate (user, movie) pairs | 0 | none needed |
| Ratings outside 1–5 | 0 | none needed |
| Rows with no matching movie | 0 | none dropped |
| Movies with no usable release year | 9 rows | imputed with the **training-split median** |
| Movies with no genre flags | 0 | none needed |
| Ratings with no user record | 0 | none needed |

All imputation is fitted on training rows only; the test split never contributes
a statistic to preprocessing.

## 8. Feature selection

Paper mode uses exactly the columns the brief attributes to the paper:
`user_id`, `movie_id`, `year`. Categorical ids are label-encoded into dense
integers; unseen categories map to bucket 0. `rating` is the target and never an
input — this is asserted by a test. An enhanced mode adds movie popularity, movie
mean rating, user activity, user mean rating and 19 genre flags, with
leave-one-out encoding for training rows and a per-fold refit of the feature
space. Enhanced results are reported separately and are not mixed with the
reproduction.

## 9. Standalone models

Each learner is fitted on the 80,000-row training split and evaluated on the
20,000-row held-out split (seed 42, 20% test).

| Model | MAE | MSE | RMSE (ours) | OOF RMSE | Paper RMSE | Δ | Fit time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Linear Regression | 0.8928 | 1.1783 | **1.0855** | 1.0938 | 1.08 | +0.0055 | 0.1 s |
| K-Nearest Neighbours (k=40) | 0.8660 | 1.1465 | **1.0708** | 1.0798 | 1.07 | +0.0008 | 1.0 s |
| Random Forest (80 trees) | 0.8435 | 1.0970 | **1.0474** | 1.0570 | 0.98 | +0.0674 | 10.2 s |
| AdaBoost.R2 (50 stages) | 0.8884 | 1.1604 | **1.0772** | 1.0853 | 0.96 | +0.1172 | 2.3 s |
| Gradient Boosting (120 stages) | 0.8544 | 1.1102 | **1.0537** | 1.0629 | 0.99 | +0.0637 | 22.8 s |
| XGBoost (180 rounds, from scratch) | 0.8326 | 1.0657 | **1.0323** | 1.0424 | 0.92 | +0.1123 | 9.9 s |

XGBoost is the strongest standalone learner, as the paper also finds. Feature
importance from that model (split gain): `year` 58.3%, `movie_id` 27.9%,
`user_id` 13.8% — the release year carries most of the signal, because it is the
only ordinal feature in the set.

## 10. Stacking architecture

Base predictions are generated out-of-fold, combined into a meta feature matrix,
and passed to the meta learner. Each base learner is then refit on the full
training split to produce test-set predictions, which are fed to the trained meta
learner. Configuration E is the paper's proposal:

```
P_final = LR(P_KNN, P_XGBoost, P_GradientBoosting)
```

## 11. Meta learner

For configuration E the fitted Linear Regression weights (rating units) are:

| Meta input | Weight |
| --- | --- |
| `knn_prediction` | −0.0862 |
| `xgboost_prediction` | +1.0553 |
| `gradient_boosting_prediction` | +0.0131 |

The meta learner is almost entirely XGBoost, with Gradient Boosting contributing
almost nothing and KNN receiving a small negative weight — a legitimate outcome
of least squares on strongly correlated inputs (XGBoost ↔ Random Forest 0.911,
AdaBoost ↔ Gradient Boosting 0.916, XGBoost ↔ Gradient Boosting 0.885), not a bug.
This is why stacking buys little here.

## 12. Experimental setup

| Setting | Value |
| --- | --- |
| Dataset scope | full 100,000 interactions |
| Feature mode | paper (`user_id`, `movie_id`, `year`) |
| Split | random, 80/20 (16,000–20,000 rows in test) |
| Cross-validation | 5 folds, out-of-fold meta features |
| Random seed | 42 (split, folds, bootstrap, subsampling) |
| Hyperparameter search | `research` depth (the paper publishes none) |
| Stacking configurations | A–G (seven) |
| Base-learner fits | 36 (6 models × (5 folds + 1 full refit)) |
| Wall clock | 50.8 s (46.4 s in base learners) |

The paper does not publish hyperparameters, seeds, split ratios or fold counts.
Those choices are ours, are recorded per run, and are the first candidates when
explaining any divergence.

## 13. Results

| # | Configuration | MAE | MSE | RMSE (ours) | Paper RMSE | Δ | Rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | XGB + KNN → LR | 0.8323 | 1.0652 | 1.0321 | 0.91 | +0.1221 | 2 |
| B | XGB + KNN → XGB | 0.8319 | 1.0684 | 1.0336 | 0.92 | +0.1136 | 5 |
| C | KNN + GB + RF → XGB | 0.8396 | 1.0869 | 1.0425 | 0.94 | +0.1025 | 6 |
| D | KNN + XGB + GB → XGB | 0.8311 | 1.0681 | 1.0335 | 0.91 | +0.1235 | 4 |
| **E** | **KNN + XGB + GB → LR (proposed)** | **0.8323** | **1.0652** | **1.0321** | **0.90** | **+0.1321** | **3** |
| F | KNN + RF → XGB | 0.8421 | 1.0948 | 1.0463 | 0.95 | +0.0963 | 7 |
| G | KNN + XGB + GB + AB → LR | 0.8323 | 1.0652 | 1.0321 | 0.91 | +0.1221 | 1 |

**Lowest-RMSE configuration in this experiment:** G (KNN + XGB + GB + AdaBoost → LR)
at RMSE 1.0321, tied with A and E to four decimals because AdaBoost's extra column
receives a numerically negligible weight. This is an experimental ordering, not a
claim that one architecture is superior.

**Proposed model (configuration E):**

| Metric | Paper (reference) | Ours | Δ |
| --- | --- | --- | --- |
| MAE | 0.69 | 0.8323 | +0.1423 |
| MSE | 0.82 | 1.0652 | +0.2452 |
| RMSE | 0.90 | 1.0321 | +0.1321 |

Figures available in the analytics page (all generated from this run):
standalone RMSE vs reference, stacking RMSE vs reference, MAE, MSE, actual vs
predicted scatter, prediction-error histogram, meta-learner weights, base
learner correlation matrix, XGBoost feature importance.

## 14. Comparison with published results

| Group | Reproduces closely? | Observation |
| --- | --- | --- |
| Linear Regression | yes (+0.0055) | suggests the feature set and split style are aligned |
| KNN | yes (+0.0008) | same conclusion; k and scaling are ours, so the agreement is reassuring |
| Random Forest | no (+0.0674) | tuned ensembles overshoot by ~0.07 |
| Gradient Boosting | no (+0.0637) | same regime |
| AdaBoost | no (+0.1172) | largest standalone gap |
| XGBoost | no (+0.1123) | strongest model, still 0.11 short |
| Proposed stack | no (+0.1321) | inherits the XGBoost gap |

The pattern matters: the two models whose behaviour is least sensitive to
hyperparameters reproduce almost exactly, while the four tuned ensembles miss by
0.06–0.12. That is the signature of an uncontrolled hyperparameter or
implementation difference rather than a different dataset.

### 14.1 Enhanced mode — evidence about the cause

Enhanced mode keeps the same rows, the same split and the same learners, and
adds four supervised aggregates: movie popularity, movie mean rating, user
activity and user mean rating (target-derived, leave-one-out encoded on training
rows, feature space refit inside every fold). Genre flags are implemented but
off by default. Full-scope run, 91.7 s:

| Model | Enhanced RMSE (ours) | Paper RMSE | Δ |
| --- | --- | --- | --- |
| Linear Regression | 0.9536 | 1.08 | −0.1264 |
| K-Nearest Neighbours | 0.9679 | 1.07 | −0.1021 |
| Random Forest | 0.9519 | 0.98 | −0.0281 |
| AdaBoost.R2 | 0.9724 | 0.96 | +0.0124 |
| Gradient Boosting | 0.9472 | 0.99 | −0.0428 |
| XGBoost | 0.9578 | 0.92 | +0.0378 |

Stacking (same seven configurations):

| # | Configuration | RMSE | Paper RMSE |
| --- | --- | --- | --- |
| A | XGB + KNN → LR | 0.9493 | 0.91 |
| B | XGB + KNN → XGB | 0.9510 | 0.92 |
| C | KNN + GB + RF → XGB | 0.9453 | 0.94 |
| D | KNN + XGB + GB → XGB | 0.9473 | 0.91 |
| **E** | **KNN + XGB + GB → LR (proposed)** | **0.9450** | **0.90** |
| F | KNN + RF → XGB | 0.9478 | 0.95 |
| G | KNN + XGB + GB + AB → LR | 0.9449 | 0.91 |

Proposed stack in enhanced mode: MAE 0.7496, MSE 0.8930, RMSE 0.9450 against the
published 0.69 / 0.82 / 0.90.

Two things change beyond the headline metric:

1. **The meta learner starts working as an ensemble.** Weights move from
   (−0.086, +1.055, +0.013) to (+0.194, +0.264, +0.599) for KNN/XGBoost/Gradient
   Boosting. With informative features the three learners are no longer redundant,
   and the stack distributes weight across all of them.
2. **Error structure improves measurably.** Residual SD (sampled test rows) falls
   from 1.0106 to 0.9224, the share of predictions within ±1 star rises from
   68.4% to 75.4%, prediction SD rises from 0.423 to 0.585 against a target SD of
   1.090, and the actual/predicted correlation rises from 0.376 to 0.533.

This does not prove what the paper did — enhanced mode is our own construction and
depends on target-derived aggregates, so it is not a like-for-like reproduction.
It does show that the ensemble gap is an information gap, not an architectural
one: the same stacking code closes most of the distance once the inputs carry
more signal. It also demonstrates why the paper's statement that only user id,
movie id, year and rating were used sits uneasily with its reported ensemble
RMSE.

Runtime note: enhanced mode takes 91.7 s at full scope (Gradient Boosting 37.3 s,
XGBoost 25.2 s, Random Forest 17.8 s, KNN 3.8 s). Beyond six features the kd-tree
loses its pruning advantage, so the KNN neighbour pool is capped at 5,000 rows and
the cap is recorded in that model's parameters.

## 15. Error analysis

Measured on the test split for configuration E (1,200-point residual sample):

| Statistic | Value | Reading |
| --- | --- | --- |
| Residual mean (pred − actual) | −0.0106 | effectively unbiased |
| Residual standard deviation | 1.0106 | RMSE is dominated by spread, not bias |
| Predictions within ±0.5 | 37.3% | most predictions are off by more than half a star |
| Predictions within ±1.0 | 68.4% | roughly Gaussian residuals |
| Mean prediction | 3.55 | close to the global mean of 3.53 |
| SD of predictions | 0.423 | severe variance compression |
| SD of actual ratings | 1.090 | the target varies 2.6× more than the model's output |
| Correlation (actual, predicted) | 0.376 | the model explains little of the per-rating variance |

Variance compression is the core failure mode. With only user id, movie id and
year, the learner can identify coarse structure — which movie, which era — but
cannot separate users' heterogeneous reactions to the same film, so it predicts
near the mean. This is why the actual-vs-predicted plot collapses onto a
horizontal band around 3.5, and it bounds the achievable RMSE near 1.0 for this
feature set. Reaching the paper's 0.92 for XGBoost would require either
information the feature set does not contain or a different evaluation protocol.

Candidate explanations, in order of likelihood (the enhanced-mode run in §14.1
provides direct evidence for #3 and nothing that discriminates the others):

1. **Unpublished hyperparameters.** Tuning depth, learning rate and regularisation
   materially moves ensemble RMSE. We searched none of it; the paper reports no
   values.
2. **A train/test protocol difference.** If the split ratios or the fold scheme
   differ, RMSE is not comparable. Our KNN and Linear Regression agreement argues
   against a large protocol difference, but does not exclude one.
3. **Additional inputs.** The paper's feature list names four columns including
   `rating`; if additional columns (demographics, genres, or a user-movie
   interaction encoding) entered the model, the ensemble gap is explained while
   the id/year-only linear and neighbour models remain unaffected. In enhanced
   mode the same code reaches 0.9450 for the proposed stack and improves every
   learner, which is consistent with this explanation.
4. **A collaborative encoding not described in the brief.** An encoding that
   carries user-movie interaction structure (for example a target-encoded pair)
   would lift the ensembles well above what plain ids allow.
5. **XGBoost library versus reimplementation.** Our XGBoost is written from
   scratch; the library's exact split search and defaults are its own. This is
   our deviation, and it is documented rather than papered over.

None of these is asserted as the cause. They are the hypotheses a follow-up
study with paper access should test first.

## 16. Limitations

- Hyperparameters, seeds, split ratios and fold counts are unpublished, so this is
  a methodology reproduction, not a numeric one.
- XGBoost is reimplemented from scratch; Random Forest, Gradient Boosting and
  AdaBoost use quantile-binned split search instead of exact splits.
- The paper's reported metrics are per-model RMSE only; MAE and MSE comparisons
  are therefore one-sided (ours measured, theirs unavailable).
- Cold start, diversification and content features are our additions. The paper
  does not describe them, and none is attributed to it.
- Rating prediction is not ranking evaluation: RMSE does not measure whether the
  Top-K list is good. No NDCG/precision@k baseline exists in the paper to compare
  against, so the recommender is evaluated qualitatively (diversity, coverage,
  support) rather than against a published ranking metric.
- The web deployment persists model metadata, not artifacts; a reload requires a
  retrain (about 51 s on the full dataset).

## 17. Future work

1. Hyperparameter search (`quick`/`research`/`production` modes already exist;
   only the middle mode is used for the reported run).
2. Ranking metrics — precision@k, recall@k, NDCG@k, coverage — with a temporal
   protocol, so the recommender is evaluated on what it actually does.
3. Exact TreeSHAP on the tree base learners, replacing gain importance for local
   explanations.
4. Matrix-factorisation or learned embeddings as an additional base learner, to
   test whether the ensemble gap is an information gap.
5. Artifact registry: version and store fitted models so a reload can serve
   without retraining.
6. A FastAPI + PostgreSQL service layer mirroring the REST surface in
   `docs/ARCHITECTURE.md`, if the project moves beyond single-user reproduction.

## 18. Conclusion

The architecture reproduces; the headline numbers do not. Feature extraction,
merging, encoding, standalone evaluation and leakage-safe stacking all rebuild
faithfully, and two of the six standalone learners land within 0.01 RMSE of the
published values on the paper's stated feature set. The four tuned ensembles and
the stacked model land 0.06–0.13 RMSE higher, with error analysis pointing at
variance compression under an id-only feature set as the mechanism. The gap is
reported as a gap, the hypotheses behind it are labelled as hypotheses, and every
number in this report can be regenerated with one command:

```bash
bun run experiment --scope full
```
