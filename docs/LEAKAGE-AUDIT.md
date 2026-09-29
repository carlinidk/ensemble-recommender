# Data-leakage audit

Scope: the full path from raw MovieLens files to a ranked Top-K list.
Every entry names the leak, where it would have appeared, what the code does
instead, and how it is verified.

Verified by `bun test tests/` (32 tests, 1115 assertions) — the checks called
out below are specifically `tests/stacking.test.ts` and `tests/core.test.ts`.

| # | Potential leak | Where it could happen | What the code does | Verification |
| --- | --- | --- | --- | --- |
| 1 | Meta learner trained on in-sample base predictions | `stacking.ts` | Base learners are fitted on K−1 folds and predict the held-out fold; the meta learner consumes only that out-of-fold matrix. Each base learner is refit on the full training split *after* the meta matrix exists. | `leakage-safe stacking > out-of-fold predictions are not in-sample predictions` fits a 1-NN model on pure-noise targets: an in-sample fit would memorise them, the out-of-fold vector cannot, and the test asserts the correlation collapses. |
| 2 | Test rows influencing feature encoders | `features.ts` | `fitFeatureSpace` iterates `trainIdx` only. Label codes, the year median, and every supervised statistic come from training rows. | `features > year imputation uses the training median when the year is missing`. |
| 3 | Test rows influencing supervised feature statistics (enhanced mode) | `features.ts`, `stacking.ts` | Movie/user means and counts are fitted on the training split, and *inside each fold* a fresh feature space is fitted on that fold's training rows, so a held-out row never contributes to its own movie/user average. | `features > enhanced mode adds supervised statistics without leaking the row's own target`. |
| 4 | A training row contributing to its own supervised feature | `features.ts` | `leaveOneOut: true` removes the row's own rating from the movie/user average before encoding. | Same test: leave-one-out value is asserted to differ from the plain group mean when the row is not exactly average. |
| 5 | Target rating used as an input feature | `features.ts` | The paper's stated feature list is `user_id`, `movie_id`, `year`; `rating` is the label. Paper mode exposes exactly three columns. | `features > paper mode exposes exactly the paper's three inputs plus nothing else`. |
| 6 | Future ratings influencing past predictions (temporal mode) | `split.ts` | The temporal split sorts by timestamp and cuts once, so every training interaction is older than every test interaction. | `splitting > temporal split puts every training interaction before every test interaction`. |
| 7 | Non-reproducible folds quietly changing results | `rng.ts`, `stacking.ts` | Folds come from a dedicated seeded stream (`seed + 101`) so adding a model to the run cannot reshuffle other models' folds. | `splitting > random split is reproducible for a fixed seed and covers every row`, `k-fold assignment is disjoint and complete`. |
| 8 | Recommending a movie the user already rated | `recommender/engine.ts` | Candidates are built from the catalogue minus the user's rated ids. | `recommendation engine > never recommends a movie the user or the persona already rated`. |
| 9 | Scoring movies the *persona* already rated (memorised interactions) | `recommender/engine.ts` | The matched MovieLens user's own history is excluded from candidates as well. | Same test asserts the exclusion count and that no returned id is in the persona's history. |
| 10 | Fallback heuristic presented as a model prediction | `recommender/engine.ts` | Cold start and no-persona cases return `predictedRating: null`, a `cold_start_fallback` / `no_persona_fallback` source, and an explicit caveat string. | `cold start > a user with no ratings gets an explicitly labelled popularity fallback`, `ratings without a persona match fall back and say why`. |
| 11 | Invented recommendation explanations | `recommender/engine.ts` | Explanations are assembled from values that exist: base-learner outputs, persona cosine and overlap, the account's own genre statistics, and MovieLens support counts. When the pipeline is a fallback, no model sentence is emitted at all. | `recommendation engine > labels model-derived recommendations with the base predictions behind them`. |
| 12 | Uncalibrated uncertainty presented as confidence | `recommender/engine.ts` | The reported figure is the standard deviation across the three base learners, labelled "a disagreement measure, not a calibrated probability". | Types expose `confidence.note`; the UI renders that note next to the value. |
| 13 | Borrowed published metrics reported as ours | `reference.ts` | Published values are read-only constants rendered only under the "Published Paper Reference Results" heading; our tables are built from computed predictions. | `reference integrity > published values are only ever reported as reference`; `unit tests never assert a published value as our result`. |
| 14 | Missing published values silently fabricated | `reference.ts` | The paper publishes RMSE for the standalone models and MAE/MSE/RMSE only for the proposed model. Missing cells are `null` and render as `—`. | `reference integrity > published values are only ever reported as reference` asserts `mae === null` for XGBoost. |
| 15 | Describing the paper's feature set incorrectly | `features.ts`, `reference.ts` | Paper mode's three inputs match `PAPER_FEATURES`; enhancements exist only in enhanced mode and are labelled. | `reference integrity` tests compare registry names against the paper constants. |
| 16 | Population statistics (cold-start fallback) mistaken for the model | `recommender/engine.ts` | Cold-start ranking uses full-dataset popularity and Bayesian mean rating; the fallback source label and caveat travel with every item. | `cold start` tests. |
| 17 | Sparse-support movies reported as well-supported | `recommender/engine.ts` | Movies with fewer than 10 dataset ratings carry `support.sparse` plus a caveat sentence naming the count. | Rendered on each card in the Top-K list. |

## Deliberate, labelled deviations (not leaks, but worth stating)

1. **Persona mapping.** The paper's models consume a MovieLens `user_id`; a new
   app user has no such row. The serving path maps the account onto the closest
   MovieLens user by cosine similarity over co-rated films. This bridge is our
   addition and is named in every explanation that uses it.
2. **Content-affinity blend.** A configurable weight blends the account's own
   genre history into the ranking. Default 0.30; setting it to 0.00 gives the
   pure paper model. The paper has no such layer.
3. **Population fallbacks.** Cold start and no-persona cases use popularity and
   Bayesian mean rating. The paper does not address cold start at all; no
   cold-start mechanism is attributed to it.
4. **Histogram binning.** Tree learners use quantile-binned features (64 bins by
   default) for tractability in the browser. XGBoost does this natively; for
   Random Forest, Gradient Boosting and AdaBoost it is an approximation of exact
   split search and is recorded in the run metadata.
5. **Standardized features for KNN and Linear Regression.** Raw ids span 1–1682;
   scaling is fitted on train rows only and is reported in the model parameters.
