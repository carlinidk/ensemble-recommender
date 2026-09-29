# Architecture — Ensemble (stacking movie recommender)

Reproduction target: Nisha Sharma & Dr. Mala Dutta, *“An Ensemble Movie Recommender
System Based on Stacking”*, Journal of Theoretical and Applied Information
Technology, Vol. 101, No. 18, 30 September 2023. Dataset: MovieLens 100K.

This document is the plan the implementation follows. Where version 1 deliberately
stops short, it says so.

---

## 1. System architecture

```
                     ┌──────────────────────────── browser (Vite + React 19) ────────────────────────────┐
                     │                                                                                  │
  MovieLens 100K ──► │  /public/data/ml-100k/*.data|item|user|genre                                      │
  (bundled files)    │            │                                                                     │
                     │            ▼                                                                     │
                     │  ┌──────────────────┐        postMessage         ┌───────────────────────────┐    │
                     │  │ React UI         │ ◄───────────────────────► │ ML web worker             │    │
                     │  │ Landing / Rate   │   typed protocol         │ • dataset parse + merge   │    │
                     │  │ Browse / Top-K   │                          │ • feature encoding        │    │
                     │  │ Analytics        │                          │ • offline training run    │    │
                     │  └──────────────────┘                          │ • resident models         │    │
                     │            │                                  │ • candidate scoring       │    │
                     └────────────┼──────────────────────────────────┴───────────────────────────┘    │
                                  │ Convex client (reactive queries + mutations)
                                  ▼
                     ┌───────────────────────────── Convex backend ──────────────────────────────┐
                     │ users (auth) · ratings · profiles · modelRuns · experimentResults         │
                     │ Model artifacts stay in memory / on disk; only metadata + metrics persist │
                     └──────────────────────────────────────────────────────────────────────────┘
```

Two processes, as the brief requires:

* **Training pipeline (offline)** — `runExperiment()` in the worker or via
  `bun run experiment`. Fits everything, evaluates, emits a result object and a
  JSON artifact.
* **Inference pipeline (serving)** — the worker's `recommend` / `score-movies`
  handlers. They only call `predict()` on already-fitted base learners and the
  meta learner. Nothing is refitted per request.

## 2. Folder structure

```
├── public/data/ml-100k/         MovieLens 100K files (u.data, u.item, u.user, u.genre, README)
├── scripts/run-experiment.ts    headless training/evaluation CLI
├── tests/                       bun:test suites (core, stacking, recommender)
├── docs/                        this file, research report, leakage audit
├── experiments/results/         JSON output of CLI runs
└── src/
    ├── ml/
    │   ├── types.ts             dataset/feature/config/regressor contracts
    │   ├── rng.ts               seeded PRNG, shuffling, k-folds, splits
    │   ├── metrics.ts           MAE / MSE / RMSE, residuals, histograms
    │   ├── dataset.ts           parsers, merge, missing-value audit
    │   ├── features.ts          encoder fitting (train-only) + matrix encoding
    │   ├── matrix.ts            matrix slicing/assembly helpers
    │   ├── split.ts             random and temporal splitting
    │   ├── reference.ts         published values (read-only, never our results)
    │   ├── models/              linearRegression, knn, tree, randomForest,
    │   │                        adaboost, gradientBoosting, xgboost, registry
    │   ├── stacking.ts          k-fold out-of-fold stacking + meta learner
    │   ├── experiments.ts       standalone + 7 stacking configs + plots payload
    │   ├── recommender/         profile (persona), diversify (MMR), engine
    │   ├── worker.ts            training + inference worker
    │   ├── workerProtocol.ts    typed message contract
    │   └── client.ts            worker client (FIFO request queue)
    ├── components/              AppShell, MlProvider, MoviePoster, StarRating, ui/*
    ├── pages/                   Landing, Auth, Dashboard, Rate, Recommendations, Browse, Analytics
    ├── convex/                  schema + queries/mutations (+ auth, untouched)
    └── lib/                     catalogue helpers
```

## 3. Technology stack

| Layer | Choice | Why |
| --- | --- | --- |
| UI | React 19, React Router 7, Tailwind v4, shadcn/ui, Framer Motion, Recharts | project template |
| Numerics | TypeScript, typed arrays, custom implementations | the brief asks for the stacking pipeline to be inspectable and reproducible; no ML framework hides the algorithm |
| Training/runtime | Web Worker | keeps a ~50 s full-dataset fit off the UI thread |
| Persistence | Convex (auth + document DB) | project backend; stores users, ratings, preferences, run metadata |
| Dataset | MovieLens 100K, bundled under `public/` | the paper's dataset, pinned and versioned so runs are reproducible offline |

**Not the official XGBoost library.** The XGBoost base learner is a from-scratch
implementation of the algorithm (second-order gradients, regularized leaf
weights, weighted quantile binning, subsampling, shrinkage). This is a deliberate
deviation from the paper's likely library use and is stated wherever the numbers
are shown.

## 4. Database schema (Convex)

| Table | Fields | Purpose |
| --- | --- | --- |
| `users` | from `@convex-dev/auth` | authentication |
| `ratings` | `userId`, `movieId`, `rating`, `ratedAt` — indexes `by_user`, `by_user_movie` | the account's taste data; excluded from candidates |
| `profiles` | `userId`, `topK`, `mode`, `diversityLambda`, `affinityWeight`, `updatedAt` | serving preferences (Top-K, accuracy vs diversity, enhancement weights) |
| `modelRuns` | `userId`, `modelId`, `version`, `algorithm`, `datasetVersion`, `featureVersion`, `randomSeed`, `cvFolds`, `datasetScope`, `splitStrategy`, `searchMode`, `hyperparameters`, `metrics`, `standalone`, `stacking`, `trainingSeconds`, `nTrain`, `nTest` | model registry metadata |
| `experimentResults` | `runId`, `userId`, `configId`, `label`, `baseLearners`, `metaLearner`, `mae`, `mse`, `rmse`, `rank` | one row per stacking configuration |

Artifacts (fitted trees, KD-tree, meta weights) are held in the worker and
serialised to `experiments/results/*.json` metadata for the CLI — never stored as
document blobs.

> Deviation from the brief: the brief targets PostgreSQL + SQLAlchemy for the
> service layer. Version 1 runs entirely on the project's Convex backend; the
> table design above mirrors the requested schema so a Postgres port is a
> mapping exercise, not a redesign.

## 5. ML pipeline

1. **Load** `u.data`, `u.item`, `u.user`, `u.genre`.
2. **Merge** movies ⨝ ratings (+ users) into one table.
3. **Audit missing values** — 9 rows lack a release year; users/movies have no
   gaps. Nothing is silently imputed.
4. **Handle missing values** — year imputed with the *training-split median*;
   demographics are merged but excluded from modelling, exactly as the paper
   states.
5. **Feature extraction** — `user_id`, `movie_id`, `year` (paper mode).
6. **Encoding** — high-cardinality ids are label-encoded to dense integers;
   unseen categories map to bucket 0. Encoders are fitted on train rows only.
7. **Split** — random (paper) or temporal (enhanced), fixed seed.
8. **Train** each base learner independently, evaluate on the held-out split.
9. **Generate out-of-fold predictions** per base learner (k-fold).
10. **Train the meta learner** on the out-of-fold matrix.
11. **Refit base learners on the whole training split** and predict the test set.
12. **Predict** through the meta learner → final ratings.
13. **Evaluate** MAE/MSE/RMSE for every model and configuration.
14. **Recommend** — candidates, ranking, optional MMR diversification, Top-K.

## 6. Stacking architecture

```
Level 1 (base)                      Level 2 (meta)
┌──────────────┐  oof  ┌──────────────────────────────┐
│ KNN          │──────►│                              │
├──────────────┤       │  Linear Regression           │──► final rating
│ XGBoost      │──────►│  (or XGBoost for configs     │
├──────────────┤       │   B, C, D, F)                │
│ GradientBoost│──────►│                              │
├──────────────┤       │  inputs: [P_knn, P_xgb, P_gb]│
│ (+ RF, AB)   │──────►│                              │
└──────────────┘       └──────────────────────────────┘
        ▲                            ▲
        │ fit per fold               │ trained on out-of-fold predictions only
        └────────── K folds ─────────┘
```

`P_final = LR(P_KNN, P_XGB, P_GB)` — with the meta learner never seeing an
in-sample base prediction. Seven configurations are evaluated (A–G), matching the
combinations the paper reports.

## 7. API design

Version 1 exposes the backend as typed Convex functions rather than REST:

| Convex function | Kind | Equivalent REST endpoint in the brief |
| --- | --- | --- |
| `recommender.myRatings` | query | `GET /users/{id}/ratings` |
| `recommender.myProfile` | query | `GET /users/{id}` (serving preferences) |
| `recommender.latestRun` | query | `GET /models`, `GET /metrics` |
| `recommender.runExperiments` | query | `GET /experiments` |
| `recommender.rateMovie` / `clearMovieRating` / `clearRatings` | mutation | `POST /ratings` |
| `recommender.saveProfile` | mutation | `POST /recommendations` (settings) |
| `recommender.recordRun` | mutation | `POST /experiments` |

Recommendation *computation* runs in the worker behind a typed message protocol
(`workerProtocol.ts`) — the browser equivalent of `GET /recommendations/{user_id}`
with a `topK` and `mode` parameter. Deferred to v2: a FastAPI service mirroring
the listed REST surface, API keys, rate limiting and an OpenAPI document.

## 8. Frontend architecture

| Route | Auth | Purpose |
| --- | --- | --- |
| `/` | public | landing: architecture, published reference results, integrity rules |
| `/auth` | public | email-OTP / guest sign-in, `returnTo` aware |
| `/dashboard` | required | account overview, persona status, Top-3 preview |
| `/rate` | required | rate starter set + search, live persona and genre signals |
| `/recommendations` | required | Top-K with K selector, accuracy/diversity, explanations, diagnostics |
| `/browse` | required | catalogue explorer with filters and predicted ratings |
| `/analytics` | required | run configuration, reproduction vs reference tables, charts, run notes |

Shared pieces: `MlProvider` (one worker for the app, above the router),
`AppShell` (sidebar + mobile drawer), `MoviePoster` (deterministic generated
artwork — the dataset ships no images), `StarRating`.

## 9. Experiment plan

| # | Configuration | Meta learner |
| --- | --- | --- |
| A | KNN + XGBoost | Linear Regression |
| B | KNN + XGBoost | XGBoost |
| C | KNN + Gradient Boosting + Random Forest | XGBoost |
| D | KNN + XGBoost + Gradient Boosting | XGBoost |
| E | KNN + XGBoost + Gradient Boosting | Linear Regression ← paper's proposal |
| F | KNN + Random Forest | XGBoost |
| G | KNN + XGBoost + Gradient Boosting + AdaBoost | Linear Regression |

Each configuration reports MAE, MSE and RMSE; the lowest-RMSE configuration is
reported as an experimental artefact, never as a claim of superiority.

## 10. Implementation phases

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | requirements → architecture (this document) | done |
| 2 | dataset pipeline: parsers, merge, audit | done |
| 3 | preprocessing + feature engineering + encoding | done |
| 4 | six standalone base learners | done |
| 5 | standalone evaluation and comparison table | done |
| 6 | leakage-safe k-fold stacking | done |
| 7 | all seven stacking experiments + charts | done |
| 8 | recommendation engine: candidates, ranking, MMR, cold start, explanations | done |
| 9 | backend: Convex schema, ratings, preferences, run registry | done |
| 10 | frontend: landing, rating, Top-K, browse, analytics | done |
| 11 | explainability: base-prediction decomposition, meta weights, gain importance | partial — exact for the linear meta learner and tree gain; full TreeSHAP is v2 |
| 12 | tests | done — 32 tests (`bun test tests/`) |
| 13 | deployment: container/orchestration files | deferred to v2 |
| 14 | research report | done — `docs/RESEARCH-REPORT.md` |

## 11. Risks and leakage concerns

Full treatment in `docs/LEAKAGE-AUDIT.md`. Summary:

| Risk | Mitigation |
| --- | --- |
| Meta learner trained on in-sample base predictions | k-fold out-of-fold matrix; asserted by a test |
| Test statistics influencing preprocessing | encoders and all supervised statistics fitted on train rows only |
| Supervised features leaking the row's own target | leave-one-out encoding for training rows; feature space refit *inside* each fold in enhanced mode |
| Target used as an input feature | target is never in the feature matrix; asserted by a test on paper-mode names |
| Recommendation candidates the user already rated | excluded, plus the persona's own history |
| Model scores presented as calibrated confidence | spread across base learners is labelled a disagreement measure |
| Borrowed published numbers | published values live in `reference.ts` and render under a separate heading only |
| Non-reproducible runs | fixed seeds for split, folds, bootstrap, subsampling; config recorded per run |
| Browser thread starvation | training runs in a worker with progress messages |
| Claiming the paper's cold-start design | none exists in the paper; every fallback is labelled as ours |

## 12. Exact commands

```bash
# install (Bun is the package manager for this project)
bun install

# typecheck the frontend + worker + Convex functions
bun run typecheck                 # tsc -b --noEmit
bunx convex dev --once            # regenerate Convex types after schema edits

# tests
bun test tests/

# offline training / evaluation (writes experiments/results/*.json)
bun run experiment                                  # paper mode, full 100k, 5-fold, seed 42
bun run experiment --scope sample --sample-size 20000
bun run experiment --mode enhanced
bun run experiment --split temporal
bun run experiment --folds 10 --seed 7

# local development (managed by the platform in Freebuff; not run by the agent)
bun run dev
```
