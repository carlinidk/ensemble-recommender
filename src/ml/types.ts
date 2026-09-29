/**
 * Shared types for the ensemble movie recommender.
 *
 * PAPER SOURCE: Nisha Sharma & Dr. Mala Dutta, "An Ensemble Movie Recommender
 * System Based on Stacking", JATIT Vol. 101, No. 18, 30 September 2023.
 *
 * Every module that follows the paper's methodology is marked `PAPER MODE`.
 * Anything beyond it is marked `ENHANCED MODE` and documented as a deviation.
 */

/** A raw MovieLens 100K movie record (u.item). */
export interface Movie {
  id: number;
  title: string;
  /** Release year extracted from title/date, null when both are absent. */
  year: number | null;
  genres: string[];
  imdbUrl: string | null;
}

/** A raw MovieLens 100K rating record (u.data). */
export interface Rating {
  userId: number;
  movieId: number;
  rating: number;
  timestamp: number;
}

/** A raw MovieLens 100K user record (u.user). Available, but deliberately unused in modeling. */
export interface UserRecord {
  id: number;
  age: number;
  gender: string;
  occupation: string;
  zip: string;
}

/** Result of merging u.data + u.item (+ u.user), i.e. the paper's single merged dataset. */
export interface MergedRow {
  userId: number;
  movieId: number;
  rating: number;
  timestamp: number;
  /** Movie release year; 0 when the source files have no usable year. */
  year: number;
  /** True when `year` is absent — the feature layer imputes it from train rows only. */
  yearMissing: boolean;
  genreCount: number;
}

/** Missing / malformed value audit over the merged dataset. */
export interface DataAudit {
  rows: number;
  users: number;
  movies: number;
  duplicateUserMoviePairs: number;
  ratingsOutOfRange: number;
  missingMovieYear: number;
  missingMovieGenre: number;
  missingUserDemographics: number;
  /** Rows dropped for any reason during cleaning. */
  droppedRows: number;
  yearFillValue: number;
}

/**
 * A dense row-major design matrix: `data[i * d + j]` is feature `j` of row `i`.
 * Kept as typed arrays because models are trained on the full 100K dataset.
 */
export interface Matrix {
  n: number;
  d: number;
  names: string[];
  data: Float64Array;
}

/** Model-agnostic scoring report. */
export interface Metrics {
  mae: number;
  mse: number;
  rmse: number;
  n: number;
}

/** Any regressor usable as a base learner or meta learner. */
export interface Regressor {
  readonly name: string;
  fit(X: Matrix, y: Float64Array): void;
  predict(X: Matrix): Float64Array;
  /** Split-gain importance for tree ensembles, |coefficient| for linear models. */
  importance?(): FeatureImportance[];
  /** Human-readable trained hyperparameters, persisted with the experiment record. */
  describe?(): Record<string, number | string | boolean>;
}

export interface FeatureImportance {
  name: string;
  value: number;
}

/** Names must match the registry in `src/ml/models/index.ts`. */
export type BaseModelName =
  | "linear_regression"
  | "knn"
  | "random_forest"
  | "adaboost"
  | "gradient_boosting"
  | "xgboost";

export type MetaModelName = "linear_regression" | "xgboost";

export const BASE_MODEL_NAMES: BaseModelName[] = [
  "linear_regression",
  "knn",
  "random_forest",
  "adaboost",
  "gradient_boosting",
  "xgboost",
];

/** How much of MovieLens 100K a training run consumes. */
export type DatasetScope = "sample" | "full";

/** Random split (paper mode) or temporal split (enhanced evaluation). */
export type SplitStrategy = "random" | "temporal";

/** Hyperparameter search depth. */
export type SearchMode = "quick" | "research" | "production";

export type TrainingMode = "paper" | "enhanced";

export interface TrainingConfig {
  /** "paper" = the paper's exact feature set; "enhanced" = added features (documented deviation). */
  mode: TrainingMode;
  /**
   * Enhanced-mode feature breadth. "core" adds the four supervised aggregates;
   * "with_genres" also appends the 19 genre flags, which multiplies tree-fitting
   * cost on the full dataset and is therefore opt-in.
   */
  enhancedFeatures?: "core" | "with_genres";
  scope: DatasetScope;
  /** Number of rows used when scope === "sample". */
  sampleSize: number;
  randomSeed: number;
  testSize: number;
  cvFolds: number;
  /**
   * Evaluation split. "random" follows the paper's random train/test split;
   * "temporal" is an ENHANCED-MODE option that trains on older interactions and
   * tests on later ones.
   */
  split: SplitStrategy;
  /** Hyperparameter search depth; derived from `scope` when omitted. */
  search?: SearchMode;
}
