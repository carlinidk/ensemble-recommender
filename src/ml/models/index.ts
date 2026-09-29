import type { BaseModelName, MetaModelName, Regressor, SearchMode } from "../types";
import { AdaBoostRegressor } from "./adaboost";
import { GradientBoostingRegressor } from "./gradientBoosting";
import { KnnRegressor } from "./knn";
import { LinearRegression } from "./linearRegression";
import { RandomForestRegressor } from "./randomForest";
import { XGBoostRegressor } from "./xgboost";

export type { SearchMode };

export interface ModelOptions {
  search: SearchMode;
  seed: number;
}

export const BASE_MODEL_LABELS: Record<BaseModelName, string> = {
  linear_regression: "Linear Regression",
  knn: "K-Nearest Neighbours",
  random_forest: "Random Forest",
  adaboost: "AdaBoost.R2",
  gradient_boosting: "Gradient Boosting",
  xgboost: "XGBoost",
};

export const BASE_MODEL_NOTES: Record<BaseModelName, string> = {
  linear_regression:
    "Ordinary least squares on the encoded features. The paper reports it as both a standalone baseline and its meta learner.",
  knn:
    "Distance-weighted-free kd-tree neighbour regression over standardized user id / movie id / year. k, weighting and scaling are not published by the paper.",
  random_forest:
    "Bagged CART regression trees with bootstrap sampling, averaged over trees.",
  adaboost:
    "AdaBoost.R2 with weighted regression trees as base estimators; predictions are combined with a weighted median.",
  gradient_boosting:
    "Squared-error gradient boosting: each stage fits a shallow tree to the current residuals.",
  xgboost:
    "From-scratch implementation of the XGBoost algorithm (second-order gradients, regularized leaf weights, quantile binning) — not the official xgboost library.",
};

/**
 * Hyperparameter sets. The paper names the algorithms but does not publish
 * hyperparameters, so these are our own reasonable choices at three search
 * depths. `quick` keeps the browser demo responsive; `research` is the default
 * for the reproduction run; `production` is the widest search.
 */
export function parameterTable(search: SearchMode): Record<string, Record<string, number | string>> {
  switch (search) {
    case "quick":
      return {
        knn: { k: 25, weights: "uniform" },
        random_forest: { n_estimators: 40, max_depth: 12, min_samples_leaf: 8 },
        adaboost: { n_estimators: 30, base_max_depth: 3 },
        gradient_boosting: { n_estimators: 80, learning_rate: 0.15, max_depth: 3 },
        xgboost: { n_estimators: 120, learning_rate: 0.15, max_depth: 5, subsample: 0.8 },
        linear_regression: { solver: "normal equations" },
      };
    case "production":
      return {
        knn: { k: 50, weights: "uniform" },
        random_forest: { n_estimators: 150, max_depth: 16, min_samples_leaf: 3 },
        adaboost: { n_estimators: 80, base_max_depth: 4 },
        gradient_boosting: { n_estimators: 300, learning_rate: 0.05, max_depth: 4 },
        xgboost: { n_estimators: 400, learning_rate: 0.05, max_depth: 7, subsample: 0.8 },
        linear_regression: { solver: "normal equations" },
      };
    case "research":
    default:
      return {
        knn: { k: 40, weights: "uniform" },
        random_forest: { n_estimators: 80, max_depth: 14, min_samples_leaf: 5 },
        adaboost: { n_estimators: 50, base_max_depth: 4 },
        gradient_boosting: { n_estimators: 120, learning_rate: 0.1, max_depth: 3, subsample: 0.7 },
        xgboost: { n_estimators: 180, learning_rate: 0.1, max_depth: 6, subsample: 0.8 },
        linear_regression: { solver: "normal equations" },
      };
  }
}

export function createBaseModel(name: BaseModelName, options: ModelOptions): Regressor {
  const { seed, search } = options;
  switch (name) {
    case "linear_regression":
      return new LinearRegression();
    case "knn":
      return new KnnRegressor(search === "quick" ? { k: 25 } : search === "production" ? { k: 50 } : { k: 40 });
    case "random_forest":
      return new RandomForestRegressor(
        search === "quick"
          ? { nEstimators: 40, maxDepth: 12, minSamplesLeaf: 8, minSamplesSplit: 16, seed }
          : search === "production"
            ? { nEstimators: 150, maxDepth: 16, minSamplesLeaf: 3, minSamplesSplit: 6, seed }
            : { nEstimators: 80, maxDepth: 14, minSamplesLeaf: 5, minSamplesSplit: 10, seed },
      );
    case "adaboost":
      return new AdaBoostRegressor(
        search === "quick"
          ? { nEstimators: 30, maxDepth: 3, seed }
          : search === "production"
            ? { nEstimators: 80, maxDepth: 4, seed }
            : { nEstimators: 50, maxDepth: 4, seed },
      );
    case "gradient_boosting":
      return new GradientBoostingRegressor(
        search === "quick"
          ? { nEstimators: 80, learningRate: 0.15, maxDepth: 3, minSamplesLeaf: 3, subsample: 0.7, seed }
          : search === "production"
            ? { nEstimators: 300, learningRate: 0.05, maxDepth: 4, minSamplesLeaf: 3, subsample: 0.7, seed }
            : { nEstimators: 120, learningRate: 0.1, maxDepth: 3, minSamplesLeaf: 3, subsample: 0.7, seed },
      );
    case "xgboost":
      return new XGBoostRegressor(
        search === "quick"
          ? { nEstimators: 120, learningRate: 0.15, maxDepth: 5, subsample: 0.8, seed }
          : search === "production"
            ? { nEstimators: 400, learningRate: 0.05, maxDepth: 7, subsample: 0.8, seed }
            : { nEstimators: 180, learningRate: 0.1, maxDepth: 6, subsample: 0.8, seed },
      );
    default: {
      const exhaustive: never = name;
      throw new Error(`Unknown base learner: ${String(exhaustive)}`);
    }
  }
}

export function createMetaModel(name: MetaModelName, options: ModelOptions): Regressor {
  if (name === "linear_regression") return new LinearRegression();
  return new XGBoostRegressor({
    nEstimators: 150,
    learningRate: 0.1,
    maxDepth: 4,
    subsample: 1,
    seed: options.seed,
  });
}
