/**
 * PUBLISHED REFERENCE VALUES — read-only constants.
 *
 * These numbers are transcribed from the paper and are displayed ONLY under the
 * heading "Published Paper Reference Results". They are never used as our own
 * results, never merged into our metric tables, and never plotted as if we had
 * measured them.
 *
 * Source: Nisha Sharma and Dr. Mala Dutta, "An Ensemble Movie Recommender System
 * Based on Stacking", Journal of Theoretical and Applied Information Technology,
 * Vol. 101, No. 18, 30 September 2023.
 *
 * Note: the paper publishes RMSE for the standalone models and for the stacking
 * configurations; it publishes MAE, MSE and RMSE only for the proposed model.
 * Missing cells stay `null` here rather than being invented.
 */

export const PAPER = {
  title: "An Ensemble Movie Recommender System Based on Stacking",
  authors: "Nisha Sharma, Dr. Mala Dutta",
  venue: "Journal of Theoretical and Applied Information Technology",
  volume: "Vol. 101, No. 18",
  date: "30 September 2023",
  dataset: "MovieLens 100K (100,000 ratings, 943 users, 1,682 movies)",
} as const;

export interface ReferenceMetric {
  rmse: number;
  mae: number | null;
  mse: number | null;
}

/** Table 4-style standalone results reported by the paper (RMSE only). */
export const PAPER_STANDALONE: Record<string, ReferenceMetric> = {
  linear_regression: { rmse: 1.08, mae: null, mse: null },
  xgboost: { rmse: 0.92, mae: null, mse: null },
  random_forest: { rmse: 0.98, mae: null, mse: null },
  adaboost: { rmse: 0.96, mae: null, mse: null },
  gradient_boosting: { rmse: 0.99, mae: null, mse: null },
  knn: { rmse: 1.07, mae: null, mse: null },
};

export interface ReferenceStack {
  id: string;
  baseLearners: string[];
  metaLearner: string;
  label: string;
  paperRmse: number;
}

/** Stacking configurations and their RMSE as reported by the paper. */
export const PAPER_STACKING: ReferenceStack[] = [
  { id: "A", baseLearners: ["knn", "xgboost"], metaLearner: "linear_regression", label: "XGB + KNN → LR", paperRmse: 0.91 },
  { id: "B", baseLearners: ["knn", "xgboost"], metaLearner: "xgboost", label: "XGB + KNN → XGB", paperRmse: 0.92 },
  { id: "C", baseLearners: ["knn", "gradient_boosting", "random_forest"], metaLearner: "xgboost", label: "KNN + GB + RF → XGB", paperRmse: 0.94 },
  { id: "D", baseLearners: ["knn", "xgboost", "gradient_boosting"], metaLearner: "xgboost", label: "KNN + XGB + GB → XGB", paperRmse: 0.91 },
  { id: "E", baseLearners: ["knn", "xgboost", "gradient_boosting"], metaLearner: "linear_regression", label: "KNN + XGB + GB → LR", paperRmse: 0.9 },
  { id: "F", baseLearners: ["knn", "random_forest"], metaLearner: "xgboost", label: "KNN + RF → XGB", paperRmse: 0.95 },
  { id: "G", baseLearners: ["knn", "xgboost", "gradient_boosting", "adaboost"], metaLearner: "linear_regression", label: "KNN + XGB + GB + AB → LR", paperRmse: 0.91 },
];

/** The paper's proposed model and its published error values. */
export const PAPER_PROPOSED = {
  id: "E",
  baseLearners: ["knn", "xgboost", "gradient_boosting"],
  metaLearner: "linear_regression",
  label: "KNN + XGBoost + Gradient Boosting → Linear Regression",
  mae: 0.69,
  mse: 0.82,
  rmse: 0.9,
} as const;

/** The paper's stated feature set for the final model. */
export const PAPER_FEATURES = ["user ID", "movie ID", "year", "rating"];

export const PAPER_NOTES = [
  "The paper merges the movie, rating and user files into one dataset, then states that user demographic information is not used in the final modeling.",
  "Reported final feature set: user ID, movie ID, year and rating (the rating column is the prediction target, not an input feature).",
  "Base learners: Linear Regression, KNN, Random Forest, AdaBoost, Gradient Boosting and XGBoost.",
  "Proposed stacked model: KNN + XGBoost + Gradient Boosting feeding a Linear Regression meta learner.",
  "The paper does not publish code, random seeds, train/test ratios, k-fold settings or hyperparameters, so those choices are ours and are recorded per run.",
  "The paper does not claim to solve the cold-start problem, and no cold-start mechanism from the paper is attributed here.",
];
