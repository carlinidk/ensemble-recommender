import { createRng } from "../rng";
import type { FeatureImportance, Matrix, Regressor } from "../types";
import {
  DEFAULT_TREE_PARAMS,
  buildTree,
  fitBinner,
  predictTree,
  transformBins,
  type BuiltTree,
  type TreeParams,
} from "./tree";
import { allIndices, gainsToImportance } from "./util";

export interface AdaBoostParams {
  nEstimators: number;
  learningRate: number;
  /** Depth of each weighted regression-tree base estimator. */
  maxDepth: number;
  minSamplesLeaf: number;
  maxBins: number;
  /** Power p of the AdaBoost.R2 loss (1 = linear loss, as in sklearn's default). */
  lossPower: number;
  seed: number;
}

export const DEFAULT_ADABOOST_PARAMS: AdaBoostParams = {
  nEstimators: 50,
  learningRate: 1,
  maxDepth: 4,
  minSamplesLeaf: 2,
  maxBins: 64,
  lossPower: 1,
  seed: 42,
};

interface WeightedTree {
  tree: BuiltTree;
  weight: number;
}

/**
 * AdaBoost.R2 (Drucker, 1997) — one of the paper's standalone base learners and
 * part of its `KNN + XGB + GB + AB -> LR` comparison.
 *
 * Follows the same steps as sklearn's `AdaBoostRegressor` with linear loss:
 * fit a weighted base estimator, compute the normalized maximum error, update
 * `beta = err/(1-err)`, re-weight samples by `beta^(1-err_i)`, and combine
 * estimators with a weighted median at prediction time.
 */
export class AdaBoostRegressor implements Regressor {
  readonly name = "adaboost";
  private estimators: WeightedTree[] = [];
  private names: string[] = [];
  private readonly params: AdaBoostParams;

  constructor(params: Partial<AdaBoostParams> = {}) {
    this.params = { ...DEFAULT_ADABOOST_PARAMS, ...params };
  }

  fit(X: Matrix, y: Float64Array): void {
    const { n } = X;
    this.names = X.names;
    const binner = fitBinner(X, this.params.maxBins);
    const bins = transformBins(X, binner);
    const rng = createRng(this.params.seed);
    const treeParams: TreeParams = {
      ...DEFAULT_TREE_PARAMS,
      maxDepth: this.params.maxDepth,
      minSamplesLeaf: this.params.minSamplesLeaf,
      minSamplesSplit: Math.max(2, this.params.minSamplesLeaf * 2),
      maxBins: this.params.maxBins,
      maxFeatures: 0,
    };
    const indices = allIndices(n);

    const weights = new Float64Array(n).fill(1 / n);
    const predictions = new Float64Array(n);
    this.estimators = [];

    for (let t = 0; t < this.params.nEstimators; t++) {
      const tree = buildTree(bins, { kind: "cart", target: y, weight: weights }, indices, treeParams, rng);
      for (let i = 0; i < n; i++) predictions[i] = predictTree(tree.root, X, i);

      let maxError = 0;
      for (let i = 0; i < n; i++) {
        const e = Math.abs(predictions[i] - y[i]);
        if (e > maxError) maxError = e;
      }
      if (maxError <= 1e-12) {
        this.estimators.push({ tree, weight: 1 });
        break;
      }

      let weightedError = 0;
      for (let i = 0; i < n; i++) {
        const err = Math.pow(Math.abs(predictions[i] - y[i]) / maxError, this.params.lossPower);
        weightedError += weights[i] * err;
      }
      const avgError = weightedError;
      // sklearn stops boosting once the base estimator is no better than chance.
      if (avgError >= 0.5) break;

      const beta = clamp(avgError / (1 - avgError), 1e-10, 1 - 1e-10);
      this.estimators.push({
        tree,
        weight: Math.log(1 / beta) * this.params.learningRate,
      });

      for (let i = 0; i < n; i++) {
        const err = Math.pow(Math.abs(predictions[i] - y[i]) / maxError, this.params.lossPower);
        weights[i] *= Math.pow(beta, 1 - err);
      }
      let total = 0;
      for (let i = 0; i < n; i++) total += weights[i];
      if (total <= 0) break;
      for (let i = 0; i < n; i++) weights[i] /= total;
    }
  }

  predict(X: Matrix): Float64Array {
    const out = new Float64Array(X.n);
    const t = this.estimators.length;
    if (t === 0) return out;
    const preds = new Float64Array(t);
    const ws = new Float64Array(t);
    const order = new Int32Array(t);
    for (let i = 0; i < X.n; i++) {
      let totalWeight = 0;
      for (let e = 0; e < t; e++) {
        preds[e] = predictTree(this.estimators[e].tree.root, X, i);
        ws[e] = this.estimators[e].weight;
        order[e] = e;
        totalWeight += ws[e];
      }
      // Insertion sort by prediction (t is small).
      for (let a = 1; a < t; a++) {
        const key = order[a];
        const keyVal = preds[key];
        let b = a - 1;
        while (b >= 0 && preds[order[b]] > keyVal) {
          order[b + 1] = order[b];
          b--;
        }
        order[b + 1] = key;
      }
      const half = totalWeight / 2;
      let cumulative = 0;
      let value = preds[order[t - 1]];
      for (let a = 0; a < t; a++) {
        cumulative += ws[order[a]];
        value = preds[order[a]];
        if (cumulative >= half) break;
      }
      out[i] = value;
    }
    return out;
  }

  importance(): FeatureImportance[] {
    if (this.estimators.length === 0) return [];
    const gains = new Float64Array(this.names.length);
    for (const est of this.estimators) {
      for (let i = 0; i < gains.length; i++) gains[i] += est.tree.importance[i] * est.weight;
    }
    return gainsToImportance(gains, this.names);
  }

  describe(): Record<string, number | string | boolean> {
    return {
      n_estimators: this.params.nEstimators,
      learning_rate: this.params.learningRate,
      base_estimator_max_depth: this.params.maxDepth,
      loss: `linear (AdaBoost.R2, p=${this.params.lossPower})`,
      estimators_fitted: this.estimators.length,
      combination: "weighted median",
    };
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}
