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
import { gainsToImportance, sampleIndices } from "./util";

export interface XGBoostParams {
  nEstimators: number;
  learningRate: number;
  maxDepth: number;
  /** Minimum hessian mass per child. */
  minChildWeight: number;
  lambda: number;
  gamma: number;
  subsample: number;
  colsampleByTree: number;
  maxBins: number;
  seed: number;
}

export const DEFAULT_XGBOOST_PARAMS: XGBoostParams = {
  nEstimators: 180,
  learningRate: 0.1,
  maxDepth: 6,
  minChildWeight: 2,
  lambda: 1,
  gamma: 0,
  subsample: 0.8,
  colsampleByTree: 1,
  maxBins: 64,
  seed: 42,
};

/**
 * XGBoost Regressor — the paper's strongest standalone base learner and a member
 * of its proposed stack.
 *
 * RESEARCH-INTEGRITY NOTE: the paper does not publish code, only algorithm and
 * hyperparameter names. This is a from-scratch implementation of the XGBoost
 * *algorithm* (second-order Taylor expansion of the loss, regularized leaf
 * weights, weighted quantile binning, subsampling, shrinkage) inside this
 * repository — it is NOT the official `xgboost` library. It is therefore an
 * implementation-level reproduction whose numeric output cannot be expected to
 * match the paper's library runs exactly; the deviation is documented in the
 * research report.
 */
export class XGBoostRegressor implements Regressor {
  readonly name = "xgboost";
  private trees: BuiltTree[] = [];
  private base = 0;
  private names: string[] = [];
  private readonly params: XGBoostParams;

  constructor(params: Partial<XGBoostParams> = {}) {
    this.params = { ...DEFAULT_XGBOOST_PARAMS, ...params };
  }

  fit(X: Matrix, y: Float64Array): void {
    const { n, d } = X;
    this.names = X.names;
    const binner = fitBinner(X, this.params.maxBins);
    const bins = transformBins(X, binner);
    const rng = createRng(this.params.seed);
    const maxFeatures = Math.max(
      1,
      Math.min(d, Math.round(this.params.colsampleByTree * d)),
    );
    const treeParams: TreeParams = {
      ...DEFAULT_TREE_PARAMS,
      maxDepth: this.params.maxDepth,
      minChildWeight: this.params.minChildWeight,
      minSamplesLeaf: 1,
      minSamplesSplit: 2,
      maxBins: this.params.maxBins,
      lambda: this.params.lambda,
      gamma: this.params.gamma,
      maxFeatures: maxFeatures < d ? maxFeatures : 0,
    };

    let sum = 0;
    for (let i = 0; i < n; i++) sum += y[i];
    this.base = sum / n; // XGBoost's base_score

    const fitted = new Float64Array(n).fill(this.base);
    const gradient = new Float64Array(n);
    const hessian = new Float64Array(n);
    const sampleSize = Math.max(1, Math.round(this.params.subsample * n));
    this.trees = [];

    for (let t = 0; t < this.params.nEstimators; t++) {
      // Squared-error loss: g = pred - y, h = 1.
      for (let i = 0; i < n; i++) {
        gradient[i] = fitted[i] - y[i];
        hessian[i] = 1;
      }
      const idx =
        this.params.subsample < 1 ? sampleIndices(rng, n, sampleSize) : undefined;
      const subset = idx ?? plainRange(n);
      const tree = buildTree(
        bins,
        { kind: "xgb", gradient, hessian },
        subset,
        treeParams,
        rng,
      );
      this.trees.push(tree);
      for (let i = 0; i < n; i++) {
        fitted[i] += this.params.learningRate * predictTree(tree.root, X, i);
      }
    }
  }

  predict(X: Matrix): Float64Array {
    const out = new Float64Array(X.n).fill(this.base);
    for (const tree of this.trees) {
      for (let i = 0; i < X.n; i++) {
        out[i] += this.params.learningRate * predictTree(tree.root, X, i);
      }
    }
    return out;
  }

  importance(): FeatureImportance[] {
    if (this.trees.length === 0) return [];
    const gains = new Float64Array(this.names.length);
    for (const tree of this.trees) {
      for (let i = 0; i < gains.length; i++) gains[i] += tree.importance[i];
    }
    return gainsToImportance(gains, this.names);
  }

  describe(): Record<string, number | string | boolean> {
    return {
      n_estimators: this.params.nEstimators,
      learning_rate: this.params.learningRate,
      max_depth: this.params.maxDepth,
      min_child_weight: this.params.minChildWeight,
      lambda: this.params.lambda,
      gamma: this.params.gamma,
      subsample: this.params.subsample,
      colsample_bytree: this.params.colsampleByTree,
      max_bins: this.params.maxBins,
      implementation: "from-scratch XGBoost algorithm (not the official library)",
    };
  }
}

function plainRange(n: number): Int32Array {
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) out[i] = i;
  return out;
}
