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

export interface GradientBoostingParams {
  nEstimators: number;
  learningRate: number;
  maxDepth: number;
  minSamplesLeaf: number;
  minSamplesSplit: number;
  maxBins: number;
  /** Row subsample per stage (sklearn's `subsample`). */
  subsample: number;
  seed: number;
}

export const DEFAULT_GRADIENT_BOOSTING_PARAMS: GradientBoostingParams = {
  nEstimators: 120,
  learningRate: 0.1,
  maxDepth: 3,
  minSamplesLeaf: 3,
  minSamplesSplit: 6,
  maxBins: 64,
  subsample: 0.7,
  seed: 42,
};

/**
 * Gradient Boosting Regressor — one of the paper's base learners and the third
 * member of its proposed stack.
 *
 * Squared-error loss, so each stage fits a regression tree to the current
 * residuals: `F_m(x) = F_{m-1}(x) + lr * h_m(x)`.
 */
export class GradientBoostingRegressor implements Regressor {
  readonly name = "gradient_boosting";
  private trees: BuiltTree[] = [];
  private base = 0;
  private names: string[] = [];
  private readonly params: GradientBoostingParams;

  constructor(params: Partial<GradientBoostingParams> = {}) {
    this.params = { ...DEFAULT_GRADIENT_BOOSTING_PARAMS, ...params };
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
      minChildWeight: this.params.minSamplesLeaf,
      minSamplesSplit: this.params.minSamplesSplit,
      maxBins: this.params.maxBins,
      maxFeatures: 0,
    };

    let sum = 0;
    for (let i = 0; i < n; i++) sum += y[i];
    this.base = sum / n;

    const fitted = new Float64Array(n).fill(this.base);
    const residual = new Float64Array(n);
    const sampleSize = Math.max(1, Math.round(this.params.subsample * n));
    this.trees = [];

    for (let t = 0; t < this.params.nEstimators; t++) {
      for (let i = 0; i < n; i++) residual[i] = y[i] - fitted[i];
      const idx =
        this.params.subsample < 1 ? sampleIndices(rng, n, sampleSize) : undefined;
      const tree = buildTree(
        bins,
        { kind: "cart", target: residual },
        idx ?? fullRange(n),
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
      min_samples_leaf: this.params.minSamplesLeaf,
      subsample: this.params.subsample,
      loss: "squared_error",
      initial_estimate: Number(this.base.toFixed(4)),
    };
  }
}

function fullRange(n: number): Int32Array {
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) out[i] = i;
  return out;
}
