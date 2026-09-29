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
import { bootstrapIndices, gainsToImportance } from "./util";

export interface RandomForestParams {
  nEstimators: number;
  maxDepth: number;
  minSamplesLeaf: number;
  minSamplesSplit: number;
  maxBins: number;
  /** 0 = every feature is considered at every node (sklearn's regression default). */
  maxFeatures: number;
  seed: number;
}

export const DEFAULT_RANDOM_FOREST_PARAMS: RandomForestParams = {
  nEstimators: 80,
  maxDepth: 14,
  minSamplesLeaf: 5,
  minSamplesSplit: 10,
  maxBins: 64,
  maxFeatures: 0,
  seed: 42,
};

/**
 * Random Forest Regressor — one of the paper's standalone base learners and part
 * of its meta-learner comparisons (KNN + GB + RF -> XGB, KNN + RF -> XGB).
 */
export class RandomForestRegressor implements Regressor {
  readonly name = "random_forest";
  private trees: BuiltTree[] = [];
  private names: string[] = [];
  private readonly params: RandomForestParams;

  constructor(params: Partial<RandomForestParams> = {}) {
    this.params = { ...DEFAULT_RANDOM_FOREST_PARAMS, ...params };
  }

  fit(X: Matrix, y: Float64Array): void {
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
      maxFeatures: this.params.maxFeatures,
    };
    this.names = X.names;
    this.trees = [];
    for (let t = 0; t < this.params.nEstimators; t++) {
      const sample = bootstrapIndices(rng, X.n);
      this.trees.push(buildTree(bins, { kind: "cart", target: y }, sample, treeParams, rng));
    }
  }

  predict(X: Matrix): Float64Array {
    if (this.trees.length === 0) throw new Error("RandomForest.predict called before fit");
    const out = new Float64Array(X.n);
    for (const tree of this.trees) {
      for (let i = 0; i < X.n; i++) out[i] += predictTree(tree.root, X, i);
    }
    for (let i = 0; i < X.n; i++) out[i] /= this.trees.length;
    return out;
  }

  importance(): FeatureImportance[] {
    if (this.trees.length === 0) return [];
    const gains = new Float64Array(this.names.length);
    for (const tree of this.trees) {
      for (let i = 0; i < gains.length; i++) gains[i] += tree.importance[i] / this.trees.length;
    }
    return gainsToImportance(gains, this.names);
  }

  describe(): Record<string, number | string | boolean> {
    return {
      n_estimators: this.params.nEstimators,
      max_depth: this.params.maxDepth,
      min_samples_leaf: this.params.minSamplesLeaf,
      min_samples_split: this.params.minSamplesSplit,
      max_bins: this.params.maxBins,
      bootstrap: true,
      trees_fitted: this.trees.length,
    };
  }
}
