import type { Rng } from "../rng";
import type { Matrix } from "../types";

/**
 * Histogram-based CART regression trees.
 *
 * One implementation backs Random Forest, Gradient Boosting, AdaBoost.R2 and
 * the XGBoost-style booster. Two objectives are supported:
 *
 *  - "cart": variance reduction with optional sample weights (weighted SSE).
 *    Used by Random Forest, Gradient Boosting (fitted on residuals) and
 *    AdaBoost.R2 (fitted on weighted samples).
 *  - "xgb": second-order gain `0.5*(GL²/(HL+λ) + GR²/(HR+λ) - G²/(H+λ)) - γ`
 *    with regularized leaf weights `-G/(H+λ)`, i.e. the XGBoost split rule.
 *
 * Split search is histogram based: continuous features are quantile-binned
 * (like XGBoost's `max_bin`), which is what keeps full-dataset training in the
 * browser tractable. Bin edges are fitted once per model fit on the training
 * split only, so binning never observes test rows.
 */

export interface TreeParams {
  maxDepth: number;
  minSamplesLeaf: number;
  /** Minimum weighted child mass (count for unweighted CART, hessian sum for xgb). */
  minChildWeight: number;
  minSamplesSplit: number;
  maxBins: number;
  /** L2 regularization on leaf weights (xgb objective). */
  lambda: number;
  /** Minimum split gain required to make a split (xgb objective). */
  gamma: number;
  /** Features considered per node; 0 means all. */
  maxFeatures: number;
}

export const DEFAULT_TREE_PARAMS: TreeParams = {
  maxDepth: 6,
  minSamplesLeaf: 2,
  minChildWeight: 2,
  minSamplesSplit: 4,
  maxBins: 64,
  lambda: 1,
  gamma: 0,
  maxFeatures: 0,
};

export interface TreeNode {
  /** -1 for a leaf. */
  feature: number;
  /** Left child holds rows whose bin index is `< binSplit`. */
  binSplit: number;
  /** Raw threshold: values `<= edge` go left (kept for explanation output). */
  edge: number;
  /** Prediction value for the node (leaf weight). */
  value: number;
  samples: number;
  left: TreeNode | null;
  right: TreeNode | null;
}

export interface Binner {
  /** Per feature: upper bound of each bin, last entry is +Infinity. */
  bounds: Float64Array[];
  /** Per feature: bin index per row of the fitted matrix. */
  columns: Uint16Array[];
  maxBins: number;
}

/** Fit quantile bin edges on the training matrix. */
export function fitBinner(X: Matrix, maxBins: number): Binner {
  const bounds: Float64Array[] = [];
  for (let j = 0; j < X.d; j++) {
    const values = new Float64Array(X.n);
    for (let i = 0; i < X.n; i++) values[i] = X.data[i * X.d + j];
    values.sort();
    const n = values.length;
    const wantBins = Math.max(2, Math.min(maxBins, Math.max(2, uniqueCount(values))));
    const edges: number[] = [];
    for (let b = 1; b < wantBins; b++) {
      const edge = values[Math.min(n - 1, Math.floor((b * n) / wantBins))];
      if (edges.length === 0 || edge > edges[edges.length - 1]) edges.push(edge);
    }
    const featureBounds = new Float64Array(edges.length + 1);
    for (let i = 0; i < edges.length; i++) featureBounds[i] = edges[i];
    featureBounds[edges.length] = Number.POSITIVE_INFINITY;
    bounds.push(featureBounds);
  }
  return { bounds, columns: [], maxBins };
}

function uniqueCount(sorted: Float64Array): number {
  if (sorted.length === 0) return 0;
  let count = 1;
  for (let i = 1; i < sorted.length; i++) if (sorted[i] !== sorted[i - 1]) count++;
  return count;
}

/** Map a matrix's rows into bin indices using previously fitted bounds. */
export function transformBins(X: Matrix, binner: Binner): Binner {
  const columns: Uint16Array[] = [];
  for (let j = 0; j < X.d; j++) {
    const col = new Uint16Array(X.n);
    const b = binner.bounds[j];
    for (let i = 0; i < X.n; i++) {
      const v = X.data[i * X.d + j];
      let lo = 0;
      let hi = b.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (v <= b[mid]) hi = mid;
        else lo = mid + 1;
      }
      col[i] = lo;
    }
    columns.push(col);
  }
  return { bounds: binner.bounds, columns, maxBins: binner.maxBins };
}

export type TreeObjective =
  | { kind: "cart"; target: Float64Array; weight?: Float64Array }
  | { kind: "xgb"; gradient: Float64Array; hessian: Float64Array };

export interface BuiltTree {
  root: TreeNode;
  /** Sum of gain per feature (not normalized). */
  importance: Float64Array;
  nodes: number;
  leaves: number;
}

function leaf(value: number, samples: number): TreeNode {
  return {
    feature: -1,
    binSplit: -1,
    edge: 0,
    value,
    samples,
    left: null,
    right: null,
  };
}

/**
 * Build one regression tree over `indices` (row indices into the binned columns).
 * `bins` must already be transformed from the same matrix the indices refer to.
 */
export function buildTree(
  bins: Binner,
  objective: TreeObjective,
  indices: Int32Array,
  params: TreeParams,
  rng?: Rng,
): BuiltTree {
  const d = bins.bounds.length;
  const importance = new Float64Array(d);
  const scratchSize = Math.max(...bins.bounds.map((b) => b.length));
  const sumBuf = new Float64Array(scratchSize);
  const cntBuf = new Float64Array(scratchSize);
  const sqBuf = new Float64Array(scratchSize);
  const columns = bins.columns;
  const cart = objective.kind === "cart";

  // Sample-weighted CART (AdaBoost.R2) normalizes weights to sum to 1, so the
  // leaf-mass threshold has to be expressed in the same units as the weights.
  let weightScale = 1;
  if (cart && (objective as { weight?: Float64Array }).weight) {
    const w = (objective as { weight: Float64Array }).weight;
    let total = 0;
    for (let i = 0; i < indices.length; i++) total += w[indices[i]];
    weightScale = total > 0 ? total / indices.length : 1;
  }
  const minChild = cart
    ? Math.max(1e-9, params.minSamplesLeaf * weightScale)
    : Math.max(0, params.minChildWeight);
  let nodes = 0;
  let leaves = 0;

  function build(idx: Int32Array, depth: number): TreeNode {
    nodes++;
    const n = idx.length;
    let totalSum = 0;
    let totalCnt = 0;
    let totalSq = 0;
    if (cart) {
      const target = (objective as { target: Float64Array }).target;
      const weight = (objective as { weight?: Float64Array }).weight;
      for (let i = 0; i < n; i++) {
        const r = idx[i];
        const w = weight ? weight[r] : 1;
        totalSum += w * target[r];
        totalCnt += w;
        totalSq += w * target[r] * target[r];
      }
    } else {
      const g = (objective as { gradient: Float64Array }).gradient;
      const h = (objective as { hessian: Float64Array }).hessian;
      for (let i = 0; i < n; i++) {
        const r = idx[i];
        totalSum += g[r];
        totalCnt += h[r];
        totalSq += 0;
      }
    }

    const lambda = cart ? 0 : params.lambda;
    const value = cart ? totalSum / totalCnt : -totalSum / (totalCnt + lambda);

    const tooSmall = n < Math.max(2, params.minSamplesSplit);
    const tooDeep = depth >= params.maxDepth;
    const degenerate = totalCnt <= 0 || totalCnt <= minChild * 2;
    if (tooSmall || tooDeep || degenerate) {
      leaves++;
      return leaf(value, n);
    }

    // Feature subset (colsample_bytree-style) for randomized ensembles.
    let features: number[];
    if (params.maxFeatures > 0 && params.maxFeatures < d && rng) {
      const pool = Array.from({ length: d }, (_, i) => i);
      features = [];
      for (let i = 0; i < params.maxFeatures && pool.length > 0; i++) {
        features.push(pool.splice(rng.nextInt(pool.length), 1)[0]);
      }
    } else {
      features = Array.from({ length: d }, (_, i) => i);
    }

    let bestFeature = -1;
    let bestBin = -1;
    let bestEdge = 0;
    let bestScore = 0;
    const parentSse = cart ? totalSq - (totalSum * totalSum) / totalCnt : 0;
    const parentXgbGain = cart
      ? 0
      : (totalSum * totalSum) / (totalCnt + lambda);

    for (const f of features) {
      const col = columns[f];
      const bounds = bins.bounds[f];
      const nbins = bounds.length;
      for (let b = 0; b < nbins; b++) {
        sumBuf[b] = 0;
        cntBuf[b] = 0;
        sqBuf[b] = 0;
      }
      if (cart) {
        const target = (objective as { target: Float64Array }).target;
        const weight = (objective as { weight?: Float64Array }).weight;
        for (let i = 0; i < n; i++) {
          const r = idx[i];
          const w = weight ? weight[r] : 1;
          const b = col[r];
          sumBuf[b] += w * target[r];
          cntBuf[b] += w;
          sqBuf[b] += w * target[r] * target[r];
        }
      } else {
        const g = (objective as { gradient: Float64Array }).gradient;
        const h = (objective as { hessian: Float64Array }).hessian;
        for (let i = 0; i < n; i++) {
          const r = idx[i];
          const b = col[r];
          sumBuf[b] += g[r];
          cntBuf[b] += h[r];
        }
      }

      let leftSum = 0;
      let leftCnt = 0;
      let leftSq = 0;
      for (let b = 0; b < nbins - 1; b++) {
        leftSum += sumBuf[b];
        leftCnt += cntBuf[b];
        leftSq += sqBuf[b];
        if (leftCnt < minChild) continue;
        const rightCnt = totalCnt - leftCnt;
        if (rightCnt < minChild) continue;
        const rightSum = totalSum - leftSum;
        let score: number;
        if (cart) {
          const leftSse = leftSq - (leftSum * leftSum) / leftCnt;
          const rightSse = totalSq - leftSq - (rightSum * rightSum) / rightCnt;
          score = parentSse - leftSse - rightSse;
        } else {
          score =
            0.5 *
              ((leftSum * leftSum) / (leftCnt + lambda) +
                (rightSum * rightSum) / (rightCnt + lambda) -
                parentXgbGain) -
            params.gamma;
        }
        if (score > bestScore) {
          bestScore = score;
          bestFeature = f;
          bestBin = b + 1;
          bestEdge = bounds[b];
        }
      }
    }

    if (bestFeature < 0) {
      leaves++;
      return leaf(value, n);
    }

    importance[bestFeature] += bestScore * n;

    let leftCount = 0;
    const col = columns[bestFeature];
    for (let i = 0; i < n; i++) if (col[idx[i]] < bestBin) leftCount++;
    const leftIdx = new Int32Array(leftCount);
    const rightIdx = new Int32Array(n - leftCount);
    let li = 0;
    let ri = 0;
    for (let i = 0; i < n; i++) {
      const r = idx[i];
      if (col[r] < bestBin) leftIdx[li++] = r;
      else rightIdx[ri++] = r;
    }

    return {
      feature: bestFeature,
      binSplit: bestBin,
      edge: bestEdge,
      value,
      samples: n,
      left: build(leftIdx, depth + 1),
      right: build(rightIdx, depth + 1),
    };
  }

  const root = build(indices, 0);
  return { root, importance, nodes, leaves };
}

/** Predict one row of the raw (unbinned) matrix. */
export function predictTree(
  root: TreeNode,
  X: Matrix,
  row: number,
): number {
  let node = root;
  while (node.feature >= 0) {
    const v = X.data[row * X.d + node.feature];
    node = v <= node.edge ? node.left! : node.right!;
  }
  return node.value;
}

export function predictTreeVector(
  root: TreeNode,
  X: Matrix,
  out: Float64Array,
  offset: number,
  scale: number,
): void {
  for (let i = 0; i < X.n; i++) out[offset + i] += scale * predictTree(root, X, i);
}

export function countLeaves(root: TreeNode): number {
  if (root.feature < 0) return 1;
  return countLeaves(root.left!) + countLeaves(root.right!);
}

export function treeDepth(root: TreeNode): number {
  if (root.feature < 0) return 1;
  return 1 + Math.max(treeDepth(root.left!), treeDepth(root.right!));
}
