import type { FeatureImportance, Matrix, Regressor } from "../types";

/**
 * K-Nearest Neighbours Regressor — one of the paper's base learners and part of
 * its proposed stack (KNN + XGBoost + Gradient Boosting -> Linear Regression).
 *
 * The paper does not state `k`, the weighting scheme or whether the id features
 * were scaled, so those choices are ours and are recorded in the run metadata:
 *
 *   - `k` is configurable (see `src/ml/models/index.ts` parameter sets).
 *   - features are standardized before distance computation, because raw ids
 *     span 1-1682 and would otherwise dominate the Euclidean metric;
 *     standardization is fitted on the training split only.
 *   - neighbours are found with a kd-tree (quickselect build) so the full
 *     80k-row training split stays tractable in a browser worker.
 */

export interface KnnParams {
  k: number;
  weights: "uniform" | "distance";
  leafSize: number;
  /**
   * Dimension threshold above which the neighbour pool is capped. kd-tree
   * pruning collapses in higher dimensions, so a query degenerates into a scan
   * of every point. The enhanced feature set crosses that threshold, and the
   * honest response is to cap the pool and say so in the model metadata rather
   * than to let a full-dataset enhanced run take many minutes.
   */
  maxDimensions: number;
  /** Reference-set size used once `maxDimensions` is exceeded. */
  cappedPoolSize: number;
}

export const DEFAULT_KNN_PARAMS: KnnParams = {
  k: 40,
  weights: "uniform",
  leafSize: 16,
  maxDimensions: 6,
  cappedPoolSize: 5000,
};

/** Small fixed-capacity max-heap-free neighbour buffer (k is tiny). */
class NeighborBuffer {
  dist: Float64Array;
  index: Int32Array;
  size = 0;
  readonly k: number;

  constructor(k: number) {
    this.k = k;
    this.dist = new Float64Array(k);
    this.index = new Int32Array(k);
  }

  reset(): void {
    this.size = 0;
  }

  worst(): number {
    return this.size < this.k ? Number.POSITIVE_INFINITY : this.dist[this.size - 1];
  }

  push(d: number, idx: number): void {
    if (this.size < this.k) {
      let i = this.size++;
      while (i > 0 && this.dist[i - 1] > d) {
        this.dist[i] = this.dist[i - 1];
        this.index[i] = this.index[i - 1];
        i--;
      }
      this.dist[i] = d;
      this.index[i] = idx;
      return;
    }
    if (d >= this.dist[this.size - 1]) return;
    let i = this.size - 1;
    while (i > 0 && this.dist[i - 1] > d) {
      this.dist[i] = this.dist[i - 1];
      this.index[i] = this.index[i - 1];
      i--;
    }
    this.dist[i] = d;
    this.index[i] = idx;
  }
}

class KdTree {
  private points: Float64Array;
  private perm: Int32Array;
  private nodeAxis: Int32Array;
  private nodeSplit: Float64Array;
  private nodeLeft: Int32Array;
  private nodeRight: Int32Array;
  private nodeStart: Int32Array;
  private nodeEnd: Int32Array;
  private nodeCount = 0;
  private readonly d: number;
  private readonly leafSize: number;

  constructor(points: Float64Array, n: number, d: number, leafSize: number) {
    this.points = points;
    this.d = d;
    this.leafSize = leafSize;
    this.perm = new Int32Array(n);
    for (let i = 0; i < n; i++) this.perm[i] = i;
    const capacity = Math.max(4, 4 * Math.ceil(n / leafSize) + 8);
    this.nodeAxis = new Int32Array(capacity);
    this.nodeSplit = new Float64Array(capacity);
    this.nodeLeft = new Int32Array(capacity);
    this.nodeRight = new Int32Array(capacity);
    this.nodeStart = new Int32Array(capacity);
    this.nodeEnd = new Int32Array(capacity);
    this.build(0, n, 0);
  }

  private coord(row: number, axis: number): number {
    return this.points[row * this.d + axis];
  }

  private build(start: number, end: number, depth: number): number {
    const id = this.nodeCount++;
    if (this.nodeCount > this.nodeAxis.length) {
      // Grow geometrically if the estimate was optimistic.
      const grown = this.nodeAxis.length * 2;
      const axis = new Int32Array(grown);
      axis.set(this.nodeAxis);
      this.nodeAxis = axis;
      const split = new Float64Array(grown);
      split.set(this.nodeSplit);
      this.nodeSplit = split;
      const left = new Int32Array(grown);
      left.set(this.nodeLeft);
      this.nodeLeft = left;
      const right = new Int32Array(grown);
      right.set(this.nodeRight);
      this.nodeRight = right;
      const s = new Int32Array(grown);
      s.set(this.nodeStart);
      this.nodeStart = s;
      const e = new Int32Array(grown);
      e.set(this.nodeEnd);
      this.nodeEnd = e;
    }

    this.nodeStart[id] = start;
    this.nodeEnd[id] = end;

    if (end - start <= this.leafSize) {
      this.nodeAxis[id] = -1;
      return id;
    }

    // Split on the axis with the largest spread among this node's points.
    let axis = depth % this.d;
    let bestSpread = -1;
    for (let a = 0; a < this.d; a++) {
      let min = Infinity;
      let max = -Infinity;
      for (let i = start; i < end; i++) {
        const v = this.coord(this.perm[i], a);
        if (v < min) min = v;
        if (v > max) max = v;
      }
      const spread = max - min;
      if (spread > bestSpread) {
        bestSpread = spread;
        axis = a;
      }
    }
    if (bestSpread <= 0) {
      this.nodeAxis[id] = -1;
      return id;
    }

    const mid = (start + end) >> 1;
    this.quickselect(start, end, mid, axis);
    const splitValue = this.coord(this.perm[mid], axis);

    this.nodeAxis[id] = axis;
    this.nodeSplit[id] = splitValue;
    this.nodeLeft[id] = this.build(start, mid, depth + 1);
    this.nodeRight[id] = this.build(mid, end, depth + 1);
    return id;
  }

  private quickselect(start: number, end: number, k: number, axis: number): void {
    let lo = start;
    let hi = end - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      const pivot = this.coord(this.perm[mid], axis);
      let i = lo;
      let j = hi;
      while (i <= j) {
        while (this.coord(this.perm[i], axis) < pivot) i++;
        while (this.coord(this.perm[j], axis) > pivot) j--;
        if (i <= j) {
          const tmp = this.perm[i];
          this.perm[i] = this.perm[j];
          this.perm[j] = tmp;
          i++;
          j--;
        }
      }
      if (k <= j) hi = j;
      else if (k >= i) lo = i;
      else break;
    }
  }

  query(q: Float64Array, buffer: NeighborBuffer): void {
    buffer.reset();
    this.search(0, q, buffer);
  }

  private search(node: number, q: Float64Array, buffer: NeighborBuffer): void {
    const axis = this.nodeAxis[node];
    if (axis < 0) {
      for (let i = this.nodeStart[node]; i < this.nodeEnd[node]; i++) {
        const p = this.perm[i];
        let dist = 0;
        for (let a = 0; a < this.d; a++) {
          const dv = this.coord(p, a) - q[a];
          dist += dv * dv;
        }
        buffer.push(dist, p);
      }
      return;
    }
    const diff = q[axis] - this.nodeSplit[node];
    const near = diff <= 0 ? this.nodeLeft[node] : this.nodeRight[node];
    const far = diff <= 0 ? this.nodeRight[node] : this.nodeLeft[node];
    this.search(near, q, buffer);
    if (diff * diff <= buffer.worst()) this.search(far, q, buffer);
  }
}

export class KnnRegressor implements Regressor {
  readonly name = "knn";
  private means = new Float64Array(0);
  private scales = new Float64Array(0);
  private tree: KdTree | null = null;
  private targets = new Float64Array(0);
  private buffer: NeighborBuffer;
  private readonly params: KnnParams;
  private scaling = true;
  private poolSize = 0;
  private poolCapped = false;

  constructor(params: Partial<KnnParams> = {}) {
    this.params = { ...DEFAULT_KNN_PARAMS, ...params };
    this.buffer = new NeighborBuffer(this.params.k);
  }

  /** Disable standardization (used by the "raw ids" ablation). */
  setScaling(enabled: boolean): void {
    this.scaling = enabled;
  }

  fit(X: Matrix, y: Float64Array): void {
    const { n, d } = X;
    this.means = new Float64Array(d);
    this.scales = new Float64Array(d);
    for (let j = 0; j < d; j++) {
      let sum = 0;
      for (let i = 0; i < n; i++) sum += X.data[i * d + j];
      const mean = sum / n;
      let variance = 0;
      for (let i = 0; i < n; i++) {
        const dv = X.data[i * d + j] - mean;
        variance += dv * dv;
      }
      this.means[j] = mean;
      this.scales[j] = this.scaling ? Math.sqrt(variance / n) || 1 : 1;
    }

    // Deterministic cap: a fixed stride keeps the pool reproducible across runs.
    this.poolCapped = d > this.params.maxDimensions && n > this.params.cappedPoolSize;
    const pool = this.poolCapped ? this.params.cappedPoolSize : n;
    const stride = this.poolCapped ? n / pool : 1;
    this.poolSize = Math.floor(pool);

    const points = new Float64Array(this.poolSize * d);
    const targets = new Float64Array(this.poolSize);
    let cursor = 0;
    for (let k = 0; k < this.poolSize; k++) {
      const i = this.poolCapped ? Math.floor(k * stride) : k;
      for (let j = 0; j < d; j++) {
        points[cursor * d + j] = (X.data[i * d + j] - this.means[j]) / this.scales[j];
      }
      targets[cursor] = y[i];
      cursor++;
    }

    this.targets = targets;
    this.tree = new KdTree(points, this.poolSize, d, this.params.leafSize);
  }

  predict(X: Matrix): Float64Array {
    if (!this.tree) throw new Error("KnnRegressor.predict called before fit");
    const out = new Float64Array(X.n);
    const d = X.d;
    const q = new Float64Array(d);
    for (let i = 0; i < X.n; i++) {
      for (let j = 0; j < d; j++) {
        q[j] = (X.data[i * d + j] - this.means[j]) / this.scales[j];
      }
      this.tree.query(q, this.buffer);
      let weighted = 0;
      let totalWeight = 0;
      for (let b = 0; b < this.buffer.size; b++) {
        const w =
          this.params.weights === "distance"
            ? 1 / (Math.sqrt(this.buffer.dist[b]) + 1e-6)
            : 1;
        weighted += w * this.targets[this.buffer.index[b]];
        totalWeight += w;
      }
      out[i] = totalWeight > 0 ? weighted / totalWeight : 0;
    }
    return out;
  }

  importance(): FeatureImportance[] {
    // Distance-based learners expose no split gain; report a zero vector rather
    // than inventing one.
    return [];
  }

  describe(): Record<string, number | string | boolean> {
    return {
      k: this.params.k,
      weights: this.params.weights,
      standardization: this.scaling,
      search: "kd-tree (quickselect build)",
      neighbor_pool: this.poolSize,
      neighbor_pool_capped: this.poolCapped
        ? `yes — capped at ${this.params.cappedPoolSize} rows because the feature set exceeds ${this.params.maxDimensions} dimensions`
        : "no",
    };
  }
}
