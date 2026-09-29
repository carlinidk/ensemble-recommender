/**
 * Deterministic pseudo-random number generation.
 *
 * The paper does not publish its random seed, so our reproduction fixes one
 * (`config.yaml` / `TrainingConfig.randomSeed`) and every stochastic step in
 * the pipeline — shuffling, train/test splitting, k-fold assignment, bootstrap
 * sampling, subsampling — draws from a `Rng` created here. Re-running the same
 * config on the same dataset therefore reproduces identical numbers.
 */

export interface Rng {
  /** Uniform in [0, 1). */
  next(): number;
  /** Uniform integer in [0, max). */
  nextInt(max: number): number;
}

/** mulberry32 — small, fast, deterministic 32-bit PRNG. */
export function createRng(seed: number): Rng {
  let state = seed >>> 0;
  const next = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return {
    next,
    nextInt: (max: number) => Math.floor(next() * max),
  };
}

/** Fisher–Yates shuffle, in place. */
export function shuffleInPlace(values: Int32Array | number[], rng: Rng): void {
  for (let i = values.length - 1; i > 0; i--) {
    const j = rng.nextInt(i + 1);
    const tmp = values[i];
    values[i] = values[j];
    values[j] = tmp;
  }
}

export function range(n: number): Int32Array {
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) out[i] = i;
  return out;
}

export interface Split {
  train: Int32Array;
  test: Int32Array;
}

/**
 * Random split (paper mode). Temporal split lives in `src/ml/split.ts` and is
 * offered as an ENHANCED-MODE evaluation alternative.
 */
export function trainTestSplit(
  n: number,
  testSize: number,
  rng: Rng,
): Split {
  const order = range(n);
  shuffleInPlace(order, rng);
  const nTest = Math.max(1, Math.min(n - 1, Math.round(n * testSize)));
  const test = order.slice(0, nTest);
  const train = order.slice(nTest);
  return { train, test };
}

/** Contiguous k-fold assignment over an already-shuffled index array. */
export function kFoldIndices(n: number, folds: number, rng: Rng): Int32Array[] {
  const order = range(n);
  shuffleInPlace(order, rng);
  const k = Math.max(2, Math.min(folds, n));
  const buckets: number[][] = Array.from({ length: k }, () => []);
  for (let i = 0; i < n; i++) buckets[i % k].push(order[i]);
  return buckets.map((b) => Int32Array.from(b));
}

/** Reservoir-free deterministic subsample of `size` indices out of `n`. */
export function subsampleIndices(n: number, size: number, rng: Rng): Int32Array {
  if (size >= n) return range(n);
  const order = range(n);
  shuffleInPlace(order, rng);
  return order.slice(0, size).sort();
}
