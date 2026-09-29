import type { Rng } from "../rng";
import type { FeatureImportance } from "../types";

export function allIndices(n: number): Int32Array {
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) out[i] = i;
  return out;
}

/** Subsample without replacement. */
export function sampleIndices(rng: Rng, n: number, size: number): Int32Array {
  if (size >= n) return allIndices(n);
  const pool = allIndices(n);
  for (let i = n - 1; i > 0; i--) {
    const j = rng.nextInt(i + 1);
    const tmp = pool[i];
    pool[i] = pool[j];
    pool[j] = tmp;
  }
  const out = pool.slice(0, size);
  out.sort();
  return out;
}

/** Bootstrap sample with replacement (Random Forest). */
export function bootstrapIndices(rng: Rng, n: number): Int32Array {
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) out[i] = rng.nextInt(n);
  return out;
}

export function gainsToImportance(
  gains: Float64Array,
  names: string[],
): FeatureImportance[] {
  let total = 0;
  for (let i = 0; i < gains.length; i++) total += gains[i];
  return names.map((name, i) => ({
    name,
    value: total > 0 ? gains[i] / total : 0,
  }));
}
