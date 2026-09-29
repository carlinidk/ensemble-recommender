import type { Metrics } from "./types";

/**
 * The paper evaluates with MAE, MSE and RMSE. All three are computed here from
 * our own predictions — no published number is ever re-used as a result.
 *
 *   MAE  = (1/n) * SUM |y_i - yhat_i|
 *   MSE  = (1/n) * SUM (y_i - yhat_i)^2
 *   RMSE = sqrt(MSE)
 */
export function evaluate(
  actual: ArrayLike<number>,
  predicted: ArrayLike<number>,
): Metrics {
  const n = Math.min(actual.length, predicted.length);
  if (n === 0) return { mae: 0, mse: 0, rmse: 0, n: 0 };
  let absSum = 0;
  let sqSum = 0;
  for (let i = 0; i < n; i++) {
    const diff = actual[i] - predicted[i];
    absSum += Math.abs(diff);
    sqSum += diff * diff;
  }
  const mse = sqSum / n;
  return { mae: absSum / n, mse, rmse: Math.sqrt(mse), n };
}

/** Residuals (prediction error) for the error-distribution chart. */
export function residuals(
  actual: ArrayLike<number>,
  predicted: ArrayLike<number>,
): Float64Array {
  const n = Math.min(actual.length, predicted.length);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = predicted[i] - actual[i];
  return out;
}

export function correlation(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = Math.min(a.length, b.length);
  if (n === 0) return 0;
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < n; i++) {
    ma += a[i];
    mb += b[i];
  }
  ma /= n;
  mb /= n;
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < n; i++) {
    const xa = a[i] - ma;
    const xb = b[i] - mb;
    num += xa * xb;
    da += xa * xa;
    db += xb * xb;
  }
  const den = Math.sqrt(da * db);
  return den === 0 ? 0 : num / den;
}

/** Histogram used by the prediction-error distribution plot. */
export function histogram(
  values: ArrayLike<number>,
  bins: number,
): { binStart: number; binEnd: number; count: number }[] {
  if (values.length === 0) return [];
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (values[i] < min) min = values[i];
    if (values[i] > max) max = values[i];
  }
  if (min === max) {
    min -= 0.5;
    max += 0.5;
  }
  const width = (max - min) / bins;
  const counts = new Array<number>(bins).fill(0);
  for (let i = 0; i < values.length; i++) {
    const idx = Math.min(bins - 1, Math.floor((values[i] - min) / width));
    counts[idx]++;
  }
  return counts.map((count, i) => ({
    binStart: min + i * width,
    binEnd: min + (i + 1) * width,
    count,
  }));
}

export function mean(values: ArrayLike<number>): number {
  if (values.length === 0) return 0;
  let s = 0;
  for (let i = 0; i < values.length; i++) s += values[i];
  return s / values.length;
}

export function stdDev(values: ArrayLike<number>): number {
  if (values.length === 0) return 0;
  const m = mean(values);
  let acc = 0;
  for (let i = 0; i < values.length; i++) {
    const d = values[i] - m;
    acc += d * d;
  }
  return Math.sqrt(acc / values.length);
}

export function round(value: number, digits = 4): number {
  const f = Math.pow(10, digits);
  return Math.round(value * f) / f;
}
