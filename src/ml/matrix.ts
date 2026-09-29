import type { Matrix } from "./types";

/** Row subset of a dense matrix (copies the rows, so callers can mutate safely). */
export function subMatrix(X: Matrix, idx: Int32Array): Matrix {
  const d = X.d;
  const data = new Float64Array(idx.length * d);
  for (let r = 0; r < idx.length; r++) {
    const src = idx[r] * d;
    const dst = r * d;
    for (let j = 0; j < d; j++) data[dst + j] = X.data[src + j];
  }
  return { n: idx.length, d, names: X.names, data };
}

/** Column-major vectors -> row-major matrix (used to build meta-learner inputs). */
export function matrixFromColumns(
  columns: ArrayLike<number>[],
  names: string[],
): Matrix {
  const d = columns.length;
  const n = d > 0 ? columns[0].length : 0;
  const data = new Float64Array(n * d);
  for (let r = 0; r < n; r++) {
    for (let j = 0; j < d; j++) data[r * d + j] = columns[j][r];
  }
  return { n, d, names, data };
}

export function columnValues(X: Matrix, j: number): Float64Array {
  const out = new Float64Array(X.n);
  for (let i = 0; i < X.n; i++) out[i] = X.data[i * X.d + j];
  return out;
}

export function complementIndices(all: Int32Array, exclude: Int32Array): Int32Array {
  const excluded = new Set<number>(Array.from(exclude));
  const out = new Int32Array(all.length - excluded.size);
  let k = 0;
  for (const v of all) if (!excluded.has(v)) out[k++] = v;
  return out.slice(0, k);
}
