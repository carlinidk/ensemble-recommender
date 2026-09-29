import type { FeatureImportance, Matrix, Regressor } from "../types";

/**
 * Linear Regression — the paper's meta learner (and one of its standalone
 * baselines).
 *
 * Plain OLS fitted with the normal equations. Columns are standardized
 * internally purely for numerical conditioning (the raw columns span 0-1682);
 * an affine rescaling does not change the fitted predictions. Coefficients are
 * reported in standardized space and converted back for feature attribution.
 */
export class LinearRegression implements Regressor {
  readonly name = "linear_regression";
  private means = new Float64Array(0);
  private scales = new Float64Array(0);
  private weights = new Float64Array(0);
  private intercept = 0;
  private featureNames: string[] = [];
  private fitted = false;

  fit(X: Matrix, y: Float64Array): void {
    const { n, d } = X;
    this.featureNames = X.names;
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
      const scale = Math.sqrt(variance / n) || 1;
      this.means[j] = mean;
      this.scales[j] = scale;
    }

    const p = d + 1;
    const xtx = new Float64Array(p * p);
    const xty = new Float64Array(p);
    for (let i = 0; i < n; i++) {
      const row = i * d;
      for (let a = 0; a < d; a++) {
        const va = (X.data[row + a] - this.means[a]) / this.scales[a];
        xty[a] += va * y[i];
        for (let b = a; b < d; b++) {
          const vb = (X.data[row + b] - this.means[b]) / this.scales[b];
          xtx[a * p + b] += va * vb;
        }
      }
      xty[d] += y[i];
      xtx[d * p + d] += 1;
      for (let a = 0; a < d; a++) xtx[a * p + d] += (X.data[row + a] - this.means[a]) / this.scales[a];
    }
    for (let a = 0; a < d; a++) {
      for (let b = 0; b < a; b++) xtx[a * p + b] = xtx[b * p + a];
    }
    // Small ridge term keeps the solve stable without meaningfully shrinking.
    for (let a = 0; a < p; a++) xtx[a * p + a] += 1e-8;

    const solution = solveLinearSystem(xtx, xty, p);
    this.weights = solution.slice(0, d);
    this.intercept = solution[d];
    this.fitted = true;
  }

  predict(X: Matrix): Float64Array {
    if (!this.fitted) throw new Error("LinearRegression.predict called before fit");
    const out = new Float64Array(X.n);
    const d = X.d;
    for (let i = 0; i < X.n; i++) {
      let acc = this.intercept;
      for (let j = 0; j < d; j++) {
        acc += this.weights[j] * ((X.data[i * d + j] - this.means[j]) / this.scales[j]);
      }
      out[i] = acc;
    }
    return out;
  }

  importance(): FeatureImportance[] {
    return this.featureNames.map((name, j) => ({
      name,
      // Standardized coefficient = effect of a one-sigma change in the feature.
      value: Math.abs(this.weights[j]),
    }));
  }

  /** Meta-learner weights on raw (unstandardized) base predictions. */
  metaWeights(): { name: string; weight: number; direction: number }[] {
    return this.featureNames.map((name, j) => ({
      name,
      weight: this.weights[j] / this.scales[j],
      direction: Math.sign(this.weights[j]),
    }));
  }

  describe(): Record<string, number | string | boolean> {
    return { features: this.featureNames.length, intercept: Number(this.intercept.toFixed(4)) };
  }
}

/** Gaussian elimination with partial pivoting. */
export function solveLinearSystem(
  a: Float64Array,
  b: Float64Array,
  size: number,
): Float64Array {
  const m = Float64Array.from(a);
  const rhs = Float64Array.from(b);
  for (let col = 0; col < size; col++) {
    let pivot = col;
    let best = Math.abs(m[col * size + col]);
    for (let row = col + 1; row < size; row++) {
      const v = Math.abs(m[row * size + col]);
      if (v > best) {
        best = v;
        pivot = row;
      }
    }
    if (best < 1e-12) continue;
    if (pivot !== col) {
      for (let k = 0; k < size; k++) {
        const tmp = m[col * size + k];
        m[col * size + k] = m[pivot * size + k];
        m[pivot * size + k] = tmp;
      }
      const tb = rhs[col];
      rhs[col] = rhs[pivot];
      rhs[pivot] = tb;
    }
    const diag = m[col * size + col];
    for (let row = col + 1; row < size; row++) {
      const factor = m[row * size + col] / diag;
      if (factor === 0) continue;
      for (let k = col; k < size; k++) m[row * size + k] -= factor * m[col * size + k];
      rhs[row] -= factor * rhs[col];
    }
  }
  const x = new Float64Array(size);
  for (let row = size - 1; row >= 0; row--) {
    let acc = rhs[row];
    for (let k = row + 1; k < size; k++) acc -= m[row * size + k] * x[k];
    const diag = m[row * size + row];
    x[row] = Math.abs(diag) < 1e-12 ? 0 : acc / diag;
  }
  return x;
}
