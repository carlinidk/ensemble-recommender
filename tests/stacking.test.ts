import { describe, expect, test } from "bun:test";
import { encodeRows, fitFeatureSpace } from "../src/ml/features";
import { evaluate, correlation } from "../src/ml/metrics";
import { createBaseModel } from "../src/ml/models";
import { buildTree, fitBinner, predictTree, transformBins } from "../src/ml/models/tree";
import { createRng, trainTestSplit } from "../src/ml/rng";
import { runStack, trainBaseLearners, type TrainedBase } from "../src/ml/stacking";
import type { Matrix } from "../src/ml/types";

const rng = createRng(11);
const n = 600;
const d = 3;

function subset(X: Matrix, idx: Int32Array): Matrix {
  const data = new Float64Array(idx.length * X.d);
  for (let r = 0; r < idx.length; r++) {
    for (let j = 0; j < X.d; j++) data[r * X.d + j] = X.data[idx[r] * X.d + j];
  }
  return { n: idx.length, d: X.d, names: X.names, data };
}

function meanOf(values: Float64Array): number {
  let total = 0;
  for (const value of values) total += value;
  return total / values.length;
}

function syntheticMatrix(): { X: Matrix; y: Float64Array } {
  const data = new Float64Array(n * d);
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const a = rng.next() * 10;
    const b = rng.next() * 10;
    const c = rng.next() * 10;
    data[i * d] = a;
    data[i * d + 1] = b;
    data[i * d + 2] = c;
    // Deterministic signal plus noise, so learners have something to fit.
    y[i] = 1 + 0.2 * a + 0.05 * b + rng.next() * 0.5;
  }
  return { X: { n, d, names: ["a", "b", "c"], data }, y };
}

describe("regression tree engine", () => {
  test("fits a step function and never predicts outside the observed target range", () => {
    const { X, y } = syntheticMatrix();
    const binner = fitBinner(X, 32);
    const bins = transformBins(X, binner);
    const indices = Int32Array.from({ length: X.n }, (_, i) => i);
    const tree = buildTree(
      bins,
      { kind: "cart", target: y },
      indices,
      {
        maxDepth: 4,
        minSamplesLeaf: 5,
        minChildWeight: 5,
        minSamplesSplit: 10,
        maxBins: 32,
        lambda: 0,
        gamma: 0,
        maxFeatures: 0,
      },
      createRng(1),
    );
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < X.n; i++) {
      const prediction = predictTree(tree.root, X, i);
      min = Math.min(min, prediction);
      max = Math.max(max, prediction);
    }
    expect(min).toBeGreaterThanOrEqual(Math.min(...y) - 1e-9);
    expect(max).toBeLessThanOrEqual(Math.max(...y) + 1e-9);
    expect(tree.nodes).toBeGreaterThan(1);
  });
});

describe("base learners", () => {
  test("features are recovered better than the mean baseline", () => {
    const { X, y } = syntheticMatrix();
    const split = trainTestSplit(n, 0.25, createRng(3));
    const train = subset(X, split.train);
    const holdout = subset(X, split.test);
    const yTrain = Float64Array.from(split.train, (i) => y[i]);
    const yHoldout = Float64Array.from(split.test, (i) => y[i]);

    const baseline = evaluate(yHoldout, new Float64Array(holdout.n).fill(meanOf(yTrain)));
    for (const name of ["random_forest", "gradient_boosting", "xgboost"] as const) {
      const model = createBaseModel(name, { search: "quick", seed: 5 });
      model.fit(train, yTrain);
      const metrics = evaluate(yHoldout, model.predict(holdout));
      expect(metrics.rmse).toBeLessThan(baseline.rmse);
    }
  });

  test("tree ensembles report normalised feature importance", () => {
    const { X, y } = syntheticMatrix();
    const model = createBaseModel("xgboost", { search: "quick", seed: 9 });
    model.fit(X, y);
    const importance = model.importance?.() ?? [];
    const total = importance.reduce((acc, item) => acc + item.value, 0);
    expect(total).toBeCloseTo(1, 5);
    expect(importance).toHaveLength(d);
  });
});

describe("leakage-safe stacking", () => {
  test("out-of-fold predictions are not in-sample predictions", () => {
    const { X, y } = syntheticMatrix();
    const shuffled = new Float64Array(n);
    for (let i = 0; i < n; i++) shuffled[i] = rng.next() * 5 + 1; // pure noise targets

    const bases = trainBaseLearners({
      rows: Array.from({ length: n }, (_, i) => ({
        userId: 1,
        movieId: i + 1,
        rating: shuffled[i],
        timestamp: i,
        year: 1990 + (i % 20),
        yearMissing: false,
        genreCount: 1,
      })),
      trainIdx: Int32Array.from({ length: n }, (_, i) => i),
      Xtrain: X,
      ytrain: shuffled,
      Xtest: X,
      ytest: shuffled,
      names: ["knn"],
      folds: 5,
      seed: 42,
      search: "quick",
      mode: "paper",
      moviesById: new Map(),
      genreNames: [],
    });

    const knn = bases[0];
    // A 1-nearest-neighbour model fitted on the full data would memorise these
    // targets exactly. Out-of-fold predictions come from other folds, so they
    // cannot: that is the property that keeps the meta learner honest.
    const inSampleCorrelation = 1;
    const oofCorrelation = Math.abs(correlation(Array.from(shuffled), Array.from(knn.oof)));
    expect(oofCorrelation).toBeLessThan(inSampleCorrelation - 0.2);
  });

  test("meta learner is trained on out-of-fold columns and scored on test predictions", () => {
    const { X, y } = syntheticMatrix();
    const bases: TrainedBase[] = ["knn", "xgboost"].map((name, index) => {
      const oof = Float64Array.from({ length: n }, (_, i) => y[i] + index * 0.1);
      const test = Float64Array.from({ length: n }, (_, i) => y[i] + index * 0.1);
      return {
        name: name as TrainedBase["name"],
        oof,
        test,
        model: createBaseModel(name as TrainedBase["name"], { search: "quick", seed: 1 }),
        standalone: { mae: 0, mse: 0, rmse: 0, n },
        oofMetrics: { mae: 0, mse: 0, rmse: 0, n },
        fitSeconds: 0,
        importance: [],
        params: {},
      };
    });

    const outcome = runStack({
      bases,
      ytrain: y,
      ytest: y,
      combo: { id: "T", baseNames: ["knn", "xgboost"], metaName: "linear_regression" },
      seed: 1,
      search: "quick",
    });

    expect(outcome.metaInputNames).toEqual(["knn_prediction", "xgboost_prediction"]);
    expect(outcome.metrics.rmse).toBeLessThan(0.001);
    expect(outcome.metaWeights?.[0].name).toBe("knn_prediction");
  });

  test("an unseen stack member is reported instead of silently ignored", () => {
    const { y } = syntheticMatrix();
    expect(() =>
      runStack({
        bases: [],
        ytrain: y,
        ytest: y,
        combo: { id: "X", baseNames: ["knn"], metaName: "linear_regression" },
        seed: 1,
        search: "quick",
      }),
    ).toThrow();
  });
});

describe("feature pipeline wiring", () => {
  test("encodeRows keeps column count and row count consistent", () => {
    const rows = Array.from({ length: 40 }, (_, i) => ({
      userId: 1,
      movieId: (i % 5) + 1,
      rating: 1 + (i % 5),
      timestamp: i,
      year: 1990 + (i % 10),
      yearMissing: false,
      genreCount: 1,
    }));
    const space = fitFeatureSpace({
      rows,
      trainIdx: Int32Array.from({ length: rows.length }, (_, i) => i),
      mode: "enhanced",
      moviesById: new Map(),
      genreNames: ["Action"],
    });
    const matrix = encodeRows(rows, Int32Array.from({ length: rows.length }, (_, i) => i), space);
    expect(matrix.n).toBe(rows.length);
    expect(matrix.d).toBe(space.names.length);
    expect(matrix.data.length).toBe(rows.length * space.names.length);
  });

  test("selected indices preserve row identity", () => {
    const { X } = syntheticMatrix();
    const sample = subset(X, Int32Array.from([2, 5, 7]));
    expect(sample.data[0]).toBe(X.data[2 * X.d]);
    expect(sample.data[3]).toBe(X.data[5 * X.d]);
    expect(sample.n).toBe(3);
  });
});
