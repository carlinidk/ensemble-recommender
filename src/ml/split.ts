import type { MergedRow, SplitStrategy } from "./types";
import { type Rng, trainTestSplit } from "./rng";

export type { SplitStrategy };

export interface SplitResult {
  strategy: SplitStrategy;
  train: Int32Array;
  test: Int32Array;
  trainTimestampRange: [number, number];
  testTimestampRange: [number, number];
  explanation: string;
}

/**
 * PAPER MODE uses a random split of all interactions.
 *
 * ENHANCED MODE offers a temporal split instead: every training interaction is
 * older than every test interaction, which is the honest simulation of
 * "given what the user had already rated, what should we recommend next?".
 * A random split lets the model see later ratings while predicting earlier
 * ones, which overstates accuracy for a live recommender.
 */
export function makeSplit(
  rows: MergedRow[],
  strategy: SplitStrategy,
  testSize: number,
  rng: Rng,
): SplitResult {
  if (strategy === "random") {
    const split = trainTestSplit(rows.length, testSize, rng);
    return {
      strategy,
      train: split.train,
      test: split.test,
      trainTimestampRange: timestampRange(rows, split.train),
      testTimestampRange: timestampRange(rows, split.test),
      explanation:
        "Random split over all interactions (the paper's evaluation style). Test interactions may be older than training interactions.",
    };
  }

  const order = Array.from({ length: rows.length }, (_, i) => i).sort(
    (a, b) => rows[a].timestamp - rows[b].timestamp || a - b,
  );
  const nTest = Math.max(1, Math.min(rows.length - 1, Math.round(rows.length * testSize)));
  const cut = rows.length - nTest;
  const test = Int32Array.from(order.slice(cut));
  const train = Int32Array.from(order.slice(0, cut));
  return {
    strategy,
    train,
    test,
    trainTimestampRange: timestampRange(rows, train),
    testTimestampRange: timestampRange(rows, test),
    explanation:
      "Temporal split: the training split holds strictly older interactions than the test split, mimicking live serving.",
  };
}

function timestampRange(rows: MergedRow[], idx: Int32Array): [number, number] {
  if (idx.length === 0) return [0, 0];
  let min = Infinity;
  let max = -Infinity;
  for (const i of idx) {
    const t = rows[i].timestamp;
    if (t < min) min = t;
    if (t > max) max = t;
  }
  return [min, max];
}
