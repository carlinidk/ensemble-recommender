import { describe, expect, test } from "bun:test";
import { buildDataset, extractTitleYear, parseGenres } from "../src/ml/dataset";
import { encodeRows, fitFeatureSpace } from "../src/ml/features";
import { evaluate } from "../src/ml/metrics";
import { createRng, kFoldIndices, trainTestSplit } from "../src/ml/rng";
import { makeSplit } from "../src/ml/split";
import { PAPER_PROPOSED, PAPER_STACKING, PAPER_STANDALONE } from "../src/ml/reference";
import { PROPOSED_CONFIG_ID, stackingConfigurations } from "../src/ml/experiments";

const base = "public/data/ml-100k";
const files = {
  movies: await Bun.file(`${base}/u.item`).text(),
  ratings: await Bun.file(`${base}/u.data`).text(),
  users: await Bun.file(`${base}/u.user`).text(),
  genres: await Bun.file(`${base}/u.genre`).text(),
};

const dataset = buildDataset(files);

describe("dataset", () => {
  test("parses the 19 MovieLens genres in flag order", () => {
    const genres = parseGenres(files.genres);
    expect(genres).toHaveLength(19);
    expect(genres[0]).toBe("unknown");
    expect(genres[1]).toBe("Action");
  });

  test("extracts release years from titles", () => {
    expect(extractTitleYear("Toy Story (1995)")).toBe(1995);
    expect(extractTitleYear("Misérables, Les (1995)")).toBe(1995);
    expect(extractTitleYear("No Year Here")).toBeNull();
  });

  test("merges files into the full 100k interaction table", () => {
    expect(dataset.audit.rows).toBe(100000);
    expect(dataset.audit.users).toBe(943);
    expect(dataset.audit.movies).toBe(1682);
    expect(dataset.audit.duplicateUserMoviePairs).toBe(0);
    expect(dataset.audit.ratingsOutOfRange).toBe(0);
    expect(dataset.audit.droppedRows).toBe(0);
  });

  test("records missing movie years instead of silently imputing them", () => {
    expect(dataset.audit.missingMovieYear).toBeGreaterThan(0);
    const missing = dataset.rows.filter((row) => row.yearMissing);
    expect(missing.length).toBe(dataset.audit.missingMovieYear);
  });

  test("exposes popularity statistics used by cold-start ranking", () => {
    const toyStory = dataset.movieStats.get(1);
    expect(toyStory?.count).toBeGreaterThan(100);
    expect(toyStory!.mean).toBeGreaterThan(1);
    expect(toyStory!.mean).toBeLessThanOrEqual(5);
    expect(dataset.globalMean).toBeGreaterThan(3);
    expect(dataset.globalMean).toBeLessThan(4);
  });
});

describe("metrics", () => {
  test("computes MAE, MSE and RMSE from the definitions", () => {
    const actual = [1, 2, 3, 4];
    const predicted = [1, 2, 4, 2];
    const result = evaluate(actual, predicted);
    expect(result.mae).toBeCloseTo((0 + 0 + 1 + 2) / 4, 10);
    expect(result.mse).toBeCloseTo((0 + 0 + 1 + 4) / 4, 10);
    expect(result.rmse).toBeCloseTo(Math.sqrt(1.25), 10);
  });

  test("returns zeros for an empty comparison", () => {
    expect(evaluate([], [])).toEqual({ mae: 0, mse: 0, rmse: 0, n: 0 });
  });
});

describe("splitting", () => {
  test("random split is reproducible for a fixed seed and covers every row", () => {
    const a = trainTestSplit(1000, 0.2, createRng(42));
    const b = trainTestSplit(1000, 0.2, createRng(42));
    const c = trainTestSplit(1000, 0.2, createRng(43));
    expect(Array.from(a.test)).toEqual(Array.from(b.test));
    expect(Array.from(a.test)).not.toEqual(Array.from(c.test));
    expect(a.train.length + a.test.length).toBe(1000);
    expect(new Set([...a.train, ...a.test]).size).toBe(1000);
  });

  test("temporal split puts every training interaction before every test interaction", () => {
    const split = makeSplit(dataset.rows, "temporal", 0.2, createRng(42));
    const maxTrain = Math.max(...Array.from(split.train, (i) => dataset.rows[i].timestamp));
    const minTest = Math.min(...Array.from(split.test, (i) => dataset.rows[i].timestamp));
    expect(maxTrain).toBeLessThanOrEqual(minTest);
    expect(split.train.length + split.test.length).toBe(dataset.rows.length);
  });

  test("k-fold assignment is disjoint and complete", () => {
    const folds = kFoldIndices(997, 5, createRng(7));
    const seen = new Set<number>();
    for (const fold of folds) {
      for (const index of fold) {
        expect(seen.has(index)).toBe(false);
        seen.add(index);
      }
    }
    expect(seen.size).toBe(997);
  });
});

describe("features", () => {
  const split = trainTestSplit(dataset.rows.length, 0.2, createRng(42));
  const space = fitFeatureSpace({
    rows: dataset.rows,
    trainIdx: split.train,
    mode: "paper",
    moviesById: dataset.moviesById,
    genreNames: dataset.genreNames,
  });

  test("paper mode exposes exactly the paper's three inputs plus nothing else", () => {
    expect(space.names).toEqual(["user_id", "movie_id", "year"]);
  });

  test("year imputation uses the training median when the year is missing", () => {
    const missingRow = dataset.rows.findIndex((row) => row.yearMissing);
    const matrix = encodeRows(dataset.rows, Int32Array.from([missingRow]), space);
    expect(matrix.data[2]).toBe(space.yearFill);
  });

  test("enhanced mode adds supervised statistics without leaking the row's own target", () => {
    const enhanced = fitFeatureSpace({
      rows: dataset.rows,
      trainIdx: split.train,
      mode: "enhanced",
      moviesById: dataset.moviesById,
      genreNames: dataset.genreNames,
    });
    expect(enhanced.names).toContain("movie_mean_rating");

    const rowIndex = split.train[0];
    const row = dataset.rows[rowIndex];
    const matrix = encodeRows(dataset.rows, Int32Array.from([rowIndex]), enhanced, {
      leaveOneOut: true,
    });
    const stats = enhanced.movieStats.get(row.movieId)!;
    const movieMean = matrix.data[4];
    if (stats.count > 1) {
      // Leave-one-out means never equal the plain group mean when the row is
      // part of that group and its rating is not exactly the group mean.
      const plainMean = stats.mean;
      const looMean = (stats.sum - row.rating) / (stats.count - 1);
      expect(movieMean).toBeCloseTo(looMean, 10);
      if (Math.abs(looMean - plainMean) > 1e-9) {
        expect(movieMean).not.toBeCloseTo(plainMean, 6);
      }
    }
  });
});

describe("reference integrity", () => {
  test("published values are only ever reported as reference", () => {
    expect(PAPER_STANDALONE.xgboost.rmse).toBe(0.92);
    expect(PAPER_PROPOSED.rmse).toBe(0.9);
    expect(PAPER_STANDALONE.xgboost.mae).toBeNull();
  });

  test("the proposed configuration matches the paper's proposed stack", () => {
    const configs = stackingConfigurations();
    const proposed = configs.find((c) => c.id === PROPOSED_CONFIG_ID);
    expect(proposed).toBeDefined();
    expect(proposed!.baseNames).toEqual(PAPER_PROPOSED.baseLearners);
    expect(proposed!.metaName).toBe(PAPER_PROPOSED.metaLearner);
  });

  test("every paper stacking row is reproduced as an experiment", () => {
    const configs = stackingConfigurations();
    expect(configs).toHaveLength(PAPER_STACKING.length);
    for (const reference of PAPER_STACKING) {
      const match = configs.find((c) => c.id === reference.id);
      expect(match).toBeDefined();
      expect(match!.baseNames.sort()).toEqual([...reference.baseLearners].sort());
      expect(match!.metaName).toBe(reference.metaLearner);
    }
  });
});
