import type { Matrix, MergedRow, Movie, TrainingMode } from "./types";

/**
 * Feature engineering + encoding.
 *
 * PAPER MODE features (exactly the four named by the paper; the target is not
 * a feature): `user_id`, `movie_id`, `year`.
 *
 * ENHANCED MODE adds supervised statistics and content features. Every statistic
 * is fitted on the TRAINING rows only, and for training rows themselves the
 * movie/user average is leave-one-out so a row never sees its own target.
 *
 * Categorical encoding: MovieLens IDs are high-cardinality but ordinal-free, so
 * they are label-encoded into dense integer codes (the paper's "categorical
 * variables must be appropriately numerically encoded"). Unseen categories map
 * to bucket 0, which the tree learners treat as their own bin.
 */

export interface MovieStat {
  count: number;
  sum: number;
  mean: number;
}

export interface FeatureSpace {
  mode: TrainingMode;
  names: string[];
  userCode: Map<number, number>;
  movieCode: Map<number, number>;
  yearFill: number;
  globalMean: number;
  movieStats: Map<number, MovieStat>;
  userStats: Map<number, MovieStat>;
  genreNames: string[];
  movieGenres: Map<number, Float64Array>;
  /** Feature-model name -> description, surfaced in the explainability UI. */
  descriptions: Record<string, string>;
}

const PAPER_FEATURES = ["user_id", "movie_id", "year"];
const ENHANCED_FEATURES = [
  "user_id",
  "movie_id",
  "year",
  "movie_popularity",
  "movie_mean_rating",
  "user_activity",
  "user_mean_rating",
];

export function featureNamesFor(
  mode: TrainingMode,
  genreCount: number,
  includeGenres = false,
): string[] {
  if (mode === "paper") return [...PAPER_FEATURES];
  return includeGenres
    ? [...ENHANCED_FEATURES, ...Array.from({ length: genreCount }, (_, i) => `genre_${i}`)]
    : [...ENHANCED_FEATURES];
}

export interface FitOptions {
  rows: MergedRow[];
  trainIdx: Int32Array;
  mode: TrainingMode;
  moviesById: Map<number, Movie>;
  genreNames: string[];
  /** Append the 19 genre flags in enhanced mode (opt-in: adds fitting cost). */
  includeGenres?: boolean;
}

/**
 * Fit encoders and supervised statistics on the training split.
 * Nothing here reads a test row, so test statistics cannot leak into
 * preprocessing.
 */
export function fitFeatureSpace(options: FitOptions): FeatureSpace {
  const { rows, trainIdx, mode, moviesById, genreNames } = options;

  const userCode = new Map<number, number>();
  const movieCode = new Map<number, number>();
  const movieStats = new Map<number, MovieStat>();
  const userStats = new Map<number, MovieStat>();
  const years: number[] = [];
  let ratingSum = 0;

  for (const i of trainIdx) {
    const row = rows[i];
    if (!userCode.has(row.userId)) userCode.set(row.userId, userCode.size + 1);
    if (!movieCode.has(row.movieId))
      movieCode.set(row.movieId, movieCode.size + 1);
    if (!row.yearMissing) years.push(row.year);

    const ms = movieStats.get(row.movieId);
    if (ms) {
      ms.count++;
      ms.sum += row.rating;
      ms.mean = ms.sum / ms.count;
    } else {
      movieStats.set(row.movieId, { count: 1, sum: row.rating, mean: row.rating });
    }

    const us = userStats.get(row.userId);
    if (us) {
      us.count++;
      us.sum += row.rating;
      us.mean = us.sum / us.count;
    } else {
      userStats.set(row.userId, { count: 1, sum: row.rating, mean: row.rating });
    }
    ratingSum += row.rating;
  }

  years.sort((a, b) => a - b);
  const yearFill = years.length > 0 ? years[Math.floor(years.length / 2)] : 1990;

  const movieGenres = new Map<number, Float64Array>();
  for (const [id] of movieCode) {
    const movie = moviesById.get(id);
    const vec = new Float64Array(genreNames.length);
    if (movie) {
      for (const g of movie.genres) {
        const gi = genreNames.indexOf(g);
        if (gi >= 0) vec[gi] = 1;
      }
    }
    movieGenres.set(id, vec);
  }

  const names = featureNamesFor(mode, genreNames.length, options.includeGenres ?? false);
  const descriptions: Record<string, string> = {
    user_id: "Label-encoded MovieLens user id (categorical).",
    movie_id: "Label-encoded MovieLens movie id (categorical).",
    year: "Movie release year; missing years imputed with the train-split median.",
    movie_popularity: "log1p(number of training ratings the movie received).",
    movie_mean_rating: "Mean training rating of the movie (leave-one-out on train rows).",
    user_activity: "log1p(number of training ratings by the user).",
    user_mean_rating: "Mean training rating of the user (leave-one-out on train rows).",
  };
  for (let i = 0; i < genreNames.length; i++) {
    descriptions[`genre_${i}`] = `Movie genre flag (${genreNames[i]}).`;
  }

  return {
    mode,
    names,
    userCode,
    movieCode,
    yearFill,
    globalMean: trainIdx.length > 0 ? ratingSum / trainIdx.length : 3.5,
    movieStats,
    userStats,
    genreNames,
    movieGenres,
    descriptions,
  };
}

export interface EncodeOptions {
  /** Build leave-one-out statistics — used for the training rows themselves. */
  leaveOneOut?: boolean;
}

/** Encode a subset of merged rows into a dense matrix using a fitted space. */
export function encodeRows(
  rows: MergedRow[],
  indices: Int32Array,
  space: FeatureSpace,
  options: EncodeOptions = {},
): Matrix {
  const d = space.names.length;
  const n = indices.length;
  const data = new Float64Array(n * d);
  const genreOffset =
    space.mode === "paper" ? 0 : space.names.length - space.genreNames.length;
  const hasGenres = space.mode === "enhanced" && space.names.includes("genre_0");

  for (let r = 0; r < n; r++) {
    const row = rows[indices[r]];
    const base = r * d;

    data[base] = space.userCode.get(row.userId) ?? 0;
    data[base + 1] = space.movieCode.get(row.movieId) ?? 0;
    data[base + 2] = row.yearMissing ? space.yearFill : row.year;

    if (space.mode === "enhanced") {
      const ms = space.movieStats.get(row.movieId);
      const us = space.userStats.get(row.userId);
      const movieCount = ms?.count ?? 0;
      const userCount = us?.count ?? 0;

      let movieMean = space.globalMean;
      if (movieCount > 0) {
        if (options.leaveOneOut && movieCount > 1) {
          movieMean = (ms!.sum - row.rating) / (movieCount - 1);
        } else {
          movieMean = ms!.mean;
        }
      }

      let userMean = space.globalMean;
      if (userCount > 0) {
        if (options.leaveOneOut && userCount > 1) {
          userMean = (us!.sum - row.rating) / (userCount - 1);
        } else {
          userMean = us!.mean;
        }
      }

      data[base + 3] = Math.log1p(movieCount);
      data[base + 4] = movieMean;
      data[base + 5] = Math.log1p(userCount);
      data[base + 6] = userMean;

      if (hasGenres) {
        const vec = space.movieGenres.get(row.movieId);
        if (vec) {
          for (let g = 0; g < vec.length; g++) {
            data[base + genreOffset + g] = vec[g];
          }
        }
      }
    }
  }

  return { n, d, names: space.names, data };
}

/** Feature row for a single (user, movie) pair — used by the serving path. */
export function encodeCandidates(
  candidates: { userId: number; movieId: number; year: number | null }[],
  space: FeatureSpace,
): Matrix {
  const d = space.names.length;
  const n = candidates.length;
  const data = new Float64Array(n * d);
  const genreOffset = space.mode === "paper" ? 0 : d - space.genreNames.length;
  const hasGenres = space.mode === "enhanced" && space.names.includes("genre_0");

  for (let r = 0; r < n; r++) {
    const c = candidates[r];
    const base = r * d;
    data[base] = space.userCode.get(c.userId) ?? 0;
    data[base + 1] = space.movieCode.get(c.movieId) ?? 0;
    data[base + 2] = c.year === null ? space.yearFill : c.year;

    if (space.mode === "enhanced") {
      const ms = space.movieStats.get(c.movieId);
      const us = space.userStats.get(c.userId);
      data[base + 3] = Math.log1p(ms?.count ?? 0);
      data[base + 4] = ms?.mean ?? space.globalMean;
      data[base + 5] = Math.log1p(us?.count ?? 0);
      data[base + 6] = us?.mean ?? space.globalMean;
      if (hasGenres) {
        const vec = space.movieGenres.get(c.movieId);
        if (vec) for (let g = 0; g < vec.length; g++) data[base + genreOffset + g] = vec[g];
      }
    }
  }

  return { n, d, names: space.names, data };
}
