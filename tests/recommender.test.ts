import { describe, expect, test } from "bun:test";
import { recommend, type ScoredBatch } from "../src/ml/recommender/engine";
import { buildTasteProfile } from "../src/ml/recommender/profile";
import { intraListSimilarity, selectWithMmr, genreVector } from "../src/ml/recommender/diversify";
import type { Movie, Rating } from "../src/ml/types";

const GENRES = ["unknown", "Action", "Comedy", "Drama", "Sci-Fi"];

// 1-9 are rated by the app user and by the matched persona, so they can never
// appear in a recommendation. 10-18 are the candidate pool.
const movies: Movie[] = [
  movie(1, "Alpha Action", 1995, ["Action"]),
  movie(2, "Action Reloaded", 1996, ["Action"]),
  movie(3, "Action Forever", 1997, ["Action"]),
  movie(4, "Comedy Hour", 1994, ["Comedy"]),
  movie(5, "Comedy Night", 1998, ["Comedy"]),
  movie(6, "Quiet Drama", 1992, ["Drama"]),
  movie(7, "Space Opera", 1999, ["Sci-Fi"]),
  movie(8, "Space Sequel", 2001, ["Sci-Fi"]),
  movie(9, "Space Finale", 2003, ["Sci-Fi"]),
  movie(10, "Bullet Rain", 1988, ["Action"]),
  movie(11, "Bullet Rain II", 1990, ["Action"]),
  movie(12, "Bullet Rain III", 1992, ["Action"]),
  movie(13, "Laugh Track", 1993, ["Comedy"]),
  movie(14, "Laugh Track II", 1997, ["Comedy"]),
  movie(15, "Long Winter", 1979, ["Drama"]),
  movie(16, "Long Winter II", 1984, ["Drama"]),
  movie(17, "Orbit Rising", 2005, ["Sci-Fi"]),
  movie(18, "Orbit Falling", 2009, ["Sci-Fi"]),
];

const moviesById = new Map(movies.map((m) => [m.id, m]));

function movie(id: number, title: string, year: number, genres: string[]): Movie {
  return { id, title, year, genres, imdbUrl: null };
}

function rating(userId: number, movieId: number, value: number): Rating {
  return { userId, movieId, rating: value, timestamp: movieId };
}

/** Persona 1 agrees with the app user on their whole overlap. */
const ratingsByMovie = new Map<number, Rating[]>([
  [1, [rating(1, 1, 5), rating(2, 1, 1)]],
  [2, [rating(1, 2, 5), rating(2, 2, 1)]],
  [3, [rating(1, 3, 4), rating(2, 3, 1)]],
  [4, [rating(1, 4, 4), rating(2, 4, 5)]],
  [5, [rating(1, 5, 4), rating(2, 5, 5)]],
  [6, [rating(1, 6, 3), rating(2, 6, 4)]],
  [7, [rating(1, 7, 5), rating(2, 7, 2)]],
  [8, [rating(1, 8, 5), rating(2, 8, 2)]],
  [9, [rating(1, 9, 4), rating(2, 9, 2)]],
]);

const ratingsByUser = new Map<number, Rating[]>([
  [1, [1, 2, 3, 4, 5, 6, 7, 8, 9].map((id) => rating(1, id, 4))],
  [2, [1, 2, 3, 4, 5, 6, 7, 8, 9].map((id) => rating(2, id, 3))],
]);

const movieStats = new Map(
  movies.map((m, index) => [m.id, { count: 500 - index * 40, mean: 4.2 - index * 0.1 }]),
);

const userRatings = [
  { movieId: 1, rating: 5 },
  { movieId: 2, rating: 5 },
  { movieId: 3, rating: 4 },
  { movieId: 4, rating: 4 },
  { movieId: 5, rating: 4 },
  { movieId: 6, rating: 3 },
  { movieId: 7, rating: 5 },
  { movieId: 8, rating: 5 },
];

const profile = buildTasteProfile({
  ratings: userRatings,
  moviesById,
  ratingsByMovie,
  globalMean: 3.5,
  minSharedMovies: 4,
});

const baseConfig = {
  topK: 3,
  mode: "accuracy" as const,
  diversityLambda: 0.65,
  affinityWeight: 0,
  poolMultiplier: 10,
};

/** Scorer that returns a fixed prediction, so ranking is fully controlled. */
function fixedScorer(value: number): (userId: number, ids: number[]) => ScoredBatch {
  return (userId, ids) => ({
    perModel: { knn: ids.map(() => value - 0.2), xgboost: ids.map(() => value + 0.2) },
    final: ids.map(() => value),
    metaRecipe: "test",
  });
}

describe("taste profile", () => {
  test("matches the MovieLens user with the closest rating pattern", () => {
    expect(profile.persona?.userId).toBe(1);
    expect(profile.persona?.sharedMovies).toBe(8);
    expect(profile.persona!.similarity).toBeGreaterThan(0.9);
  });

  test("computes genre affinity against the user's own mean", () => {
    const action = profile.genreAffinity.find((g) => g.genre === "Action");
    expect(action).toBeDefined();
    expect(action!.count).toBe(3);
    expect(action!.meanRating).toBeCloseTo((5 + 5 + 4) / 3, 6);
  });

  test("refuses a persona match below the shared-film threshold", () => {
    const thin = buildTasteProfile({
      ratings: userRatings.slice(0, 2),
      moviesById,
      ratingsByMovie,
      globalMean: 3.5,
      minSharedMovies: 8,
    });
    expect(thin.persona).toBeNull();
  });
});

describe("recommendation engine", () => {
  test("never recommends a movie the user or the persona already rated", () => {
    const output = recommend({
      ratings: userRatings,
      profile,
      moviesById,
      movieStats,
      ratingsByUser,
      genreNames: GENRES,
      globalMean: 3.5,
      config: baseConfig,
      score: fixedScorer(4.5),
    });
    const rated = new Set(userRatings.map((r) => r.movieId));
    const personaRated = new Set((ratingsByUser.get(1) ?? []).map((r) => r.movieId));
    for (const rec of output.recommendations) {
      expect(rated.has(rec.movieId)).toBe(false);
      expect(personaRated.has(rec.movieId)).toBe(false);
    }
    expect(output.diagnostics.excludedAlreadyRated).toBe(8);
    expect(output.diagnostics.excludedPersonaHistory).toBe(1);
  });

  test("labels model-derived recommendations with the base predictions behind them", () => {
    const output = recommend({
      ratings: userRatings,
      profile,
      moviesById,
      movieStats,
      ratingsByUser,
      genreNames: GENRES,
      globalMean: 3.5,
      config: baseConfig,
      score: fixedScorer(4.25),
    });
    expect(output.source).toBe("stacking_model");
    const first = output.recommendations[0];
    expect(output.recommendations).toHaveLength(3);
    expect(first.predictedRating).toBeCloseTo(4.25, 6);
    expect(first.basePredictions.map((b) => b.name).sort()).toEqual(["knn", "xgboost"]);
    expect(first.explanation.modelDerived).toContain("4.25");
    // Explanations are assembled only from data that exists.
    expect(first.explanation.evidence.join(" ")).toContain("MovieLens 100K");
    expect(first.explanation.evidence.join(" ")).toContain("cosine");
  });

  test("diversity mode lowers intra-list similarity relative to accuracy mode", () => {
    // The three most relevant candidates share one genre, which is exactly the
    // "ten nearly identical movies" case the MMR pass exists to break up.
    const ranked = [10, 11, 12, 13, 14, 15, 16, 17, 18].map((id, index) => {
      const entry = moviesById.get(id)!;
      return {
        id,
        relevance: 4.9 - index * 0.1,
        genreVector: genreVector(entry.genres, GENRES),
        year: entry.year,
      };
    });

    const accuracy = selectWithMmr(ranked, 3, 1);
    const diverse = selectWithMmr(ranked, 3, 0.2);
    expect(accuracy.ids).toEqual([10, 11, 12]);
    const byId = new Map(ranked.map((r) => [r.id, r]));
    const accuracyItems = accuracy.ids.map((id) => byId.get(id)!);
    const diverseItems = diverse.ids.map((id) => byId.get(id)!);

    expect(intraListSimilarity(diverseItems)).toBeLessThan(intraListSimilarity(accuracyItems));
    expect(new Set(diverseItems.map((i) => i.id)).size).toBe(3);
  });
});

describe("cold start", () => {
  test("a user with no ratings gets an explicitly labelled popularity fallback", () => {
    const empty = buildTasteProfile({
      ratings: [],
      moviesById,
      ratingsByMovie,
      globalMean: 3.5,
    });
    const output = recommend({
      ratings: [],
      profile: empty,
      moviesById,
      movieStats,
      ratingsByUser,
      genreNames: GENRES,
      globalMean: 3.5,
      config: baseConfig,
      score: fixedScorer(5),
    });
    expect(output.source).toBe("cold_start_fallback");
    expect(output.recommendations).toHaveLength(3);
    for (const rec of output.recommendations) {
      expect(rec.predictedRating).toBeNull();
      expect(rec.explanation.caveat).toContain("not a model prediction");
    }
    // Popularity ordering: the most-rated film comes first.
    expect(output.recommendations[0].movieId).toBe(1);
  });

  test("ratings without a persona match fall back and say why", () => {
    const noPersona = { ...profile, persona: null };
    const output = recommend({
      ratings: userRatings,
      profile: noPersona,
      moviesById,
      movieStats,
      ratingsByUser,
      genreNames: GENRES,
      globalMean: 3.5,
      config: baseConfig,
      score: fixedScorer(5),
    });
    expect(output.source).toBe("no_persona_fallback");
    expect(output.recommendations[0].explanation.caveat).toContain("no MovieLens user");
  });
});
