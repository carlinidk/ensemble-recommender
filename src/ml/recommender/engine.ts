import { stdDev } from "../metrics";
import type { Movie, Rating } from "../types";
import {
  genreCoverage,
  genreVector,
  intraListSimilarity,
  selectWithMmr,
  type RankableItem,
} from "./diversify";
import { affinityBonus, type TasteProfile, type UserRating } from "./profile";

/**
 * Recommendation engine (Phase 8).
 *
 * Pipeline for a signed-in user:
 *   1. take the user's own ratings and exclude every movie they already rated;
 *   2. map the user onto a MovieLens persona (`profile.persona`) and exclude
 *      that persona's history too, so we never score memorised interactions;
 *   3. build candidate features for the remaining catalogue;
 *   4. score candidates with the paper's stacking pipeline
 *      (KNN + XGBoost + Gradient Boosting -> Linear Regression);
 *   5. optionally blend the user's own content affinity (ENHANCED MODE);
 *   6. rank by relevance, or re-rank with MMR in diversity mode;
 *   7. return Top-K with prediction, support, confidence and evidence.
 *
 * When the model cannot legitimately be used (no ratings at all, or no persona
 * match), the engine switches to an explicitly-labelled fallback instead of
 * dressing a heuristic up as a model prediction.
 */

export type RecommendationSource =
  | "stacking_model"
  | "cold_start_fallback"
  | "no_persona_fallback";

export interface RecommendConfig {
  topK: number;
  mode: "accuracy" | "diversity";
  /** λ for MMR: 1 = pure relevance, 0 = pure diversity. */
  diversityLambda: number;
  /** How much the user's own content affinity can move a candidate (0-1). */
  affinityWeight: number;
  poolMultiplier: number;
}

export const DEFAULT_RECOMMEND_CONFIG: RecommendConfig = {
  topK: 10,
  mode: "accuracy",
  diversityLambda: 0.65,
  affinityWeight: 0.3,
  poolMultiplier: 4,
};

export interface ScoredBatch {
  perModel: Record<string, number[]>;
  final: number[];
  metaRecipe: string;
}

export interface MovieSupport {
  datasetRatings: number;
  datasetMean: number;
  sparse: boolean;
}

export interface Recommendation {
  movieId: number;
  title: string;
  year: number | null;
  genres: string[];
  predictedRating: number | null;
  basePredictions: { name: string; value: number }[];
  metaRecipe: string | null;
  affinity: number;
  relevance: number;
  diversityPenalty: number | null;
  rank: number;
  support: MovieSupport;
  confidence: { spread: number; label: "low" | "moderate" | "high"; note: string };
  explanation: { modelDerived: string | null; evidence: string[]; caveat: string | null };
  source: RecommendationSource;
}

export interface RecommendInput {
  ratings: UserRating[];
  profile: TasteProfile;
  moviesById: Map<number, Movie>;
  movieStats: Map<number, { count: number; mean: number }>;
  /** Index used to exclude the persona's own rated movies. */
  ratingsByUser: Map<number, Rating[]>;
  genreNames: string[];
  globalMean: number;
  config: RecommendConfig;
  /** Stacking-pipeline scorer; receives a MovieLens user id and candidate ids. */
  score: (movielensUserId: number, movieIds: number[]) => ScoredBatch;
}

export interface RecommendDiagnostics {
  candidatesConsidered: number;
  excludedAlreadyRated: number;
  excludedPersonaHistory: number;
  personaUserId: number | null;
  personaSimilarity: number | null;
  personaSharedMovies: number | null;
  meanPredictedRating: number | null;
  meanConfidenceSpread: number | null;
  distinctGenres: number;
  genreCoverage: { genre: string; count: number }[];
  intraListSimilarity: number;
  rationale: string;
}

export interface RecommendOutput {
  source: RecommendationSource;
  sourceLabel: string;
  recommendations: Recommendation[];
  diagnostics: RecommendDiagnostics;
}

const MIN_RATINGS_FOR_MODEL_SUPPORT = 10;

export function recommend(input: RecommendInput): RecommendOutput {
  const {
    ratings,
    profile,
    moviesById,
    movieStats,
    ratingsByUser,
    genreNames,
    globalMean,
    config,
    score,
  } = input;

  const ratedIds = new Set(ratings.map((r) => r.movieId));
  const persona = profile.persona;
  const personaRatedIds = new Set(
    persona ? (ratingsByUser.get(persona.userId) ?? []).map((r) => r.movieId) : [],
  );

  const candidates: Movie[] = [];
  let excludedPersonaHistory = 0;
  for (const movie of moviesById.values()) {
    if (ratedIds.has(movie.id)) continue;
    if (personaRatedIds.has(movie.id)) {
      excludedPersonaHistory++;
      continue;
    }
    candidates.push(movie);
  }

  const coldStart = ratings.length === 0;
  const useModel = !coldStart && persona !== null;

  const poolSize = Math.max(
    config.topK,
    Math.min(candidates.length, config.topK * Math.max(1, config.poolMultiplier)),
  );

  const baseContext = {
    excludedAlreadyRated: ratedIds.size,
    excludedPersonaHistory,
    genreNames,
  };

  if (useModel) {
    const shortlist = shortlistByPopularity(candidates, movieStats, globalMean, poolSize);
    const ids = shortlist.map((m) => m.id);
    const batch = score(persona.userId, ids);
    const rankable: RankableItem[] = shortlist.map((movie, i) => {
      const bonus = affinityBonus(movie, profile).bonus;
      const relevance = clamp(clamp(batch.final[i], 1, 5) + config.affinityWeight * bonus, 1, 5);
      return {
        id: movie.id,
        relevance,
        genreVector: genreVector(movie.genres, genreNames),
        year: movie.year,
      };
    });

    const selection = selectRankable(rankable, config);
    const relevanceById = new Map(rankable.map((r) => [r.id, r.relevance]));
    const positionById = new Map(shortlist.map((m, i) => [m.id, i]));

    const list: Recommendation[] = selection.ids.map((id, index) => {
      const movie = moviesById.get(id)!;
      const position = positionById.get(id) ?? 0;
      const affinity = affinityBonus(movie, profile);
      const basePredictions = Object.entries(batch.perModel).map(([name, values]) => ({
        name,
        value: round3(values[position]),
      }));
      const spread = stdDev(basePredictions.map((p) => p.value));
      return buildRecommendation({
        movie,
        movieStats,
        globalMean,
        predicted: batch.final[position],
        basePredictions,
        metaRecipe: batch.metaRecipe,
        affinity: affinity.bonus,
        affinityDrivers: affinity.drivers,
        relevance: relevanceById.get(id) ?? 0,
        confidence: spread,
        rank: index + 1,
        profile,
        persona,
        diversityPenalty: config.mode === "diversity" ? (selection.penalties.get(id) ?? null) : null,
        source: "stacking_model",
        caveat: null,
      });
    });

    return summarize(list, {
      ...baseContext,
      candidatesConsidered: shortlist.length,
      persona,
      rationale:
        "Candidates outside the user's and the persona's histories were scored by the stacking ensemble (KNN + XGBoost + Gradient Boosting → Linear Regression). Affinity blending and MMR re-ranking are enhancements added by this project, not part of the paper.",
    });
  }

  // --- documented fallbacks -----------------------------------------------
  const shortlist = coldStart
    ? shortlistByPopularity(candidates, movieStats, globalMean, poolSize)
    : candidates
        .map((movie) => {
          const stats = movieStats.get(movie.id) ?? { count: 0, mean: globalMean };
          return { movie, stats };
        })
        .sort((a, b) => b.stats.count - a.stats.count)
        .slice(0, Math.max(poolSize, 400))
        .map((s) => s.movie);

  const maxCount = Math.max(1, ...shortlist.map((m) => movieStats.get(m.id)?.count ?? 0));
  const rankable: RankableItem[] = shortlist.map((movie) => {
    const stats = movieStats.get(movie.id) ?? { count: 0, mean: globalMean };
    const quality = bayesianMean(stats.count, stats.mean, globalMean, 25) / 5;
    const popularity = Math.log1p(stats.count) / Math.log1p(maxCount);
    const affinityNorm = clamp(0.5 + affinityBonus(movie, profile).bonus / 2, 0, 1);
    return {
      id: movie.id,
      relevance: 5 * (0.5 * quality + 0.3 * popularity + 0.2 * affinityNorm),
      genreVector: genreVector(movie.genres, genreNames),
      year: movie.year,
    };
  });

  const selection = selectRankable(rankable, config);
  const relevanceById = new Map(rankable.map((r) => [r.id, r.relevance]));
  const source: RecommendationSource = coldStart ? "cold_start_fallback" : "no_persona_fallback";

  const list: Recommendation[] = selection.ids.map((id, index) => {
    const movie = moviesById.get(id)!;
    const affinity = affinityBonus(movie, profile);
    return buildRecommendation({
      movie,
      movieStats,
      globalMean,
      predicted: null,
      basePredictions: [],
      metaRecipe: null,
      affinity: affinity.bonus,
      affinityDrivers: affinity.drivers,
      relevance: relevanceById.get(id) ?? 0,
      confidence: 0,
      rank: index + 1,
      profile,
      persona,
      diversityPenalty: config.mode === "diversity" ? (selection.penalties.get(id) ?? null) : null,
      source,
      caveat: coldStart
        ? "Fallback, not a model prediction: this account has no ratings yet, so the list comes from MovieLens popularity and Bayesian-average rating."
        : "Fallback, not a model prediction: the model needs a MovieLens user id, and no MovieLens user shared enough co-rated movies with this account.",
    });
  });

  return summarize(list, {
    ...baseContext,
    candidatesConsidered: shortlist.length,
    persona: null,
    rationale: coldStart
      ? "Cold start: no ratings for this user, so ranking is popularity + Bayesian average rating + genre popularity, labelled as a fallback throughout."
      : "No persona match: the paper's models consume a MovieLens user id, and no MovieLens user shared enough co-rated movies with this account.",
  });
}

function selectRankable(
  rankable: RankableItem[],
  config: RecommendConfig,
): { ids: number[]; penalties: Map<number, number> } {
  if (config.mode === "diversity") {
    return selectWithMmr(rankable, config.topK, config.diversityLambda);
  }
  return {
    ids: [...rankable]
      .sort((a, b) => b.relevance - a.relevance || a.id - b.id)
      .slice(0, config.topK)
      .map((r) => r.id),
    penalties: new Map<number, number>(),
  };
}

interface BuildArgs {
  movie: Movie;
  movieStats: Map<number, { count: number; mean: number }>;
  globalMean: number;
  predicted: number | null;
  basePredictions: { name: string; value: number }[];
  metaRecipe: string | null;
  affinity: number;
  affinityDrivers: { genre: string; weight: number }[];
  relevance: number;
  confidence: number;
  rank: number;
  profile: TasteProfile;
  persona: TasteProfile["persona"];
  diversityPenalty: number | null;
  source: RecommendationSource;
  caveat: string | null;
}

function buildRecommendation(args: BuildArgs): Recommendation {
  const stats = args.movieStats.get(args.movie.id) ?? { count: 0, mean: args.globalMean };
  const sparse = stats.count < MIN_RATINGS_FOR_MODEL_SUPPORT;

  const modelDerived =
    args.predicted === null
      ? null
      : `Stacking ensemble predicted ${args.predicted.toFixed(2)} — ${args.basePredictions
          .map((p) => `${LABELS[p.name] ?? p.name} ${p.value.toFixed(2)}`)
          .join(", ")} → Linear Regression meta-learner.`;

  const evidence: string[] = [];
  if (args.persona) {
    evidence.push(
      `Scored for MovieLens user ${args.persona.userId}, the closest match to this account (cosine ${args.persona.similarity.toFixed(2)} over ${args.persona.sharedMovies} co-rated movies).`,
    );
  }
  for (const driver of args.affinityDrivers.slice(0, 2)) {
    const entry = args.profile.genreAffinity.find((g) => g.genre === driver.genre);
    if (!entry) continue;
    evidence.push(
      `This account rates ${entry.genre} ${Math.abs(entry.lift).toFixed(2)} ${entry.lift >= 0 ? "above" : "below"} its own average across ${entry.count} rated movies.`,
    );
  }
  evidence.push(`MovieLens 100K: ${stats.count} ratings, mean ${stats.mean.toFixed(2)}.`);
  if (args.affinity !== 0 && args.source === "stacking_model") {
    evidence.push(
      `Content affinity adjustment of ${args.affinity >= 0 ? "+" : ""}${args.affinity.toFixed(2)} rating points from this account's own genre history.`,
    );
  }

  const caveat =
    args.caveat ??
    (sparse
      ? `Only ${stats.count} MovieLens ratings back this movie, so the prediction rests on little data.`
      : null);

  return {
    movieId: args.movie.id,
    title: args.movie.title,
    year: args.movie.year,
    genres: args.movie.genres,
    predictedRating: args.predicted === null ? null : clamp(args.predicted, 1, 5),
    basePredictions: args.basePredictions,
    metaRecipe: args.metaRecipe,
    affinity: round3(args.affinity),
    relevance: round3(args.relevance),
    diversityPenalty: args.diversityPenalty,
    rank: args.rank,
    support: { datasetRatings: stats.count, datasetMean: round3(stats.mean), sparse },
    confidence: {
      spread: round3(args.confidence),
      label: args.confidence < 0.2 ? "low" : args.confidence < 0.4 ? "moderate" : "high",
      note:
        args.source === "stacking_model"
          ? "Spread across the three base learners — a disagreement measure, not a calibrated probability."
          : "Not applicable to fallback ranking.",
    },
    explanation: { modelDerived, evidence, caveat },
    source: args.source,
  };
}

function summarize(
  list: Recommendation[],
  context: {
    candidatesConsidered: number;
    excludedAlreadyRated: number;
    excludedPersonaHistory: number;
    persona: TasteProfile["persona"];
    genreNames: string[];
    rationale: string;
  },
): RecommendOutput {
  const predictions = list.map((r) => r.predictedRating).filter((v): v is number => v !== null);
  const spreads = list.filter((r) => r.source === "stacking_model").map((r) => r.confidence.spread);
  const coverage = genreCoverage(
    list.map((r) => ({ genreVector: genreVector(r.genres, context.genreNames) })),
    context.genreNames,
  );
  const source: RecommendationSource = list.length > 0 ? list[0].source : "cold_start_fallback";

  return {
    source,
    sourceLabel:
      source === "stacking_model"
        ? "Stacking ensemble (paper pipeline)"
        : source === "cold_start_fallback"
          ? "Cold-start fallback — not a model prediction"
          : "No-persona fallback — not a model prediction",
    recommendations: list,
    diagnostics: {
      candidatesConsidered: context.candidatesConsidered,
      excludedAlreadyRated: context.excludedAlreadyRated,
      excludedPersonaHistory: context.excludedPersonaHistory,
      personaUserId: context.persona?.userId ?? null,
      personaSimilarity: context.persona?.similarity ?? null,
      personaSharedMovies: context.persona?.sharedMovies ?? null,
      meanPredictedRating:
        predictions.length > 0 ? predictions.reduce((a, b) => a + b, 0) / predictions.length : null,
      meanConfidenceSpread:
        spreads.length > 0 ? spreads.reduce((a, b) => a + b, 0) / spreads.length : null,
      distinctGenres: coverage.length,
      genreCoverage: coverage,
      intraListSimilarity: intraListSimilarity(
        list.map((r) => ({
          id: r.movieId,
          relevance: r.relevance,
          genreVector: genreVector(r.genres, context.genreNames),
          year: r.year,
        })),
      ),
      rationale: context.rationale,
    },
  };
}

function bayesianMean(count: number, mean: number, prior: number, priorStrength: number): number {
  return (mean * count + prior * priorStrength) / (count + priorStrength);
}

/** Popularity × quality shortlist; bounds how many movies are scored per request. */
function shortlistByPopularity(
  candidates: Movie[],
  movieStats: Map<number, { count: number; mean: number }>,
  globalMean: number,
  size: number,
): Movie[] {
  if (candidates.length <= size) return candidates;
  const scored = candidates.map((movie) => {
    const stats = movieStats.get(movie.id) ?? { count: 0, mean: globalMean };
    const quality = bayesianMean(stats.count, stats.mean, globalMean, 25);
    return { movie, score: quality * Math.log1p(stats.count) };
  });
  scored.sort((a, b) => b.score - a.score || a.movie.id - b.movie.id);
  return scored.slice(0, size).map((s) => s.movie);
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

function round3(v: number): number {
  return Math.round(v * 1000) / 1000;
}

const LABELS: Record<string, string> = {
  knn: "KNN",
  xgboost: "XGBoost",
  gradient_boosting: "Gradient Boosting",
  random_forest: "Random Forest",
  adaboost: "AdaBoost",
  linear_regression: "Linear Regression",
};
