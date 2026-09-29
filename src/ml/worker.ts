import { buildDataset, fetchRawMl100k, type LoadedDataset } from "./dataset";
import { runExperiment, type ExperimentResult } from "./experiments";
import { encodeCandidates } from "./features";
import { matrixFromColumns } from "./matrix";
import { recommend } from "./recommender/engine";
import { buildTasteProfile, type UserRating } from "./recommender/profile";
import type { ScoredBatch } from "./recommender/engine";
import type {
  DatasetMeta,
  MovieScore,
  WorkerRequest,
  WorkerResponse,
} from "./workerProtocol";
import type { BaseModelName, Movie, TrainingConfig } from "./types";

/**
 * Training/inference worker.
 *
 * Everything expensive happens here so the UI thread stays interactive:
 *  - `load`   parses MovieLens 100K once and publishes catalogue metadata;
 *  - `train`  runs the full experiment (standalone models + 7 stacking
 *             configurations) with k-fold out-of-fold predictions, and KEEPS the
 *             fitted base learners and meta learner resident;
 *  - `profile`/`recommend`/`score-movies` serve predictions from those resident
 *             models. No request ever retrains a model.
 */

const PROPOSED_BASES: BaseModelName[] = ["knn", "xgboost", "gradient_boosting"];
const META_RECIPE =
  "KNN + XGBoost + Gradient Boosting → Linear Regression meta-learner";

const ctx = self as unknown as {
  postMessage: (message: WorkerResponse) => void;
  addEventListener: (type: "message", handler: (event: MessageEvent<WorkerRequest>) => void) => void;
};

let dataset: LoadedDataset | null = null;
let trained: ExperimentResult | null = null;

function post(message: WorkerResponse): void {
  ctx.postMessage(message);
}

function requireDataset(): LoadedDataset {
  if (!dataset) throw new Error("Dataset has not been loaded yet.");
  return dataset;
}

function requireTrained(): ExperimentResult {
  if (!trained) {
    throw new Error(
      "No trained models in this session. Run the training pipeline first (Models → Run experiment).",
    );
  }
  return trained;
}

function datasetMeta(source: LoadedDataset): DatasetMeta {
  return {
    audit: source.audit,
    globalMean: source.globalMean,
    genres: source.genreNames,
    movies: source.movies,
    movieStats: [...source.movieStats.entries()].map(([movieId, stats]) => ({
      movieId,
      count: stats.count,
      mean: stats.mean,
    })),
    users: [...source.usersById.values()],
  };
}

function buildProfile(ratings: UserRating[], minSharedMovies = 8) {
  const source = requireDataset();
  const cleaned = ratings.filter(
    (r) => Number.isFinite(r.movieId) && Number.isFinite(r.rating) && r.rating >= 1 && r.rating <= 5,
  );
  return buildTasteProfile({
    ratings: cleaned,
    moviesById: source.moviesById,
    ratingsByMovie: source.ratingsByMovie,
    globalMean: source.globalMean,
    minSharedMovies,
  });
}

function scoreCandidateIds(userId: number, movieIds: number[]): ScoredBatch {
  const result = requireTrained();
  const source = requireDataset();
  const space = result.artifacts.featureSpace;
  const X = encodeCandidates(
    movieIds.map((movieId) => ({
      userId,
      movieId,
      year: source.moviesById.get(movieId)?.year ?? null,
    })),
    space,
  );
  const perModel: Record<string, number[]> = {};
  for (const name of PROPOSED_BASES) {
    const base = result.artifacts.bases.find((b) => b.name === name);
    if (!base) throw new Error(`Base learner ${name} is missing from the trained run.`);
    perModel[name] = Array.from(base.model.predict(X));
  }
  const meta = result.artifacts.proposedMeta;
  if (!meta) throw new Error("The proposed meta learner is missing from the trained run.");
  const final = meta.predict(matrixFromColumns(PROPOSED_BASES.map((n) => perModel[n]), [...PROPOSED_BASES]));
  return { perModel, final: Array.from(final), metaRecipe: META_RECIPE };
}

/** The fitted models stay resident in the worker; only metadata crosses the boundary. */
function stripArtifacts(result: ExperimentResult) {
  const rest: Partial<ExperimentResult> = { ...result };
  delete rest.artifacts;
  return rest as Omit<ExperimentResult, "artifacts">;
}

async function handle(request: WorkerRequest): Promise<void> {
  switch (request.type) {
    case "load": {
      if (!dataset) {
        const files = await fetchRawMl100k();
        dataset = buildDataset(files);
      }
      post({ type: "ready", meta: datasetMeta(dataset) });
      return;
    }
    case "train": {
      const source = requireDataset();
      const result = await runExperiment(source, request.config as TrainingConfig, {
        onProgress: (update) =>
          post({ type: "progress", phase: update.phase, pct: update.pct, message: update.message }),
      });
      trained = result;
      post({ type: "trained", result: stripArtifacts(result) });
      return;
    }
    case "profile": {
      const profile = buildProfile(request.ratings, request.minSharedMovies ?? 8);
      post({ type: "profiled", profile });
      return;
    }
    case "recommend": {
      const source = requireDataset();
      const ratings = request.ratings.filter((r) => r.rating >= 1 && r.rating <= 5);
      const profile = buildProfile(ratings);
      const output = recommend({
        ratings,
        profile,
        moviesById: source.moviesById,
        movieStats: source.movieStats,
        ratingsByUser: source.ratingsByUser,
        genreNames: source.genreNames,
        globalMean: source.globalMean,
        config: request.config,
        score: scoreCandidateIds,
      });
      post({ type: "recommended", output });
      return;
    }
    case "score-movies": {
      const source = requireDataset();
      const ratings = request.ratings.filter((r) => r.rating >= 1 && r.rating <= 5);
      const profile = buildProfile(ratings);
      const movies = request.movieIds
        .map((id) => source.moviesById.get(id))
        .filter((m): m is Movie => m !== undefined);

      if (!profile.persona || movies.length === 0) {
        post({
          type: "scored",
          scores: movies.map((m) => ({
            movieId: m.id,
            predictedRating: null,
            basePredictions: [],
            personaUserId: profile.persona?.userId ?? null,
            source: "no_persona" as const,
          })),
        });
        return;
      }

      const batch = scoreCandidateIds(profile.persona.userId, movies.map((m) => m.id));
      const scores: MovieScore[] = movies.map((movie, i) => ({
        movieId: movie.id,
        predictedRating: Math.min(5, Math.max(1, batch.final[i])),
        basePredictions: Object.entries(batch.perModel).map(([name, values]) => ({
          name,
          value: Math.round(values[i] * 100) / 100,
        })),
        personaUserId: profile.persona?.userId ?? null,
        source: "stacking_model" as const,
      }));
      post({ type: "scored", scores });
      return;
    }
    default: {
      post({ type: "error", message: `Unknown request: ${JSON.stringify(request)}` });
    }
  }
}

ctx.addEventListener("message", (event) => {
  void handle(event.data).catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    post({ type: "error", message });
  });
});
