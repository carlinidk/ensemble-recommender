import type { ExperimentResult } from "./experiments";
import type { RecommendConfig, RecommendOutput } from "./recommender/engine";
import type { TasteProfile, UserRating } from "./recommender/profile";
import type { DataAudit, Movie, TrainingConfig } from "./types";

/**
 * The worker is the "offline training + inference service": models are trained
 * once, stay resident in the worker, and every recommendation request scores
 * candidates from the already-fitted base learners and meta learner instead of
 * retraining anything.
 */

export interface DatasetMeta {
  audit: DataAudit;
  globalMean: number;
  genres: string[];
  movies: Movie[];
  movieStats: { movieId: number; count: number; mean: number }[];
  users: { id: number; age: number; gender: string; occupation: string }[];
}

export type WorkerRequest =
  | { type: "load" }
  | { type: "train"; config: TrainingConfig }
  | { type: "profile"; ratings: UserRating[]; minSharedMovies?: number }
  | { type: "recommend"; ratings: UserRating[]; config: RecommendConfig }
  | {
      type: "score-movies";
      ratings: UserRating[];
      movieIds: number[];
    };

export interface MovieScore {
  movieId: number;
  predictedRating: number | null;
  basePredictions: { name: string; value: number }[];
  personaUserId: number | null;
  source: "stacking_model" | "no_persona";
}

export type WorkerResponse =
  | { type: "ready"; meta: DatasetMeta }
  | { type: "progress"; phase: string; pct: number; message: string }
  | { type: "trained"; result: PersistableExperiment }
  | { type: "profiled"; profile: TasteProfile }
  | { type: "recommended"; output: RecommendOutput }
  | { type: "scored"; scores: MovieScore[] }
  | { type: "error"; message: string };

/** Result payload shape once in-memory artifacts are removed. */
export type PersistableExperiment = Omit<ExperimentResult, "artifacts">;
