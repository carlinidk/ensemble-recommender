import type { RecommendConfig, RecommendOutput } from "./recommender/engine";
import type { TasteProfile, UserRating } from "./recommender/profile";
import type { TrainingConfig } from "./types";
import type {
  DatasetMeta,
  MovieScore,
  PersistableExperiment,
  WorkerRequest,
  WorkerResponse,
} from "./workerProtocol";

interface PendingRequest {
  resolve: (message: WorkerResponse) => void;
  reject: (error: Error) => void;
  onProgress?: (message: Extract<WorkerResponse, { type: "progress" }>) => void;
}

/**
 * Client for the ML worker.
 *
 * Requests are strictly FIFO: the worker handles one message at a time, so the
 * next non-progress response always belongs to the oldest pending request.
 */
export class MlWorkerClient {
  private worker: Worker;
  private pending: PendingRequest[] = [];
  private disposed = false;

  constructor() {
    this.worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    this.worker.addEventListener("message", (event: MessageEvent<WorkerResponse>) => {
      const message = event.data;
      if (message.type === "progress") {
        this.pending[0]?.onProgress?.(message);
        return;
      }
      const head = this.pending.shift();
      if (!head) return;
      if (message.type === "error") head.reject(new Error(message.message));
      else head.resolve(message);
    });
    this.worker.addEventListener("error", (event) => {
      const error = new Error(event.message || "The ML worker crashed.");
      const head = this.pending.shift();
      if (head) head.reject(error);
    });
  }

  private send(
    request: WorkerRequest,
    onProgress?: PendingRequest["onProgress"],
  ): Promise<WorkerResponse> {
    if (this.disposed) return Promise.reject(new Error("ML worker has been disposed."));
    return new Promise<WorkerResponse>((resolve, reject) => {
      this.pending.push({ resolve, reject, onProgress });
      this.worker.postMessage(request);
    });
  }

  async load(): Promise<DatasetMeta> {
    const response = await this.send({ type: "load" });
    if (response.type !== "ready") throw new Error(`Unexpected response: ${response.type}`);
    return response.meta;
  }

  async train(
    config: TrainingConfig,
    onProgress?: PendingRequest["onProgress"],
  ): Promise<PersistableExperiment> {
    const response = await this.send({ type: "train", config }, onProgress);
    if (response.type !== "trained") throw new Error(`Unexpected response: ${response.type}`);
    return response.result;
  }

  async profile(ratings: UserRating[]): Promise<TasteProfile> {
    const response = await this.send({ type: "profile", ratings });
    if (response.type !== "profiled") throw new Error(`Unexpected response: ${response.type}`);
    return response.profile;
  }

  async recommend(ratings: UserRating[], config: RecommendConfig): Promise<RecommendOutput> {
    const response = await this.send({ type: "recommend", ratings, config });
    if (response.type !== "recommended") throw new Error(`Unexpected response: ${response.type}`);
    return response.output;
  }

  async scoreMovies(ratings: UserRating[], movieIds: number[]): Promise<MovieScore[]> {
    const response = await this.send({ type: "score-movies", ratings, movieIds });
    if (response.type !== "scored") throw new Error(`Unexpected response: ${response.type}`);
    return response.scores;
  }

  dispose(): void {
    this.disposed = true;
    this.worker.terminate();
  }
}
