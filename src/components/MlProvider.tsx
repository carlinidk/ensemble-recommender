import { api } from "@/convex/_generated/api";
import { MlWorkerClient } from "@/ml/client";
import type { RecommendConfig, RecommendOutput } from "@/ml/recommender/engine";
import type { TasteProfile, UserRating } from "@/ml/recommender/profile";
import type { TrainingConfig } from "@/ml/types";
import type { DatasetMeta, MovieScore, PersistableExperiment } from "@/ml/workerProtocol";
import { useMutation } from "convex/react";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";

/**
 * Single owner of the ML worker for the whole app.
 *
 * The provider sits above the router, so navigating between pages never
 * destroys the trained models: they stay resident in the worker and every
 * recommendation request is served from them. Training is the only operation
 * that (re)fits anything, and it is triggered explicitly by the user.
 */

export interface TrainingProgress {
  phase: string;
  pct: number;
  message: string;
}

export interface MlContextValue {
  ready: boolean;
  loading: boolean;
  error: string | null;
  meta: DatasetMeta | null;
  progress: TrainingProgress | null;
  training: boolean;
  result: PersistableExperiment | null;
  hasTrainedModels: boolean;
  train: (config: TrainingConfig) => Promise<PersistableExperiment>;
  profile: (ratings: UserRating[]) => Promise<TasteProfile>;
  recommendations: (ratings: UserRating[], config: RecommendConfig) => Promise<RecommendOutput>;
  scoreMovies: (ratings: UserRating[], movieIds: number[]) => Promise<MovieScore[]>;
  clearError: () => void;
}

const MlContext = createContext<MlContextValue | null>(null);

export function MlProvider({ children }: { children: ReactNode }) {
  const clientRef = useRef<MlWorkerClient | null>(null);
  const [meta, setMeta] = useState<DatasetMeta | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<TrainingProgress | null>(null);
  const [training, setTraining] = useState(false);
  const [result, setResult] = useState<PersistableExperiment | null>(null);
  const recordRun = useMutation(api.recommender.recordRun);

  useEffect(() => {
    const client = new MlWorkerClient();
    clientRef.current = client;
    let cancelled = false;
    client
      .load()
      .then((loaded) => {
        if (cancelled) return;
        setMeta(loaded);
        setLoading(false);
      })
      .catch((cause: unknown) => {
        if (cancelled) return;
        setError(cause instanceof Error ? cause.message : String(cause));
        setLoading(false);
      });
    return () => {
      cancelled = true;
      client.dispose();
      if (clientRef.current === client) clientRef.current = null;
    };
  }, []);

  const requireClient = useCallback((): MlWorkerClient => {
    const client = clientRef.current;
    if (!client) throw new Error("The ML worker is not ready yet.");
    return client;
  }, []);

  const train = useCallback(
    async (config: TrainingConfig) => {
      const client = requireClient();
      setTraining(true);
      setError(null);
      setProgress({ phase: "start", pct: 0, message: "Starting experiment" });
      try {
        const trained = await client.train(config, (update) =>
          setProgress({ phase: update.phase, pct: update.pct, message: update.message }),
        );
        setResult(trained);
        setProgress({ phase: "done", pct: 100, message: "Experiment complete" });
        try {
          await recordRun({
            modelId: trained.modelId,
            version: trained.version,
            algorithm: trained.proposed.label,
            datasetVersion: trained.datasetVersion,
            featureVersion: trained.featureVersion,
            randomSeed: trained.config.randomSeed,
            cvFolds: trained.config.cvFolds,
            datasetScope: trained.config.scope,
            splitStrategy: trained.config.split,
            searchMode: trained.search,
            hyperparameters: {
              standalone: trained.standalone.map((s) => ({ model: s.model, params: s.params })),
              proposed: trained.stacking.find((s) => s.id === "E")?.metaParams ?? {},
              search: trained.search,
            },
            metrics: {
              proposed: trained.proposed,
              lowestRmse: trained.lowestRmse,
              timings: trained.timings,
              dataset: trained.dataset,
              audit: trained.dataset,
            },
            standalone: trained.standalone,
            stacking: trained.stacking,
            trainingSeconds: trained.timings.totalSeconds,
            nTrain: trained.dataset.nTrain,
            nTest: trained.dataset.nTest,
          });
        } catch {
          // Signed-out sessions cannot persist metadata; training still works.
        }
        return trained;
      } catch (cause: unknown) {
        const message = cause instanceof Error ? cause.message : String(cause);
        setError(message);
        setProgress(null);
        throw cause;
      } finally {
        setTraining(false);
      }
    },
    [recordRun, requireClient],
  );

  const profile = useCallback(
    (ratings: UserRating[]) => requireClient().profile(ratings),
    [requireClient],
  );

  const recommendations = useCallback(
    (ratings: UserRating[], config: RecommendConfig) => requireClient().recommend(ratings, config),
    [requireClient],
  );

  const scoreMovies = useCallback(
    (ratings: UserRating[], movieIds: number[]) => requireClient().scoreMovies(ratings, movieIds),
    [requireClient],
  );

  const clearError = useCallback(() => setError(null), []);

  const value: MlContextValue = {
    ready: meta !== null,
    loading,
    error,
    meta,
    progress,
    training,
    result,
    hasTrainedModels: result !== null,
    train,
    profile,
    recommendations,
    scoreMovies,
    clearError,
  };

  return <MlContext.Provider value={value}>{children}</MlContext.Provider>;
}

export function useMl(): MlContextValue {
  const context = useContext(MlContext);
  if (!context) throw new Error("useMl must be used inside <MlProvider>.");
  return context;
}
