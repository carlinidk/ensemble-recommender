import { ML100K_VERSION, type LoadedDataset } from "./dataset";
import { encodeRows, fitFeatureSpace, type FeatureSpace } from "./features";
import { histogram, residuals } from "./metrics";
import { BASE_MODEL_LABELS, type SearchMode } from "./models";
import {
  PAPER,
  PAPER_PROPOSED,
  PAPER_STACKING,
  PAPER_STANDALONE,
} from "./reference";
import { createRng, subsampleIndices } from "./rng";
import { makeSplit, type SplitStrategy } from "./split";
import { basePredictionCorrelations, runStack, trainBaseLearners, type TrainedBase } from "./stacking";
import {
  BASE_MODEL_NAMES,
  type BaseModelName,
  type DataAudit,
  type FeatureImportance,
  type MergedRow,
  type MetaModelName,
  type Regressor,
  type TrainingConfig,
} from "./types";

/**
 * Phase 5 + 7: standalone evaluation and every stacking configuration.
 *
 * The seven configurations are the ones listed in the project brief, which
 * mirror the meta-learner combinations the paper reports. All of them are
 * re-run here — the paper's published numbers appear only as reference columns.
 */

export interface StandaloneRow {
  model: BaseModelName;
  label: string;
  mae: number;
  mse: number;
  rmse: number;
  oofMae: number;
  oofRmse: number;
  fitSeconds: number;
  params: Record<string, number | string | boolean>;
  importance: FeatureImportance[];
  referenceRmse: number | null;
  deltaVsReference: number | null;
}

export interface StackingRow {
  id: string;
  label: string;
  baseNames: BaseModelName[];
  metaName: MetaModelName;
  mae: number;
  mse: number;
  rmse: number;
  referenceRmse: number;
  deltaVsReference: number;
  /** 1 = lowest RMSE in this experiment (experimental reporting only). */
  rank: number;
  metaWeights: { name: string; weight: number; direction: number }[] | null;
  metaParams: Record<string, number | string | boolean>;
}

export interface ProgressUpdate {
  phase: string;
  pct: number;
  message: string;
}

export interface ExperimentResult {
  modelId: string;
  version: string;
  createdAt: number;
  datasetVersion: string;
  featureVersion: string;
  config: TrainingConfig & { split: SplitStrategy };
  search: SearchMode;
  dataset: DataAudit & {
    scope: TrainingConfig["scope"];
    rowsUsed: number;
    nTrain: number;
    nTest: number;
  };
  features: { mode: string; names: string[]; descriptions: Record<string, string> };
  split: {
    strategy: SplitStrategy;
    explanation: string;
    trainTimestampRange: [number, number];
    testTimestampRange: [number, number];
  };
  standalone: StandaloneRow[];
  stacking: StackingRow[];
  lowestRmse: { scope: "stacking" | "standalone"; label: string; rmse: number };
  proposed: {
    id: string;
    label: string;
    ourMae: number;
    ourMse: number;
    ourRmse: number;
    paperMae: number;
    paperMse: number;
    paperRmse: number;
  };
  plots: {
    actual: number[];
    predicted: number[];
    residuals: number[];
    errorHistogram: { bin: number; count: number }[];
    standaloneRmse: { name: string; rmse: number; reference: number | null }[];
    standaloneMae: { name: string; mae: number }[];
    standaloneMse: { name: string; mse: number }[];
    stackingRmse: { name: string; rmse: number; reference: number }[];
    correlation: { names: string[]; matrix: number[][] };
    metaWeights: { name: string; weight: number }[];
  };
  timings: { totalSeconds: number; baseLearnersSeconds: number; perModel: { name: string; seconds: number }[] };
  paper: {
    citation: typeof PAPER;
    standalone: typeof PAPER_STANDALONE;
    stacking: typeof PAPER_STACKING;
    proposed: typeof PAPER_PROPOSED;
  };
  notes: string[];
  /** In-memory serving artifacts — stripped before persisting to the database. */
  artifacts: {
    bases: TrainedBase[];
    proposedMeta: Regressor | null;
    proposedMetaName: MetaModelName;
    proposedMetaWeights: { name: string; weight: number; direction: number }[] | null;
    featureSpace: FeatureSpace;
  };
}

const STACKING_CONFIGS: {
  id: string;
  baseNames: BaseModelName[];
  metaName: MetaModelName;
}[] = [
  { id: "A", baseNames: ["knn", "xgboost"], metaName: "linear_regression" },
  { id: "B", baseNames: ["knn", "xgboost"], metaName: "xgboost" },
  { id: "C", baseNames: ["knn", "gradient_boosting", "random_forest"], metaName: "xgboost" },
  { id: "D", baseNames: ["knn", "xgboost", "gradient_boosting"], metaName: "xgboost" },
  { id: "E", baseNames: ["knn", "xgboost", "gradient_boosting"], metaName: "linear_regression" },
  { id: "F", baseNames: ["knn", "random_forest"], metaName: "xgboost" },
  { id: "G", baseNames: ["knn", "xgboost", "gradient_boosting", "adaboost"], metaName: "linear_regression" },
];

export const PROPOSED_CONFIG_ID = "E";

export function stackingConfigurations() {
  return STACKING_CONFIGS.map((c) => ({ ...c }));
}

export interface RunOptions {
  onProgress?: (update: ProgressUpdate) => void;
}

const SEARCH_BY_SCOPE: Record<TrainingConfig["scope"], SearchMode> = {
  sample: "quick",
  full: "research",
};

export async function runExperiment(
  dataset: LoadedDataset,
  config: TrainingConfig,
  options: RunOptions = {},
): Promise<ExperimentResult> {
  const startedAt = performance.now();
  const report = (phase: string, pct: number, message: string) =>
    options.onProgress?.({ phase, pct, message });
  const search: SearchMode = config.search ?? SEARCH_BY_SCOPE[config.scope];

  report("prepare", 2, "Selecting dataset scope");

  let rows: MergedRow[] = dataset.rows;
  if (config.scope === "sample" && config.sampleSize < rows.length) {
    const pick = subsampleIndices(rows.length, config.sampleSize, createRng(config.randomSeed));
    rows = Array.from(pick).map((i) => dataset.rows[i]);
  }

  const split = makeSplit(rows, config.split, config.testSize, createRng(config.randomSeed + 7));
  report("split", 8, `Splitting rows (${split.strategy})`);

  const includeGenres =
    config.mode === "enhanced" && config.enhancedFeatures === "with_genres";

  const space = fitFeatureSpace({
    rows,
    trainIdx: split.train,
    mode: config.mode,
    moviesById: dataset.moviesById,
    genreNames: dataset.genreNames,
    includeGenres,
  });

  const Xtrain = encodeRows(rows, split.train, space, { leaveOneOut: config.mode === "enhanced" });
  const Xtest = encodeRows(rows, split.test, space);
  const ytrain = Float64Array.from(split.train, (i) => rows[i].rating);
  const ytest = Float64Array.from(split.test, (i) => rows[i].rating);
  report("features", 12, `Encoded ${Xtrain.d} features`);

  const bases = trainBaseLearners({
    rows,
    trainIdx: split.train,
    Xtrain,
    ytrain,
    Xtest,
    ytest,
    names: BASE_MODEL_NAMES,
    folds: config.cvFolds,
    seed: config.randomSeed,
    search,
    mode: config.mode,
    moviesById: dataset.moviesById,
    genreNames: dataset.genreNames,
    includeGenres,
    onFit: (model, done, total) => {
      const pct = 12 + (done / total) * 74;
      report("base-learners", pct, `Fitting ${BASE_MODEL_LABELS[model]} (${done}/${total})`);
    },
  });

  report("meta-learner", 88, "Training meta-learner configurations");

  const stacking: StackingRow[] = [];
  let proposedMeta: Regressor | null = null;
  let proposedMetaWeights: { name: string; weight: number; direction: number }[] | null = null;
  const predictionsById = new Map<string, Float64Array>();

  STACKING_CONFIGS.forEach((combo, index) => {
    const outcome = runStack({
      bases,
      ytrain,
      ytest,
      combo,
      seed: config.randomSeed + 500 + index,
      search,
    });
    const reference = PAPER_STACKING.find((r) => r.id === combo.id)!;
    if (combo.id === PROPOSED_CONFIG_ID) {
      proposedMeta = outcome.meta;
      proposedMetaWeights = outcome.metaWeights;
    }
    predictionsById.set(combo.id, outcome.testPredictions);
    stacking.push({
      id: combo.id,
      label: reference.label,
      baseNames: combo.baseNames,
      metaName: combo.metaName,
      mae: outcome.metrics.mae,
      mse: outcome.metrics.mse,
      rmse: outcome.metrics.rmse,
      referenceRmse: reference.paperRmse,
      deltaVsReference: outcome.metrics.rmse - reference.paperRmse,
      rank: 0,
      metaWeights: outcome.metaWeights,
      metaParams: outcome.metaParams,
    });
  });

  const ranked = [...stacking].sort((a, b) => a.rmse - b.rmse);
  ranked.forEach((row, i) => {
    row.rank = i + 1;
  });

  const standalone: StandaloneRow[] = bases.map((b) => {
    const reference = PAPER_STANDALONE[b.name];
    return {
      model: b.name,
      label: BASE_MODEL_LABELS[b.name],
      mae: b.standalone.mae,
      mse: b.standalone.mse,
      rmse: b.standalone.rmse,
      oofMae: b.oofMetrics.mae,
      oofRmse: b.oofMetrics.rmse,
      fitSeconds: b.fitSeconds,
      params: b.params,
      importance: b.importance,
      referenceRmse: reference ? reference.rmse : null,
      deltaVsReference: reference ? b.standalone.rmse - reference.rmse : null,
    };
  });

  const proposedRow = stacking.find((s) => s.id === PROPOSED_CONFIG_ID) ?? stacking[0];
  const proposedPredictions = predictionsById.get(proposedRow.id) ?? bases[0].test;

  const plots = {
    ...samplePlotPoints(ytest, proposedPredictions, 1200),
    errorHistogram: histogram(residuals(ytest, proposedPredictions), 30).map((b) => ({
      bin: Math.round(((b.binStart + b.binEnd) / 2) * 100) / 100,
      count: b.count,
    })),
    standaloneRmse: standalone.map((s) => ({
      name: s.label,
      rmse: s.rmse,
      reference: s.referenceRmse,
    })),
    standaloneMae: standalone.map((s) => ({ name: s.label, mae: s.mae })),
    standaloneMse: standalone.map((s) => ({ name: s.label, mse: s.mse })),
    stackingRmse: stacking.map((s) => ({ name: s.label, rmse: s.rmse, reference: s.referenceRmse })),
    correlation: basePredictionCorrelations(bases),
    metaWeights: (proposedRow.metaWeights ?? []).map((w) => ({ name: w.name, weight: w.weight })),
  };

  const bestStandalone = [...standalone].sort((a, b) => a.rmse - b.rmse)[0];
  const lowestRmse = ranked[0].rmse <= bestStandalone.rmse
    ? { scope: "stacking" as const, label: ranked[0].label, rmse: ranked[0].rmse }
    : { scope: "standalone" as const, label: bestStandalone.label, rmse: bestStandalone.rmse };

  const featureVersion =
    config.mode === "paper"
      ? "paper-v1"
      : config.enhancedFeatures === "with_genres"
        ? "enhanced-genres-v1"
        : "enhanced-core-v1";
  const modelId = `stacking-v1.0.0-${config.mode}-${config.scope}-seed${config.randomSeed}`;

  const totalSeconds = (performance.now() - startedAt) / 1000;
  report("evaluation", 96, "Computing metrics and plots");

  const result: ExperimentResult = {
    modelId,
    version: "1.0.0",
    createdAt: Date.now(),
    datasetVersion: ML100K_VERSION,
    featureVersion,
    config,
    search,
    dataset: {
      ...dataset.audit,
      scope: config.scope,
      rowsUsed: rows.length,
      nTrain: split.train.length,
      nTest: split.test.length,
    },
    features: { mode: config.mode, names: Xtrain.names, descriptions: space.descriptions },
    split: {
      strategy: split.strategy,
      explanation: split.explanation,
      trainTimestampRange: split.trainTimestampRange,
      testTimestampRange: split.testTimestampRange,
    },
    standalone,
    stacking,
    lowestRmse,
    proposed: {
      id: PROPOSED_CONFIG_ID,
      label: PAPER_PROPOSED.label,
      ourMae: proposedRow.mae,
      ourMse: proposedRow.mse,
      ourRmse: proposedRow.rmse,
      paperMae: PAPER_PROPOSED.mae,
      paperMse: PAPER_PROPOSED.mse,
      paperRmse: PAPER_PROPOSED.rmse,
    },
    plots,
    timings: {
      totalSeconds,
      baseLearnersSeconds: bases.reduce((acc, b) => acc + b.fitSeconds, 0),
      perModel: bases.map((b) => ({ name: b.name, seconds: b.fitSeconds })),
    },
    paper: {
      citation: PAPER,
      standalone: PAPER_STANDALONE,
      stacking: PAPER_STACKING,
      proposed: PAPER_PROPOSED,
    },
    notes: buildNotes(config, search),
    artifacts: {
      bases,
      proposedMeta,
      proposedMetaName:
        STACKING_CONFIGS.find((c) => c.id === PROPOSED_CONFIG_ID)?.metaName ?? "linear_regression",
      proposedMetaWeights,
      featureSpace: space,
    },
  };

  report("done", 100, "Experiment complete");
  return result;
}

function samplePlotPoints(
  actual: Float64Array,
  predicted: Float64Array,
  cap: number,
): { actual: number[]; predicted: number[]; residuals: number[] } {
  const n = actual.length;
  const stride = Math.max(1, Math.ceil(n / cap));
  const a: number[] = [];
  const p: number[] = [];
  const r: number[] = [];
  for (let i = 0; i < n; i += stride) {
    a.push(round2(actual[i]));
    p.push(round2(predicted[i]));
    r.push(round2(predicted[i] - actual[i]));
  }
  return { actual: a, predicted: p, residuals: r };
}

function round2(v: number): number {
  return Math.round(v * 100) / 100;
}

function buildNotes(config: TrainingConfig, search: SearchMode): string[] {
  const notes = [
    `Run configuration: mode=${config.mode}, scope=${config.scope}, ${config.cvFolds}-fold cross-validated stacking, seed=${config.randomSeed}, test_size=${config.testSize}, search=${search}.`,
    "Meta-learner training features are out-of-fold base predictions, so no in-sample base prediction reaches the meta learner.",
    "Base learners are refit on the complete training split before generating test-set predictions.",
    "Our numbers are computed by this repository. The only published values shown anywhere are labelled 'Published Paper Reference Results'.",
  ];
  if (config.mode === "enhanced") {
    notes.push(
      "ENHANCED MODE: adds movie popularity, movie mean rating, user activity and user mean rating; supervised statistics are fitted inside each fold with leave-one-out encoding for training rows.",
    );
    if (config.enhancedFeatures === "with_genres") {
      notes.push(
        `The 19 genre flags are also included (${config.enhancedFeatures}), which roughly multiplies tree-fitting cost on the full dataset.`,
      );
    }
    if (search === "research" && config.scope === "full") {
      notes.push(
        "KNN is the one learner that degrades with dimensionality: beyond six features its kd-tree pruning stops helping, so the neighbour pool is capped and the cap is recorded in the model parameters.",
      );
    }
  }
  if (config.split === "temporal") {
    notes.push(
      "Temporal split: training interactions are strictly older than test interactions, so accuracy is expected to be lower than under a random split.",
    );
  }
  if (config.scope === "sample") {
    notes.push(
      `Sample scope: only ${config.sampleSize.toLocaleString()} randomly chosen interactions were used, so results are not directly comparable to a full-dataset run.`,
    );
  }
  return notes;
}
