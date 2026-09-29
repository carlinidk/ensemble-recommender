import { encodeRows, fitFeatureSpace, type FeatureSpace } from "./features";
import { columnValues, complementIndices, matrixFromColumns, subMatrix } from "./matrix";
import { evaluate } from "./metrics";
import { createBaseModel, createMetaModel, type SearchMode } from "./models";
import { LinearRegression } from "./models/linearRegression";
import { createRng, kFoldIndices } from "./rng";
import type {
  BaseModelName,
  FeatureImportance,
  Matrix,
  MergedRow,
  Metrics,
  MetaModelName,
  Movie,
  Regressor,
  TrainingMode,
} from "./types";

/**
 * LEAKAGE-SAFE STACKING (Level 1 + Level 2).
 *
 * The meta learner must never see in-sample base predictions. The pipeline is:
 *
 *   1. split the training data into K folds;
 *   2. for each fold, train every base learner on the other K-1 folds and
 *      predict the held-out fold -> out-of-fold (OOF) prediction matrix;
 *   3. train the meta learner on [OOF_KNN, OOF_XGB, OOF_GB, ...] vs the target;
 *   4. refit every base learner on the COMPLETE training split (these are the
 *      models used for serving);
 *   5. run the test set through those refit base learners to build the
 *      meta-learner input matrix, and predict with the trained meta learner;
 *   6. evaluate the final stacked prediction against the test target.
 *
 * Because base-learner training does not depend on the meta learner, the OOF
 * matrix is computed once per run and every candidate meta configuration reuses
 * it — that keeps all seven paper configurations cheap while remaining
 * methodologically identical to training them separately.
 */

export interface TrainedBase {
  name: BaseModelName;
  /** Out-of-fold predictions aligned with the training rows. */
  oof: Float64Array;
  /** Test predictions from the model refit on the full training split. */
  test: Float64Array;
  model: Regressor;
  standalone: Metrics;
  oofMetrics: Metrics;
  fitSeconds: number;
  importance: FeatureImportance[];
  params: Record<string, number | string | boolean>;
}

export interface BaseTrainingInput {
  rows: MergedRow[];
  trainIdx: Int32Array;
  Xtrain: Matrix;
  ytrain: Float64Array;
  Xtest: Matrix;
  ytest: Float64Array;
  names: BaseModelName[];
  folds: number;
  seed: number;
  search: SearchMode;
  mode: TrainingMode;
  moviesById: Map<number, Movie>;
  genreNames: string[];
  includeGenres: boolean;
  onFit?: (model: BaseModelName, done: number, total: number) => void;
}

export function trainBaseLearners(input: BaseTrainingInput): TrainedBase[] {
  const {
    rows,
    trainIdx,
    Xtrain,
    ytrain,
    Xtest,
    ytest,
    names,
    folds,
    seed,
    search,
    mode,
    moviesById,
    genreNames,
    includeGenres,
  } = input;

  // Use a dedicated stream so that adding a base learner does not change the
  // fold assignment of the others (reproducibility across configurations).
  const foldRng = createRng(seed + 101);
  const foldBuckets = kFoldIndices(trainIdx.length, folds, foldRng);
  const totalFits = names.length * (foldBuckets.length + 1);
  let done = 0;

  const result: TrainedBase[] = [];

  for (const name of names) {
    const oof = new Float64Array(trainIdx.length);
    const started = performance.now();
    let foldSpace: FeatureSpace | null = null;

    for (let f = 0; f < foldBuckets.length; f++) {
      const holdLocal = foldBuckets[f];
      const fitLocal = complementIndices(
        Int32Array.from({ length: trainIdx.length }, (_, i) => i),
        holdLocal,
      );

      let Xfit = subMatrix(Xtrain, fitLocal);
      let Xhold = subMatrix(Xtrain, holdLocal);
      let yfit = new Float64Array(fitLocal.length);
      for (let i = 0; i < fitLocal.length; i++) yfit[i] = ytrain[fitLocal[i]];
      const yhold = new Float64Array(holdLocal.length);
      for (let i = 0; i < holdLocal.length; i++) yhold[i] = ytrain[holdLocal[i]];

      if (mode === "enhanced") {
        // Supervised features must be refit inside the fold, otherwise the
        // held-out rows would contribute to their own movie/user averages.
        const rowsFit = Int32Array.from(fitLocal, (i) => trainIdx[i]);
        foldSpace = fitFeatureSpace({
          rows,
          trainIdx: rowsFit,
          mode,
          moviesById,
          genreNames,
          includeGenres,
        });
        const rowsHold = Int32Array.from(holdLocal, (i) => trainIdx[i]);
        Xfit = encodeRows(rows, rowsFit, foldSpace, { leaveOneOut: true });
        Xhold = encodeRows(rows, rowsHold, foldSpace);
        yfit = Float64Array.from(rowsFit, (i) => rows[i].rating);
      }

      const model = createBaseModel(name, { search, seed: seed + f + 1 });
      model.fit(Xfit, yfit);
      const preds = model.predict(Xhold);
      for (let i = 0; i < holdLocal.length; i++) oof[holdLocal[i]] = preds[i];
      done++;
      input.onFit?.(name, done, totalFits);
    }

    const fullModel = createBaseModel(name, { search, seed });
    fullModel.fit(Xtrain, ytrain);
    const testPreds = fullModel.predict(Xtest);
    done++;
    input.onFit?.(name, done, totalFits);

    result.push({
      name,
      oof,
      test: testPreds,
      model: fullModel,
      standalone: evaluate(ytest, testPreds),
      oofMetrics: evaluate(ytrain, oof),
      fitSeconds: (performance.now() - started) / 1000,
      importance: fullModel.importance?.() ?? [],
      params: fullModel.describe?.() ?? {},
    });
  }

  return result;
}

export interface StackOutcome {
  id: string;
  baseNames: BaseModelName[];
  metaName: MetaModelName;
  metrics: Metrics;
  testPredictions: Float64Array;
  meta: Regressor;
  metaInputNames: string[];
  metaWeights: { name: string; weight: number; direction: number }[] | null;
  metaParams: Record<string, number | string | boolean>;
}

export interface StackInput {
  bases: TrainedBase[];
  ytrain: Float64Array;
  ytest: Float64Array;
  combo: { id: string; baseNames: BaseModelName[]; metaName: MetaModelName };
  seed: number;
  search: SearchMode;
}

export function runStack(input: StackInput): StackOutcome {
  const { bases, ytrain, ytest, combo, seed, search } = input;
  const selected = combo.baseNames.map((name) => {
    const found = bases.find((b) => b.name === name);
    if (!found) throw new Error(`Base learner ${name} was not trained`);
    return found;
  });

  const metaInputNames = selected.map((b) => `${b.name}_prediction`);
  const Ztrain = matrixFromColumns(
    selected.map((b) => b.oof),
    metaInputNames,
  );
  const Ztest = matrixFromColumns(
    selected.map((b) => b.test),
    metaInputNames,
  );

  const meta = createMetaModel(combo.metaName, { search, seed });
  meta.fit(Ztrain, ytrain);
  const finalPredictions = meta.predict(Ztest);

  const metaWeights =
    meta instanceof LinearRegression ? meta.metaWeights() : null;

  return {
    id: combo.id,
    baseNames: combo.baseNames,
    metaName: combo.metaName,
    metrics: evaluate(ytest, finalPredictions),
    testPredictions: finalPredictions,
    meta,
    metaInputNames,
    metaWeights,
    metaParams: meta.describe?.() ?? {},
  };
}

/** Column vectors used for the prediction-correlation heat map. */
export function basePredictionCorrelations(
  bases: TrainedBase[],
): { names: string[]; matrix: number[][] } {
  const names = bases.map((b) => b.name);
  const cols = bases.map((b) => b.test);
  const matrix = names.map((_, i) =>
    names.map((__, j) => pearson(cols[i], cols[j])),
  );
  return { names, matrix };
}

function pearson(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = Math.min(a.length, b.length);
  if (n === 0) return 0;
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < n; i++) {
    ma += a[i];
    mb += b[i];
  }
  ma /= n;
  mb /= n;
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < n; i++) {
    const xa = a[i] - ma;
    const xb = b[i] - mb;
    num += xa * xb;
    da += xa * xa;
    db += xb * xb;
  }
  const den = Math.sqrt(da * db);
  return den === 0 ? 0 : num / den;
}

export { columnValues };
