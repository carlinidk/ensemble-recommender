/**
 * Headless experiment runner (the project's "offline training job").
 *
 *   bun run experiment                       # paper mode, full 100k, 5-fold CV
 *   bun run experiment --scope sample        # 30k-row quick run
 *   bun run experiment --mode enhanced       # documented feature enhancements
 *   bun run experiment --split temporal      # older interactions train, later test
 *   bun run experiment --folds 10 --seed 7
 *
 * Writes the full result object to experiments/results/ so runs can be diffed
 * and cited. The same pipeline runs in the browser worker for the live app; the
 * numbers are produced by identical code paths.
 */
import { buildDataset, ML100K_BASE_PATH } from "../src/ml/dataset";
import { runExperiment } from "../src/ml/experiments";
import type { DatasetScope, SplitStrategy, TrainingMode } from "../src/ml/types";

const args = new Map<string, string>();
for (let i = 2; i < process.argv.length; i++) {
  const token = process.argv[i];
  if (!token.startsWith("--")) continue;
  const [key, inline] = token.slice(2).split("=");
  const value = inline ?? process.argv[i + 1];
  if (inline === undefined && value !== undefined && !value.startsWith("--")) i++;
  args.set(key, value ?? "true");
}

const mode = (args.get("mode") ?? "paper") as TrainingMode;
const scope = (args.get("scope") ?? "full") as DatasetScope;
const split = (args.get("split") ?? "random") as SplitStrategy;
const folds = Number(args.get("folds") ?? 5);
const seed = Number(args.get("seed") ?? 42);
const sampleSize = Number(args.get("sample-size") ?? 30000);
const enhancedFeatures = args.get("genres") === "true" ? "with_genres" : "core";

const base = `public${ML100K_BASE_PATH}`;
const dataset = buildDataset({
  movies: await Bun.file(`${base}/u.item`).text(),
  ratings: await Bun.file(`${base}/u.data`).text(),
  users: await Bun.file(`${base}/u.user`).text(),
  genres: await Bun.file(`${base}/u.genre`).text(),
});

console.log(
  `MovieLens 100K loaded: ${dataset.audit.rows} ratings · ${dataset.audit.users} users · ${dataset.audit.movies} movies`,
);
console.log(
  `Run: mode=${mode}${mode === "enhanced" ? ` (${enhancedFeatures})` : ""} scope=${scope} split=${split} folds=${folds} seed=${seed}\n`,
);

const started = performance.now();
const result = await runExperiment(
  dataset,
  {
    mode,
    scope,
    sampleSize,
    randomSeed: seed,
    testSize: 0.2,
    cvFolds: folds,
    split,
    enhancedFeatures,
  },
  {
    onProgress: (update) =>
      process.stdout.write(`\r[${update.pct.toFixed(0).padStart(3)}%] ${update.message.padEnd(60)}`),
  },
);

console.log("\n");
console.log("STANDALONE MODELS (ours vs published reference)");
console.log(
  ["model".padEnd(22), "mae".padStart(7), "mse".padStart(7), "rmse".padStart(7), "oof".padStart(7), "paper".padStart(7), "delta".padStart(8)].join(" "),
);
for (const row of result.standalone) {
  console.log(
    [
      row.label.padEnd(22),
      row.mae.toFixed(4).padStart(7),
      row.mse.toFixed(4).padStart(7),
      row.rmse.toFixed(4).padStart(7),
      row.oofRmse.toFixed(4).padStart(7),
      String(row.referenceRmse ?? "-").padStart(7),
      (row.deltaVsReference === null ? "-" : row.deltaVsReference.toFixed(4)).padStart(8),
    ].join(" "),
  );
}

console.log("\nSTACKING CONFIGURATIONS");
console.log(
  ["id".padEnd(3), "config".padEnd(30), "mae".padStart(7), "mse".padStart(7), "rmse".padStart(7), "paper".padStart(7), "rank".padStart(5)].join(" "),
);
for (const row of result.stacking) {
  console.log(
    [
      row.id.padEnd(3),
      row.label.padEnd(30),
      row.mae.toFixed(4).padStart(7),
      row.mse.toFixed(4).padStart(7),
      row.rmse.toFixed(4).padStart(7),
      row.referenceRmse.toFixed(2).padStart(7),
      String(row.rank).padStart(5),
    ].join(" "),
  );
}

console.log("\nProposed stack (paper configuration):", result.proposed.label);
console.log(
  `  ours   MAE ${result.proposed.ourMae.toFixed(4)}  MSE ${result.proposed.ourMse.toFixed(4)}  RMSE ${result.proposed.ourRmse.toFixed(4)}`,
);
console.log(
  `  paper  MAE ${result.proposed.paperMae.toFixed(2)}  MSE ${result.proposed.paperMse.toFixed(2)}  RMSE ${result.proposed.paperRmse.toFixed(2)}  (published reference)`,
);
console.log(`\nLowest-RMSE configuration in this experiment: ${result.lowestRmse.label} (${result.lowestRmse.rmse.toFixed(4)})`);
console.log(`Total wall clock: ${((performance.now() - started) / 1000).toFixed(1)}s`);

const { artifacts: _artifacts, ...persistable } = result;
void _artifacts;
const stamp = new Date(result.createdAt).toISOString().replace(/[:.]/g, "-");
const outPath = `experiments/results/${stamp}_${mode}_${scope}_${split}_seed${seed}.json`;
await Bun.write(outPath, JSON.stringify(persistable, null, 2));
console.log(`\nWrote ${outPath}`);
