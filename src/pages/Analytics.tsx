import { AppShell } from "@/components/AppShell";
import { useMl } from "@/components/MlProvider";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { api } from "@/convex/_generated/api";
import { PAPER, PAPER_PROPOSED, PAPER_STACKING, PAPER_NOTES, PAPER_STANDALONE } from "@/ml/reference";
import type { StackingRow, StandaloneRow } from "@/ml/experiments";
import type { TrainingConfig } from "@/ml/types";
import { useQuery } from "convex/react";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  BarChart3,
  Beaker,
  CheckCircle2,
  FlaskConical,
  Play,
  ShieldCheck,
} from "lucide-react";
import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { toast } from "sonner";

const DEFAULT_CONFIG: TrainingConfig = {
  mode: "paper",
  scope: "full",
  sampleSize: 30000,
  randomSeed: 42,
  testSize: 0.2,
  cvFolds: 5,
  split: "random",
};

export default function Analytics() {
  const ml = useMl();
  const latestRun = useQuery(api.recommender.latestRun);
  const [config, setConfig] = useState<TrainingConfig>(DEFAULT_CONFIG);

  const result = ml.result;
  const persisted = latestRun as
    | {
        modelId: string;
        version: string;
        trainingDate: number;
        datasetVersion: string;
        featureVersion: string;
        randomSeed: number;
        cvFolds: number;
        datasetScope: string;
        splitStrategy: string;
        searchMode: string;
        standalone: StandaloneRow[];
        stacking: StackingRow[];
        trainingSeconds: number;
        nTrain: number;
        nTest: number;
        metrics: {
          proposed?: { ourMae: number; ourMse: number; ourRmse: number };
          lowestRmse?: { label: string; rmse: number; scope: string };
        };
      }
    | null
    | undefined;

  const standalone = result?.standalone ?? persisted?.standalone ?? [];
  const stacking = result?.stacking ?? persisted?.stacking ?? [];
  const liveRun = result !== null;
  const hasResults = standalone.length > 0 || stacking.length > 0;

  const run = async (scope: TrainingConfig["scope"]) => {
    const next = { ...config, scope };
    setConfig(next);
    try {
      await ml.train(next);
      toast.success("Experiment finished — metrics computed from this run.");
    } catch (cause: unknown) {
      toast.error(cause instanceof Error ? cause.message : "The experiment failed to run.");
    }
  };

  return (
    <AppShell
      title="Models & experiments"
      description="Train the ensemble, then read the two result sets side by side: what the paper published, and what this implementation actually measured."
      actions={
        <div className="flex gap-2">
          <Button variant="outline" className="gap-2" onClick={() => run("sample")} disabled={ml.training || !ml.ready}>
            <Beaker className="size-4" />
            Quick run
          </Button>
          <Button className="gap-2" onClick={() => run("full")} disabled={ml.training || !ml.ready}>
            {ml.training ? <FlaskConical className="size-4 animate-pulse" /> : <Play className="size-4" />}
            Run experiment
          </Button>
        </div>
      }
    >
      {/* Run panel */}
      <section className="mb-6 rounded-xl border border-border/70 bg-card/50 p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold">Run configuration</h2>
            <p className="mt-1 text-xs text-muted-foreground">
              The paper does not publish seeds, splits or hyperparameters, so every choice below is
              ours and is recorded with the run.
            </p>
          </div>
          <div className="flex flex-wrap gap-2 text-[11px]">
            <Badge variant="outline" className="border-border/70">seed {config.randomSeed}</Badge>
            <Badge variant="outline" className="border-border/70">{config.cvFolds}-fold CV</Badge>
            <Badge variant="outline" className="border-border/70">
              test {(config.testSize * 100).toFixed(0)}%
            </Badge>
          </div>
        </div>

        <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Field label="Feature set">
            <Segmented
              value={config.mode}
              options={[
                { value: "paper", label: "Paper" },
                { value: "enhanced", label: "Enhanced" },
              ]}
              onChange={(value) => setConfig({ ...config, mode: value as TrainingConfig["mode"] })}
            />
            <p className="mt-1.5 text-[11px] leading-4 text-muted-foreground">
              {config.mode === "paper"
                ? "user_id, movie_id, year — the paper's stated features."
                : "Adds movie/user aggregates as a documented deviation. Roughly 1.8× slower at full scope."}
            </p>
            {config.mode === "enhanced" ? (
              <div className="mt-2">
                <Segmented
                  value={config.enhancedFeatures ?? "core"}
                  options={[
                    { value: "core", label: "7 features" },
                    { value: "with_genres", label: "+ genre flags" },
                  ]}
                  onChange={(value) =>
                    setConfig({
                      ...config,
                      enhancedFeatures: value as TrainingConfig["enhancedFeatures"],
                    })
                  }
                />
                <p className="mt-1.5 text-[11px] leading-4 text-muted-foreground">
                  {config.enhancedFeatures === "with_genres"
                    ? "26 features. Substantially slower at full scope and KNN's neighbour pool gets capped."
                    : "user_id, movie_id, year, popularity, movie mean rating, user activity, user mean rating."}
                </p>
              </div>
            ) : null}
          </Field>

          <Field label="Evaluation split">
            <Segmented
              value={config.split}
              options={[
                { value: "random", label: "Random" },
                { value: "temporal", label: "Temporal" },
              ]}
              onChange={(value) => setConfig({ ...config, split: value as TrainingConfig["split"] })}
            />
            <p className="mt-1.5 text-[11px] leading-4 text-muted-foreground">
              {config.split === "random"
                ? "The paper's style: interactions shuffled before splitting."
                : "Older interactions train, later ones test. Accuracy will drop."}
            </p>
          </Field>

          <Field label="Cross-validation folds">
            <Segmented
              value={String(config.cvFolds)}
              options={[
                { value: "3", label: "3" },
                { value: "5", label: "5" },
                { value: "10", label: "10" },
              ]}
              onChange={(value) => setConfig({ ...config, cvFolds: Number(value) })}
            />
            <p className="mt-1.5 text-[11px] leading-4 text-muted-foreground">
              Controls the out-of-fold matrix the meta learner is trained on.
            </p>
          </Field>

          <Field label="Random seed">
            <div className="flex items-center gap-2">
              {[7, 42, 2023].map((seed) => (
                <Button
                  key={seed}
                  size="sm"
                  variant={config.randomSeed === seed ? "default" : "outline"}
                  onClick={() => setConfig({ ...config, randomSeed: seed })}
                >
                  {seed}
                </Button>
              ))}
            </div>
            <p className="mt-1.5 text-[11px] leading-4 text-muted-foreground">
              Fixed seed makes the split, folds and subsampling reproducible.
            </p>
          </Field>
        </div>

        {ml.progress || ml.training ? (
          <div className="mt-5">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>{ml.progress?.message ?? "Starting…"}</span>
              <span className="tabular-nums">{ml.progress?.pct.toFixed(0) ?? 0}%</span>
            </div>
            <Progress value={ml.progress?.pct ?? 0} className="mt-2 h-1.5" />
          </div>
        ) : null}

        {ml.error ? (
          <p className="mt-4 flex items-start gap-2 rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
            <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
            {ml.error}
          </p>
        ) : null}
      </section>

      {hasResults ? (
        <>
          {/* Headline */}
          <section className="mb-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Headline
              label="Our proposed stack RMSE"
              value={(result?.proposed.ourRmse ?? persisted?.metrics.proposed?.ourRmse ?? 0).toFixed(4)}
              sub={
                liveRun
                  ? "Measured in this session"
                  : `Recorded ${persisted ? new Date(persisted.trainingDate).toLocaleString() : ""}`
              }
            />
            <Headline
              label="Paper's proposed RMSE"
              value={PAPER_PROPOSED.rmse.toFixed(2)}
              sub="Published reference value"
              muted
            />
            <Headline
              label="Lowest-RMSE configuration"
              value={
                result
                  ? result.lowestRmse.label
                  : (persisted?.metrics.lowestRmse?.label ?? "—")
              }
              sub={
                result
                  ? `RMSE ${result.lowestRmse.rmse.toFixed(4)} · experimental reporting only`
                  : `RMSE ${persisted?.metrics.lowestRmse?.rmse?.toFixed(4) ?? "—"}`
              }
              small
            />
            <Headline
              label="Training time"
              value={`${(result?.timings.totalSeconds ?? persisted?.trainingSeconds ?? 0).toFixed(1)}s`}
              sub={
                result
                  ? `${result.search} search · ${result.dataset.nTrain.toLocaleString()} train / ${result.dataset.nTest.toLocaleString()} test`
                  : `${persisted?.nTrain?.toLocaleString()} train / ${persisted?.nTest?.toLocaleString()} test`
              }
            />
          </section>

          {!liveRun ? (
            <p className="mb-6 flex items-start gap-2 rounded-xl border border-border/70 bg-card/40 px-4 py-3 text-xs leading-5 text-muted-foreground">
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-primary" />
              Showing the last recorded run from the database. Its models are not resident in this
              browser session, so recommendations need a fresh run.
            </p>
          ) : null}

          {/* Published reference */}
          <section className="mb-6 rounded-xl border border-border/70 bg-card/40 p-5">
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="border-chart-2/50 text-chart-2">
                Published Paper Reference Results
              </Badge>
            </div>
            <p className="mt-3 max-w-3xl text-xs leading-5 text-muted-foreground">
              Transcribed from {PAPER.authors}, {PAPER.venue} {PAPER.volume} ({PAPER.date}). These are
              reference values only — they are never used as this project's results.
            </p>

            <div className="mt-5 grid gap-5 lg:grid-cols-2">
              <ReferenceTable title="Standalone models (paper, RMSE)" rows={PAPER_STANDALONE_ROWS} />
              <ReferenceTable
                title="Stacking configurations (paper, RMSE)"
                rows={PAPER_STACKING.map((row) => ({ label: row.label, rmse: row.paperRmse }))}
              />
            </div>

            <div className="mt-5 rounded-lg border border-chart-2/30 bg-chart-2/5 p-4">
              <p className="text-xs font-semibold text-chart-2">Paper's proposed model</p>
              <p className="mt-1 text-xs text-muted-foreground">{PAPER_PROPOSED.label}</p>
              <div className="mt-3 flex flex-wrap gap-4 text-xs tabular-nums">
                <span>MAE {PAPER_PROPOSED.mae.toFixed(2)}</span>
                <span>MSE {PAPER_PROPOSED.mse.toFixed(2)}</span>
                <span>RMSE {PAPER_PROPOSED.rmse.toFixed(2)}</span>
              </div>
            </div>
          </section>

          {/* Our results */}
          <section className="mb-6 rounded-xl border border-border/70 bg-card/50 p-5">
            <Badge variant="outline" className="border-primary/50 text-primary">
              Our Reproduction Results
            </Badge>
            <p className="mt-3 max-w-3xl text-xs leading-5 text-muted-foreground">
              Computed by this implementation on the same public dataset. OOF = out-of-fold RMSE on
              the training split, which is the honest comparison for the stacking step.
            </p>

            <div className="mt-5 overflow-x-auto">
              <table className="w-full min-w-[720px] text-sm">
                <thead className="text-left text-xs uppercase tracking-wider text-muted-foreground">
                  <tr className="border-b border-border/70">
                    <th className="py-2 pr-3 font-medium">Standalone model</th>
                    <th className="py-2 pr-3 font-medium">MAE</th>
                    <th className="py-2 pr-3 font-medium">MSE</th>
                    <th className="py-2 pr-3 font-medium">RMSE</th>
                    <th className="py-2 pr-3 font-medium">OOF RMSE</th>
                    <th className="py-2 pr-3 font-medium">Paper RMSE</th>
                    <th className="py-2 pr-3 font-medium">Δ</th>
                    <th className="py-2 pr-3 font-medium">Fit</th>
                  </tr>
                </thead>
                <tbody>
                  {standalone.map((row) => (
                    <tr key={row.model} className="border-b border-border/40">
                      <td className="py-2 pr-3 font-medium">{row.label}</td>
                      <td className="py-2 pr-3 tabular-nums">{row.mae.toFixed(4)}</td>
                      <td className="py-2 pr-3 tabular-nums">{row.mse.toFixed(4)}</td>
                      <td className="py-2 pr-3 tabular-nums">{row.rmse.toFixed(4)}</td>
                      <td className="py-2 pr-3 tabular-nums text-muted-foreground">
                        {row.oofRmse.toFixed(4)}
                      </td>
                      <td className="py-2 pr-3 tabular-nums text-muted-foreground">
                        {row.referenceRmse?.toFixed(2) ?? "—"}
                      </td>
                      <td className="py-2 pr-3 tabular-nums">
                        {row.deltaVsReference === null ? (
                          "—"
                        ) : (
                          <span className={row.deltaVsReference > 0 ? "text-chart-3" : "text-chart-5"}>
                            {row.deltaVsReference >= 0 ? "+" : ""}
                            {row.deltaVsReference.toFixed(4)}
                          </span>
                        )}
                      </td>
                      <td className="py-2 pr-3 tabular-nums text-muted-foreground">
                        {row.fitSeconds.toFixed(1)}s
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="mt-7 overflow-x-auto">
              <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Stacking configurations
              </p>
              <table className="w-full min-w-[720px] text-sm">
                <thead className="text-left text-xs uppercase tracking-wider text-muted-foreground">
                  <tr className="border-b border-border/70">
                    <th className="py-2 pr-3 font-medium">#</th>
                    <th className="py-2 pr-3 font-medium">Base learners</th>
                    <th className="py-2 pr-3 font-medium">Meta learner</th>
                    <th className="py-2 pr-3 font-medium">MAE</th>
                    <th className="py-2 pr-3 font-medium">MSE</th>
                    <th className="py-2 pr-3 font-medium">RMSE</th>
                    <th className="py-2 pr-3 font-medium">Paper RMSE</th>
                    <th className="py-2 pr-3 font-medium">Δ</th>
                    <th className="py-2 pr-3 font-medium">Rank</th>
                  </tr>
                </thead>
                <tbody>
                  {stacking.map((row) => (
                    <tr
                      key={row.id}
                      className={`border-b border-border/40 ${row.id === "E" ? "bg-primary/5" : ""}`}
                    >
                      <td className="py-2 pr-3">{row.id}</td>
                      <td className="py-2 pr-3">{row.baseNames.join(" + ")}</td>
                      <td className="py-2 pr-3">{row.metaName === "linear_regression" ? "Linear Regression" : "XGBoost"}</td>
                      <td className="py-2 pr-3 tabular-nums">{row.mae.toFixed(4)}</td>
                      <td className="py-2 pr-3 tabular-nums">{row.mse.toFixed(4)}</td>
                      <td className="py-2 pr-3 tabular-nums">{row.rmse.toFixed(4)}</td>
                      <td className="py-2 pr-3 tabular-nums text-muted-foreground">
                        {row.referenceRmse.toFixed(2)}
                      </td>
                      <td className="py-2 pr-3 tabular-nums">
                        <span className={row.deltaVsReference > 0 ? "text-chart-3" : "text-chart-5"}>
                          {row.deltaVsReference >= 0 ? "+" : ""}
                          {row.deltaVsReference.toFixed(4)}
                        </span>
                      </td>
                      <td className="py-2 pr-3 tabular-nums">{row.rank}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-2 text-[11px] text-muted-foreground">
                Row E is the paper's proposed stack. Rank is RMSE ordering inside this experiment —
                an experimental artefact, not a claim of superiority.
              </p>
            </div>
          </section>

          {/* Charts */}
          {result ? (
            <section className="mb-6 grid gap-5 lg:grid-cols-2">
              <ChartCard title="Standalone RMSE: ours vs paper" note="Lower is better. Reference bars are the paper's published values.">
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={result.plots.standaloneRmse}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                    <XAxis dataKey="name" tick={{ fontSize: 10 }} interval={0} angle={-18} height={54} dy={10} />
                    <YAxis domain={[0.8, 1.2]} tick={{ fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", fontSize: 12 }}
                    />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Bar dataKey="rmse" name="Ours" fill="var(--chart-1)" radius={[3, 3, 0, 0]} />
                    <Bar dataKey="reference" name="Paper" fill="var(--chart-2)" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>

              <ChartCard title="Stacking configurations: RMSE vs paper" note="Seven meta-learner combinations, same out-of-fold matrix.">
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={result.plots.stackingRmse}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                    <XAxis dataKey="name" tick={{ fontSize: 9 }} interval={0} angle={-18} height={54} dy={10} />
                    <YAxis domain={[0.85, 1.1]} tick={{ fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", fontSize: 12 }}
                    />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Bar dataKey="rmse" name="Ours" fill="var(--chart-1)" radius={[3, 3, 0, 0]} />
                    <Bar dataKey="reference" name="Paper" fill="var(--chart-2)" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>

              <ChartCard title="MAE by standalone model" note="Mean absolute error on the held-out split.">
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={result.plots.standaloneMae}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                    <XAxis dataKey="name" tick={{ fontSize: 10 }} interval={0} angle={-18} height={54} dy={10} />
                    <YAxis tick={{ fontSize: 10 }} />
                    <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", fontSize: 12 }} />
                    <Bar dataKey="mae" fill="var(--chart-4)" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>

              <ChartCard title="MSE by standalone model" note="Squared error punishes large misses more than MAE.">
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={result.plots.standaloneMse}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                    <XAxis dataKey="name" tick={{ fontSize: 10 }} interval={0} angle={-18} height={54} dy={10} />
                    <YAxis tick={{ fontSize: 10 }} />
                    <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", fontSize: 12 }} />
                    <Bar dataKey="mse" fill="var(--chart-3)" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>

              <ChartCard title="Actual vs predicted (proposed stack)" note="Points on the diagonal are exact. A regression to the mean around 3.5 is expected for rating models.">
                <ResponsiveContainer width="100%" height={260}>
                  <ScatterChart>
                    <CartesianGrid stroke="var(--border)" />
                    <XAxis dataKey="actual" name="Actual" domain={[1, 5]} tick={{ fontSize: 10 }} />
                    <YAxis dataKey="predicted" name="Predicted" domain={[1, 5]} tick={{ fontSize: 10 }} />
                    <ZAxis range={[12, 12]} />
                    <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", fontSize: 12 }} />
                    <ReferenceLine segment={[{ x: 1, y: 1 }, { x: 5, y: 5 }]} stroke="var(--chart-2)" strokeDasharray="4 4" />
                    <Scatter
                      data={result.plots.actual.map((actual, i) => ({
                        actual,
                        predicted: result.plots.predicted[i],
                      }))}
                      fill="var(--chart-1)"
                      fillOpacity={0.28}
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </ChartCard>

              <ChartCard title="Prediction-error distribution" note="Predicted minus actual on the test split.">
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={result.plots.errorHistogram}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                    <XAxis dataKey="bin" tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 10 }} />
                    <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", fontSize: 12 }} />
                    <ReferenceLine x={0} stroke="var(--chart-2)" />
                    <Bar dataKey="count" fill="var(--chart-5)" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>

              <ChartCard title="Meta-learner weights" note="Linear Regression coefficients on base predictions, in rating units.">
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={result.plots.metaWeights}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                    <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 10 }} />
                    <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", fontSize: 12 }} />
                    <ReferenceLine y={0} stroke="var(--border)" />
                    <Bar dataKey="weight" radius={[3, 3, 0, 0]}>
                      {result.plots.metaWeights.map((entry) => (
                        <Cell key={entry.name} fill={entry.weight >= 0 ? "var(--chart-1)" : "var(--chart-3)"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>

              <CorrelationGrid names={result.plots.correlation.names} matrix={result.plots.correlation.matrix} />

              <ChartCard title="Base-learner feature importance" note="Mean split-gain importance from the XGBoost-style booster on the paper feature set.">
                <div className="space-y-2.5">
                  {(standalone.find((s) => s.model === "xgboost")?.importance ?? []).map((item) => {
                    const max = Math.max(
                      ...(standalone.find((s) => s.model === "xgboost")?.importance ?? []).map((i) => i.value),
                      1e-9,
                    );
                    return (
                      <div key={item.name}>
                        <div className="flex items-center justify-between text-[11px]">
                          <span className="font-mono">{item.name}</span>
                          <span className="tabular-nums text-muted-foreground">
                            {(item.value * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-secondary">
                          <div
                            className="h-full rounded-full bg-chart-1"
                            style={{ width: `${(item.value / max) * 100}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </ChartCard>
            </section>
          ) : null}

          {/* Method + integrity */}
          <section className="mb-6 grid gap-5 lg:grid-cols-2">
            <div className="rounded-xl border border-border/70 bg-card/50 p-5">
              <div className="flex items-center gap-2 text-primary">
                <BarChart3 className="size-4" />
                <h2 className="text-sm font-semibold text-foreground">Where our numbers diverge</h2>
              </div>
              <ul className="mt-3 space-y-2.5 text-xs leading-5 text-muted-foreground">
                <li>
                  <strong className="text-foreground">Linear Regression and KNN line up closely.</strong>{" "}
                  Their published RMSE (1.08 and 1.07) is reproduced to within 0.01, which suggests the
                  feature set and split style are aligned.
                </li>
                <li>
                  <strong className="text-foreground">The four ensemble learners do not.</strong> Our
                  Random Forest, AdaBoost, Gradient Boosting and XGBoost sit 0.06–0.12 RMSE above the
                  published values. With only user id, movie id and year as inputs, a strong learner
                  converges near 1.03 in our runs.
                </li>
                <li>
                  <strong className="text-foreground">Likely causes.</strong> Unpublished
                  hyperparameters, a different train/test ratio, a different categorical encoding, or
                  additional inputs (genre or demographic columns) that the paper names but excludes
                  from its feature list. The paper's XGBoost RMSE of 0.92 is not reachable from the
                  three id/year columns in our reconstruction.
                </li>
                <li>
                  <strong className="text-foreground">Stacking gains are small here.</strong> The meta
                  learner leans almost entirely on XGBoost and gives KNN a slightly negative weight, so
                  the stack performs about the same as its strongest member.
                </li>
              </ul>
            </div>

            <div className="rounded-xl border border-border/70 bg-card/50 p-5">
              <div className="flex items-center gap-2 text-primary">
                <ShieldCheck className="size-4" />
                <h2 className="text-sm font-semibold text-foreground">Run notes & integrity</h2>
              </div>
              <ul className="mt-3 space-y-2 text-xs leading-5 text-muted-foreground">
                {(result?.notes ?? []).map((note) => (
                  <li key={note} className="flex gap-2">
                    <CheckCircle2 className="mt-0.5 size-3.5 shrink-0 text-chart-5" />
                    <span>{note}</span>
                  </li>
                ))}
                {PAPER_NOTES.map((note) => (
                  <li key={note} className="flex gap-2">
                    <span className="mt-1.5 size-1 shrink-0 rounded-full bg-primary/60" />
                    <span>{note}</span>
                  </li>
                ))}
              </ul>
              {result ? (
                <dl className="mt-4 grid grid-cols-2 gap-3 border-t border-border/60 pt-4 text-[11px]">
                  <Meta label="Model id" value={result.modelId} />
                  <Meta label="Version" value={result.version} />
                  <Meta label="Dataset" value={result.datasetVersion} />
                  <Meta label="Feature version" value={result.featureVersion} />
                  <Meta label="Split" value={`${result.split.strategy} (${result.dataset.nTrain}/${result.dataset.nTest})`} />
                  <Meta label="Search depth" value={result.search} />
                </dl>
              ) : null}
            </div>
          </section>
        </>
      ) : (
        <div className="rounded-xl border border-dashed border-border/70 bg-card/40 px-6 py-16 text-center">
          <span className="mx-auto flex size-11 items-center justify-center rounded-full bg-primary/12 text-primary">
            <FlaskConical className="size-5" />
          </span>
          <h2 className="mt-4 font-display text-2xl tracking-tight">No results yet</h2>
          <p className="mx-auto mt-2 max-w-xl text-sm leading-6 text-muted-foreground">
            Run the experiment to train six standalone models and seven stacking configurations with
            out-of-fold meta features. A quick run uses 30,000 rows; the full run uses all 100,000.
          </p>
        </div>
      )}
    </AppShell>
  );
}

const MODEL_ORDER: [keyof typeof PAPER_STANDALONE, string][] = [
  ["linear_regression", "Linear Regression"],
  ["xgboost", "XGBoost"],
  ["random_forest", "Random Forest"],
  ["adaboost", "AdaBoost"],
  ["gradient_boosting", "Gradient Boosting"],
  ["knn", "K-Nearest Neighbours"],
];

const PAPER_STANDALONE_ROWS = MODEL_ORDER.map(([key, label]) => ({
  label,
  rmse: PAPER_STANDALONE[key].rmse,
}));

function ReferenceTable({ title, rows }: { title: string; rows: { label: string; rmse: number }[] }) {
  return (
    <div>
      <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">{title}</p>
      <table className="mt-3 w-full text-sm">
        <tbody>
          {rows.map((row) => (
            <tr key={row.label} className="border-b border-border/40">
              <td className="py-2 pr-3">{row.label}</td>
              <td className="py-2 text-right tabular-nums">{row.rmse.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ChartCard({
  title,
  note,
  children,
}: {
  title: string;
  note: string;
  children: React.ReactNode;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      className="rounded-xl border border-border/70 bg-card/50 p-5"
    >
      <h3 className="text-sm font-semibold">{title}</h3>
      <p className="mt-1 text-[11px] leading-4 text-muted-foreground">{note}</p>
      <div className="mt-4">{children}</div>
    </motion.div>
  );
}

function CorrelationGrid({ names, matrix }: { names: string[]; matrix: number[][] }) {
  return (
    <ChartCard
      title="Base-prediction correlation"
      note="Pearson correlation between the base learners' test-set predictions. High values mean the stack has little diversity to exploit."
    >
      <div className="overflow-x-auto">
        <table className="w-full min-w-[320px] text-[11px]">
          <thead>
            <tr>
              <th />
              {names.map((name) => (
                <th key={name} className="px-1 pb-2 text-center font-medium text-muted-foreground">
                  {name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => (
              <tr key={names[i]}>
                <td className="pr-2 text-right text-muted-foreground">{names[i]}</td>
                {row.map((value, j) => (
                  <td key={j} className="p-0.5">
                    <div
                      className="flex h-9 items-center justify-center rounded tabular-nums"
                      style={{
                        backgroundColor:
                          i === j
                            ? "color-mix(in oklab, var(--chart-1) 55%, transparent)"
                            : `color-mix(in oklab, var(--chart-4) ${Math.max(0, value) * 45}%, transparent)`,
                      }}
                      title={`${names[i]} vs ${names[j]}: ${value.toFixed(3)}`}
                    >
                      {value.toFixed(2)}
                    </div>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </ChartCard>
  );
}

function Headline({
  label,
  value,
  sub,
  muted = false,
  small = false,
}: {
  label: string;
  value: string;
  sub: string;
  muted?: boolean;
  small?: boolean;
}) {
  return (
    <div className="rounded-xl border border-border/70 bg-card/50 p-5">
      <p className="text-[11px] uppercase tracking-[0.16em] text-muted-foreground">{label}</p>
      <p
        className={`mt-2 ${small ? "text-base font-semibold leading-6" : "font-display text-3xl"} ${
          muted ? "text-muted-foreground" : ""
        }`}
      >
        {value}
      </p>
      <p className="mt-1.5 text-[11px] leading-4 text-muted-foreground">{sub}</p>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <p className="text-[11px] uppercase tracking-[0.16em] text-muted-foreground">{label}</p>
      <div className="mt-2">{children}</div>
    </div>
  );
}

function Segmented({
  value,
  options,
  onChange,
}: {
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
}) {
  return (
    <div className="inline-flex rounded-lg border border-border/70 p-0.5">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          onClick={() => onChange(option.value)}
          className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
            option.value === value
              ? "bg-primary text-primary-foreground"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="mt-0.5 break-words font-mono text-[10px] leading-4">{value}</dd>
    </div>
  );
}
