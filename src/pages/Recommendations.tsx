import { AppShell } from "@/components/AppShell";
import { MoviePoster } from "@/components/MoviePoster";
import { useMl } from "@/components/MlProvider";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { api } from "@/convex/_generated/api";
import { ratingTone } from "@/lib/catalog";
import type { RecommendConfig, RecommendOutput, Recommendation } from "@/ml/recommender/engine";
import { DEFAULT_RECOMMEND_CONFIG } from "@/ml/recommender/engine";
import { useMutation, useQuery } from "convex/react";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  BarChart3,
  Brain,
  Info,
  Loader2,
  RefreshCw,
  Shuffle,
  Sparkles,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router";
import { toast } from "sonner";

const K_OPTIONS = [5, 10, 20, 50];

export default function Recommendations() {
  const ml = useMl();
  const myRatings = useQuery(api.recommender.myRatings);
  const savedProfile = useQuery(api.recommender.myProfile);
  const saveProfile = useMutation(api.recommender.saveProfile);

  const [config, setConfig] = useState<RecommendConfig>(DEFAULT_RECOMMEND_CONFIG);
  const [output, setOutput] = useState<RecommendOutput | null>(null);
  const [busy, setBusy] = useState(false);
  const requestId = useRef(0);

  useEffect(() => {
    if (!savedProfile) return;
    setConfig((prev) => ({
      ...prev,
      topK: savedProfile.topK || prev.topK,
      mode: savedProfile.mode === "diversity" ? "diversity" : "accuracy",
      diversityLambda: savedProfile.diversityLambda,
      affinityWeight: savedProfile.affinityWeight,
    }));
  }, [savedProfile]);

  const ratings = useMemo(
    () => (myRatings ?? []).map((row) => ({ movieId: row.movieId, rating: row.rating })),
    [myRatings],
  );

  const { recommendations: requestRecommendations, hasTrainedModels } = ml;

  const run = useCallback(
    async (next: RecommendConfig) => {
      if (ratings.length === 0 || !hasTrainedModels) return;
      const id = ++requestId.current;
      setBusy(true);
      try {
        const result = await requestRecommendations(ratings, next);
        if (id === requestId.current) setOutput(result);
      } catch (cause: unknown) {
        toast.error(cause instanceof Error ? cause.message : "Recommendation request failed.");
      } finally {
        if (id === requestId.current) setBusy(false);
      }
    },
    [requestRecommendations, hasTrainedModels, ratings],
  );

  useEffect(() => {
    void run(config);
  }, [config, run]);

  const updateConfig = (patch: Partial<RecommendConfig>) => {
    const next = { ...config, ...patch };
    setConfig(next);
    const timer = window.setTimeout(() => {
      saveProfile({
        topK: next.topK,
        mode: next.mode,
        diversityLambda: next.diversityLambda,
        affinityWeight: next.affinityWeight,
      }).catch(() => undefined);
    }, 500);
    return () => window.clearTimeout(timer);
  };

  const trainNow = async () => {
    try {
      await ml.train({
        mode: "paper",
        scope: "full",
        sampleSize: 30000,
        randomSeed: 42,
        testSize: 0.2,
        cvFolds: 5,
        split: "random",
      });
      toast.success("Ensemble trained — serving predictions from the fitted stack.");
    } catch (cause: unknown) {
      toast.error(cause instanceof Error ? cause.message : "Training failed.");
    }
  };

  const needsRatings = (myRatings?.length ?? 0) === 0;

  return (
    <AppShell
      title="Top-K picks"
      description="Ranked by the paper's stacking ensemble: KNN + XGBoost + Gradient Boosting feed a Linear Regression meta-learner. Accuracy and diversity modes re-rank the same predictions differently."
      actions={
        <Button variant="outline" className="gap-2" onClick={() => run(config)} disabled={busy}>
          {busy ? <Loader2 className="size-4 animate-spin" /> : <RefreshCw className="size-4" />}
          Recompute
        </Button>
      }
    >
      <div className="mb-6 grid gap-4 rounded-xl border border-border/70 bg-card/50 p-5 lg:grid-cols-[auto_1fr]">
        <div className="flex flex-col gap-4">
          <div>
            <p className="text-[11px] uppercase tracking-[0.16em] text-muted-foreground">List size</p>
            <div className="mt-2 flex gap-1.5">
              {K_OPTIONS.map((k) => (
                <Button
                  key={k}
                  size="sm"
                  variant={config.topK === k ? "default" : "outline"}
                  onClick={() => updateConfig({ topK: k })}
                >
                  {k}
                </Button>
              ))}
            </div>
          </div>
          <div>
            <p className="text-[11px] uppercase tracking-[0.16em] text-muted-foreground">Ranking</p>
            <div className="mt-2 flex gap-1.5">
              <Button
                size="sm"
                variant={config.mode === "accuracy" ? "default" : "outline"}
                className="gap-1.5"
                onClick={() => updateConfig({ mode: "accuracy" })}
              >
                <BarChart3 className="size-3.5" />
                Accuracy
              </Button>
              <Button
                size="sm"
                variant={config.mode === "diversity" ? "default" : "outline"}
                className="gap-1.5"
                onClick={() => updateConfig({ mode: "diversity" })}
              >
                <Shuffle className="size-3.5" />
                Diversity
              </Button>
            </div>
          </div>
        </div>

        <div className="grid gap-5 sm:grid-cols-2">
          <div>
            <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.16em] text-muted-foreground">
              <span>MMR λ</span>
              <span className="tabular-nums">{config.diversityLambda.toFixed(2)}</span>
            </div>
            <Slider
              className="mt-3"
              value={[config.diversityLambda]}
              min={0}
              max={1}
              step={0.05}
              disabled={config.mode !== "diversity"}
              onValueChange={([value]) => updateConfig({ diversityLambda: value })}
            />
            <p className="mt-2 text-[11px] leading-4 text-muted-foreground">
              1.00 = pure relevance, 0.00 = pure diversity. Only applied in diversity mode.
            </p>
          </div>
          <div>
            <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.16em] text-muted-foreground">
              <span>Affinity weight</span>
              <span className="tabular-nums">{config.affinityWeight.toFixed(2)}</span>
            </div>
            <Slider
              className="mt-3"
              value={[config.affinityWeight]}
              min={0}
              max={1}
              step={0.05}
              onValueChange={([value]) => updateConfig({ affinityWeight: value })}
            />
            <p className="mt-2 text-[11px] leading-4 text-muted-foreground">
              Set to 0.00 for the pure paper model. Above 0 blends in your own genre history — an
              enhancement, not something the paper does.
            </p>
          </div>
        </div>
      </div>

      {needsRatings ? (
        <EmptyState
          icon={<Sparkles className="size-5" />}
          title="Rate a few films first"
          body="The engine needs your ratings to map this account onto the MovieLens user space and to filter out everything you have already seen."
          action={
            <Button asChild className="gap-2">
              <Link to="/rate">Go to rating</Link>
            </Button>
          }
        />
      ) : !ml.hasTrainedModels ? (
        <EmptyState
          icon={<Brain className="size-5" />}
          title="No trained ensemble in this session"
          body="Training runs offline in a web worker: six standalone models plus seven stacking configurations, roughly 50 seconds on the full dataset. Nothing is retrained per request."
          action={
            <div className="flex flex-col items-center gap-3">
              <Button className="gap-2" onClick={trainNow} disabled={ml.training}>
                {ml.training ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <Brain className="size-4" />
                )}
                {ml.training ? "Training…" : "Train the ensemble now"}
              </Button>
              {ml.progress ? (
                <p className="text-xs text-muted-foreground">
                  {ml.progress.message} · {ml.progress.pct.toFixed(0)}%
                </p>
              ) : (
                <Link to="/analytics" className="text-xs text-primary underline">
                  Or configure the run first
                </Link>
              )}
            </div>
          }
        />
      ) : output === null ? (
        <div className="flex items-center gap-3 rounded-xl border border-border/70 bg-card/50 p-6 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" />
          Scoring the catalogue…
        </div>
      ) : (
        <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_330px]">
          <div className="flex flex-col gap-4">
            <div className="flex flex-wrap items-center gap-3">
              <Badge
                variant="outline"
                className={
                  output.source === "stacking_model"
                    ? "border-primary/50 text-primary"
                    : "border-chart-3/50 text-chart-3"
                }
              >
                {output.sourceLabel}
              </Badge>
              <span className="text-xs text-muted-foreground">
                {output.recommendations.length} films · scored from{" "}
                {output.diagnostics.candidatesConsidered.toLocaleString()} candidates
              </span>
            </div>

            {output.recommendations.map((rec, index) => (
              <RecommendationCard key={rec.movieId} rec={rec} index={index} />
            ))}
          </div>

          <aside className="flex flex-col gap-4 xl:sticky xl:top-6 xl:self-start">
            <div className="rounded-xl border border-border/70 bg-card/60 p-5">
              <h2 className="text-sm font-semibold">How this list was built</h2>
              <ul className="mt-3 space-y-2.5 text-xs">
                <Diagnostic
                  label="Persona"
                  value={
                    output.diagnostics.personaUserId === null
                      ? "none matched"
                      : `MovieLens user ${output.diagnostics.personaUserId}`
                  }
                />
                <Diagnostic
                  label="Match strength"
                  value={
                    output.diagnostics.personaSimilarity === null
                      ? "—"
                      : `cosine ${output.diagnostics.personaSimilarity.toFixed(2)} over ${output.diagnostics.personaSharedMovies} films`
                  }
                />
                <Diagnostic
                  label="Excluded: already rated"
                  value={String(output.diagnostics.excludedAlreadyRated)}
                />
                <Diagnostic
                  label="Excluded: persona history"
                  value={String(output.diagnostics.excludedPersonaHistory)}
                />
                <Diagnostic
                  label="Mean predicted rating"
                  value={
                    output.diagnostics.meanPredictedRating === null
                      ? "n/a (fallback)"
                      : output.diagnostics.meanPredictedRating.toFixed(2)
                  }
                />
                <Diagnostic
                  label="Mean learner spread"
                  value={
                    output.diagnostics.meanConfidenceSpread === null
                      ? "n/a"
                      : `±${output.diagnostics.meanConfidenceSpread.toFixed(2)}`
                  }
                />
                <Diagnostic
                  label="Distinct genres"
                  value={String(output.diagnostics.distinctGenres)}
                />
                <Diagnostic
                  label="Intra-list similarity"
                  value={output.diagnostics.intraListSimilarity.toFixed(3)}
                />
              </ul>
              <p className="mt-4 border-t border-border/60 pt-3 text-[11px] leading-4 text-muted-foreground">
                {output.diagnostics.rationale}
              </p>
            </div>

            <div className="rounded-xl border border-border/70 bg-card/60 p-5">
              <div className="flex items-center gap-2 text-muted-foreground">
                <Info className="size-3.5" />
                <h2 className="text-sm font-semibold text-foreground">Reading the numbers</h2>
              </div>
              <p className="mt-2 text-[11px] leading-5 text-muted-foreground">
                Predicted rating is the meta-learner's output. Spread is the disagreement between
                KNN, XGBoost and Gradient Boosting — it is not a probability that you will like the
                film. Support counts come from MovieLens 100K and describe the dataset, not you.
              </p>
            </div>
          </aside>
        </div>
      )}
    </AppShell>
  );
}

function RecommendationCard({ rec, index }: { rec: Recommendation; index: number }) {
  const fallback = rec.source !== "stacking_model";
  return (
    <motion.article
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: Math.min(index * 0.04, 0.4) }}
      className="rounded-xl border border-border/70 bg-card/50 p-4 sm:p-5"
    >
      <div className="flex flex-col gap-4 sm:flex-row">
        <div className="flex gap-4">
          <span className="font-display text-2xl leading-none text-muted-foreground/70">
            {String(rec.rank).padStart(2, "0")}
          </span>
          <MoviePoster
            movieId={rec.movieId}
            title={rec.title}
            year={rec.year}
            className="aspect-[2/3] w-20 shrink-0 sm:w-24"
          />
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0">
              <h3 className="text-base font-semibold leading-tight">{rec.title}</h3>
              <p className="mt-1 text-xs text-muted-foreground">
                {rec.year ?? "year unknown"} ·{" "}
                {rec.genres.filter((g) => g !== "unknown").join(", ") || "genre unlisted"}
              </p>
            </div>
            <div className="text-right">
              {rec.predictedRating === null ? (
                <p className="text-xs text-chart-3">fallback score</p>
              ) : (
                <>
                  <p className={`font-display text-3xl leading-none ${ratingTone(rec.predictedRating)}`}>
                    {rec.predictedRating.toFixed(2)}
                  </p>
                  <p className="text-[11px] text-muted-foreground">predicted rating</p>
                </>
              )}
              <p className="mt-1 text-[11px] tabular-nums text-muted-foreground">
                relevance {rec.relevance.toFixed(2)}
              </p>
            </div>
          </div>

          {rec.basePredictions.length > 0 ? (
            <div className="mt-3 flex flex-wrap gap-2">
              {rec.basePredictions.map((base) => (
                <span
                  key={base.name}
                  className="rounded-md border border-border/70 bg-background/40 px-2 py-1 text-[11px] tabular-nums"
                >
                  {base.name} <span className="text-muted-foreground">{base.value.toFixed(2)}</span>
                </span>
              ))}
              <span className="rounded-md border border-border/70 bg-background/40 px-2 py-1 text-[11px]">
                spread ±{rec.confidence.spread.toFixed(2)} · {rec.confidence.label}
              </span>
              {rec.metaRecipe ? (
                <span className="rounded-md border border-primary/30 bg-primary/10 px-2 py-1 text-[11px] text-primary">
                  {rec.metaRecipe}
                </span>
              ) : null}
            </div>
          ) : null}

          <div className="mt-4 space-y-1.5 border-t border-border/60 pt-3">
            {rec.explanation.modelDerived ? (
              <p className="flex gap-2 text-xs leading-5 text-foreground/90">
                <Brain className="mt-0.5 size-3.5 shrink-0 text-primary" />
                <span>{rec.explanation.modelDerived}</span>
              </p>
            ) : null}
            {rec.explanation.evidence.map((line) => (
              <p key={line} className="flex gap-2 text-xs leading-5 text-muted-foreground">
                <span className="mt-1.5 size-1 shrink-0 rounded-full bg-primary/60" />
                <span>{line}</span>
              </p>
            ))}
            {rec.explanation.caveat ? (
              <p className="flex gap-2 text-xs leading-5 text-chart-3">
                <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
                <span>{rec.explanation.caveat}</span>
              </p>
            ) : null}
            {!fallback && rec.diversityPenalty !== null ? (
              <p className="text-[11px] text-muted-foreground">
                MMR similarity penalty at selection: {rec.diversityPenalty.toFixed(3)}
              </p>
            ) : null}
          </div>
        </div>
      </div>
    </motion.article>
  );
}

function Diagnostic({ label, value }: { label: string; value: string }) {
  return (
    <li className="flex items-baseline justify-between gap-3">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-right tabular-nums">{value}</span>
    </li>
  );
}

function EmptyState({
  icon,
  title,
  body,
  action,
}: {
  icon: React.ReactNode;
  title: string;
  body: string;
  action: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center gap-4 rounded-xl border border-dashed border-border/70 bg-card/40 px-6 py-14 text-center">
      <span className="flex size-11 items-center justify-center rounded-full bg-primary/12 text-primary">
        {icon}
      </span>
      <h2 className="font-display text-2xl tracking-tight">{title}</h2>
      <p className="max-w-lg text-sm leading-6 text-muted-foreground">{body}</p>
      {action}
    </div>
  );
}
