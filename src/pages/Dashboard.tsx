import { AppShell } from "@/components/AppShell";
import { MoviePoster } from "@/components/MoviePoster";
import { useMl } from "@/components/MlProvider";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { api } from "@/convex/_generated/api";
import { useAuth } from "@/hooks/use-auth";
import { ratingTone } from "@/lib/catalog";
import type { RecommendOutput } from "@/ml/recommender/engine";
import { DEFAULT_RECOMMEND_CONFIG } from "@/ml/recommender/engine";
import type { TasteProfile } from "@/ml/recommender/profile";
import { useQuery } from "convex/react";
import { motion } from "framer-motion";
import {
  ArrowRight,
  Brain,
  Film,
  Loader2,
  Shuffle,
  Sparkles,
  Star,
  TrendingUp,
  Users,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router";

export default function Dashboard() {
  const { user } = useAuth();
  const ml = useMl();
  const myRatings = useQuery(api.recommender.myRatings);
  const latestRun = useQuery(api.recommender.latestRun);
  const [profile, setProfile] = useState<TasteProfile | null>(null);
  const [preview, setPreview] = useState<RecommendOutput | null>(null);

  const ratings = useMemo(
    () => (myRatings ?? []).map((row) => ({ movieId: row.movieId, rating: row.rating })),
    [myRatings],
  );

  const fetchProfile = ml.profile;
  const requestRecommendations = ml.recommendations;
  const hasTrainedModels = ml.hasTrainedModels;

  useEffect(() => {
    if (!ml.ready || ratings.length === 0) return;
    let cancelled = false;
    fetchProfile(ratings)
      .then((result) => {
        if (!cancelled) setProfile(result);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [ratings, ml.ready, fetchProfile]);

  useEffect(() => {
    if (!hasTrainedModels || ratings.length === 0) return;
    let cancelled = false;
    requestRecommendations(ratings, { ...DEFAULT_RECOMMEND_CONFIG, topK: 3 })
      .then((result) => {
        if (!cancelled) setPreview(result);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [hasTrainedModels, ratings, requestRecommendations]);

  const ratedCount = ratings.length;
  const totalFilms = ml.meta?.movies.length ?? 1682;
  // Derived views: clearing every rating must clear the panels too, without
  // pushing null into state from an effect.
  const profileView = ratedCount === 0 ? null : profile;
  const previewView = ratedCount === 0 || !hasTrainedModels ? null : preview;

  return (
    <AppShell
      title={`Welcome back${user?.name ? `, ${user.name}` : ""}`}
      description="Your ratings, your persona match and the state of the stacking ensemble — all in one place."
      actions={
        <Button asChild className="gap-2">
          <Link to="/recommendations">
            <Sparkles className="size-4" />
            Open my Top-K
          </Link>
        </Button>
      }
    >
      {ml.loading ? (
        <div className="flex items-center gap-3 rounded-xl border border-border/70 bg-card/50 p-6 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" />
          Parsing MovieLens 100K…
        </div>
      ) : (
        <div className="flex flex-col gap-6">
          <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Metric
              icon={<Star className="size-4" />}
              label="Films rated"
              value={String(ratedCount)}
              hint={`of ${totalFilms.toLocaleString()} in the catalogue`}
            />
            <Metric
              icon={<Users className="size-4" />}
              label="Persona match"
              value={
                profileView?.persona ? `User ${profileView.persona.userId}` : "Not matched"
              }
              hint={
                profileView?.persona
                  ? `cosine ${profileView.persona.similarity.toFixed(2)} · ${profileView.persona.sharedMovies} shared films`
                  : "8 co-rated films required"
              }
            />
            <Metric
              icon={<Brain className="size-4" />}
              label="Ensemble"
              value={ml.hasTrainedModels ? "Trained" : "Not trained"}
              hint={
                ml.result
                  ? `proposed stack RMSE ${ml.result.proposed.ourRmse.toFixed(4)} this session`
                  : latestRun
                    ? `last run RMSE ${Number(latestRun.metrics?.proposed?.ourRmse ?? 0).toFixed(4)}`
                    : "run the experiment to serve predictions"
              }
            />
            <Metric
              icon={<TrendingUp className="size-4" />}
              label="Your mean rating"
              value={profileView ? profileView.meanRating.toFixed(2) : "—"}
              hint={
                profileView
                  ? `dataset mean ${profileView.globalMean.toFixed(2)}`
                  : "rate a few films"
              }
            />
          </section>

          <section className="grid gap-4 lg:grid-cols-3">
            <ActionCard
              to="/rate"
              icon={<Star className="size-4" />}
              title="Rate more films"
              body="Every rating sharpens the persona match and the content-affinity layer."
              cta={ratedCount === 0 ? "Start rating" : "Add ratings"}
            />
            <ActionCard
              to="/analytics"
              icon={<Brain className="size-4" />}
              title={ml.hasTrainedModels ? "Inspect the ensemble" : "Train the ensemble"}
              body="Six standalone models, seven stacking configurations, out-of-fold meta features."
              cta={ml.hasTrainedModels ? "Open analytics" : "Run experiment"}
            />
            <ActionCard
              to="/browse"
              icon={<Film className="size-4" />}
              title="Browse the catalogue"
              body="Filter 1,682 films by genre and see predicted ratings once the stack is trained."
              cta="Open catalogue"
            />
          </section>

          <section className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_340px]">
            <div className="rounded-xl border border-border/70 bg-card/50 p-5">
              <div className="flex items-center justify-between gap-3">
                <h2 className="text-sm font-semibold">Top-3 preview</h2>
                {previewView ? (
                  <Badge
                    variant="outline"
                    className={
                      previewView.source === "stacking_model"
                        ? "border-primary/50 text-primary"
                        : "border-chart-3/50 text-chart-3"
                    }
                  >
                    {previewView.sourceLabel}
                  </Badge>
                ) : null}
              </div>

              {ratedCount === 0 ? (
                <p className="mt-3 text-xs leading-5 text-muted-foreground">
                  No ratings yet, so there is nothing personalised to show. Rate a few films and this
                  panel fills in.
                </p>
              ) : !ml.hasTrainedModels ? (
                <p className="mt-3 text-xs leading-5 text-muted-foreground">
                  The ensemble is not resident in this session. Open the analytics page and run the
                  experiment — roughly 50 seconds on the full dataset.
                </p>
              ) : previewView === null ? (
                <p className="mt-3 flex items-center gap-2 text-xs text-muted-foreground">
                  <Loader2 className="size-3.5 animate-spin" />
                  Scoring the catalogue…
                </p>
              ) : (
                <div className="mt-4 grid gap-4 sm:grid-cols-3">
                  {previewView.recommendations.map((rec) => (
                    <motion.div
                      key={rec.movieId}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="flex flex-col gap-2"
                    >
                      <MoviePoster
                        movieId={rec.movieId}
                        title={rec.title}
                        year={rec.year}
                        className="aspect-[2/3] w-full"
                      />
                      <p className="truncate text-xs font-medium">{rec.title}</p>
                      <p className={`text-xs tabular-nums ${ratingTone(rec.predictedRating)}`}>
                        {rec.predictedRating === null
                          ? "fallback score"
                          : `${rec.predictedRating.toFixed(2)} predicted`}
                      </p>
                    </motion.div>
                  ))}
                </div>
              )}

              <p className="mt-4 border-t border-border/60 pt-3 text-[11px] leading-4 text-muted-foreground">
                Preview uses {DEFAULT_RECOMMEND_CONFIG.mode} mode with affinity weight{" "}
                {DEFAULT_RECOMMEND_CONFIG.affinityWeight.toFixed(2)}. Adjust ranking in the Top-K page.
              </p>
            </div>

            <aside className="flex flex-col gap-4">
              <div className="rounded-xl border border-border/70 bg-card/50 p-5">
                <div className="flex items-center gap-2 text-primary">
                  <Shuffle className="size-4" />
                  <h2 className="text-sm font-semibold text-foreground">How the pipeline runs</h2>
                </div>
                <ol className="mt-3 space-y-2.5 text-xs text-muted-foreground">
                  {[
                    "Your ratings are stored in Convex and form the taste profile.",
                    "A cosine match maps you onto the closest MovieLens 100K user.",
                    "KNN, XGBoost and Gradient Boosting score the unseen catalogue.",
                    "A Linear Regression meta-learner combines those three predictions.",
                    "MMR re-ranking and content affinity are optional enhancements.",
                  ].map((line, index) => (
                    <li key={line} className="flex gap-2.5">
                      <span className="flex size-4 shrink-0 items-center justify-center rounded-full bg-primary/15 text-[10px] font-semibold text-primary">
                        {index + 1}
                      </span>
                      {line}
                    </li>
                  ))}
                </ol>
              </div>

              {profileView && profileView.genreAffinity.length > 0 ? (
                <div className="rounded-xl border border-border/70 bg-card/50 p-5">
                  <h2 className="text-sm font-semibold">Strongest signals</h2>
                  <ul className="mt-3 space-y-2 text-xs">
                    {profileView.genreAffinity.slice(0, 5).map((entry) => (
                      <li key={entry.genre} className="flex items-center justify-between gap-3">
                        <span>{entry.genre}</span>
                        <span
                          className={`tabular-nums ${entry.lift >= 0 ? "text-chart-5" : "text-chart-3"}`}
                        >
                          {entry.lift >= 0 ? "+" : ""}
                          {entry.lift.toFixed(2)} over {entry.count}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </aside>
          </section>
        </div>
      )}
    </AppShell>
  );
}

function Metric({
  icon,
  label,
  value,
  hint,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  hint: string;
}) {
  return (
    <div className="rounded-xl border border-border/70 bg-card/50 p-5">
      <div className="flex items-center gap-2 text-primary">{icon}</div>
      <p className="mt-3 text-[11px] uppercase tracking-[0.16em] text-muted-foreground">{label}</p>
      <p className="mt-1 font-display text-2xl">{value}</p>
      <p className="mt-1 text-[11px] leading-4 text-muted-foreground">{hint}</p>
    </div>
  );
}

function ActionCard({
  to,
  icon,
  title,
  body,
  cta,
}: {
  to: string;
  icon: React.ReactNode;
  title: string;
  body: string;
  cta: string;
}) {
  return (
    <Link
      to={to}
      className="group flex flex-col justify-between rounded-xl border border-border/70 bg-card/50 p-5 transition-colors hover:border-primary/40"
    >
      <div>
        <div className="flex items-center gap-2 text-primary">{icon}</div>
        <h2 className="mt-3 text-sm font-semibold">{title}</h2>
        <p className="mt-1.5 text-xs leading-5 text-muted-foreground">{body}</p>
      </div>
      <span className="mt-4 flex items-center gap-1.5 text-xs font-medium text-primary">
        {cta}
        <ArrowRight className="size-3.5 transition-transform group-hover:translate-x-0.5" />
      </span>
    </Link>
  );
}
