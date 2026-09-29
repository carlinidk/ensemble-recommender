import { AppShell } from "@/components/AppShell";
import { MoviePoster } from "@/components/MoviePoster";
import { useMl } from "@/components/MlProvider";
import { StarRating } from "@/components/StarRating";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { api } from "@/convex/_generated/api";
import { genreList, searchMovies, starterSelection, statMap } from "@/lib/catalog";
import type { TasteProfile } from "@/ml/recommender/profile";
import { useMutation, useQuery } from "convex/react";
import { motion } from "framer-motion";
import { Loader2, Search, Sparkles, Target, Trash2, UserRoundSearch } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router";
import { toast } from "sonner";

const PERSONA_MIN_SHARED = 8;
const STARTER_COUNT = 18;

export default function Rate() {
  const ml = useMl();
  const myRatings = useQuery(api.recommender.myRatings);
  const rateMovie = useMutation(api.recommender.rateMovie);
  const clearRating = useMutation(api.recommender.clearMovieRating);

  const [query, setQuery] = useState("");
  const [profile, setProfile] = useState<TasteProfile | null>(null);
  // Local edits layered over the reactive `myRatings` query, so a tap never waits
  // on a round trip and the stored value wins once Convex pushes it back.
  const [edits, setEdits] = useState<Record<number, number>>({});

  const saved = useMemo(
    () => Object.fromEntries((myRatings ?? []).map((row) => [row.movieId, row.rating])),
    [myRatings],
  );
  /** 0 marks a removed rating; everything else is a 1-5 score. */
  const draft = useMemo<Record<number, number>>(() => {
    const merged: Record<number, number> = { ...saved };
    for (const [id, value] of Object.entries(edits)) {
      const movieId = Number(id);
      if (value === 0) delete merged[movieId];
      else merged[movieId] = value;
    }
    return merged;
  }, [saved, edits]);

  const stats = useMemo(
    () => statMap(ml.meta?.movieStats ?? []),
    [ml.meta?.movieStats],
  );
  const movies = useMemo(() => ml.meta?.movies ?? [], [ml.meta]);
  const moviesById = useMemo(() => new Map(movies.map((m) => [m.id, m])), [movies]);
  const genres = useMemo(() => genreList(movies), [movies]);

  const starter = useMemo(
    () => (ml.meta ? starterSelection(movies, stats, STARTER_COUNT) : []),
    [ml.meta, movies, stats],
  );
  const results = useMemo(() => searchMovies(movies, query), [movies, query]);

  const ratedIds = Object.keys(draft).map(Number);
  const ratingCount = ratedIds.length;

  // Profile refresh is debounced so a click never waits on the worker. The state
  // update happens inside the debounced promise callback rather than in the
  // effect body, so it cannot cascade renders.
  useEffect(() => {
    const ratings = Object.entries(draft).map(([movieId, rating]) => ({
      movieId: Number(movieId),
      rating,
    }));
    if (!ml.ready || ratings.length === 0) return;
    const fetchProfile = ml.profile;
    const timer = window.setTimeout(() => {
      fetchProfile(ratings)
        .then(setProfile)
        .catch(() => undefined);
    }, 400);
    return () => window.clearTimeout(timer);
    // `ml.profile` is a stable callback; depending on the context object itself
    // would re-fire on every provider render.
  }, [draft, ml.ready, ml.profile]);

  const handleRate = (movieId: number, rating: number) => {
    setEdits((prev) => ({ ...prev, [movieId]: rating }));
    rateMovie({ movieId, rating }).catch((error: unknown) => {
      toast.error(error instanceof Error ? error.message : "Could not save that rating.");
    });
  };

  const handleClear = (movieId: number) => {
    setEdits((prev) => ({ ...prev, [movieId]: 0 }));
    clearRating({ movieId }).catch(() => toast.error("Could not remove that rating."));
  };

  // Persona matching is a hard requirement, not a smooth score: the bar shows how
  // close the best candidate is to the minimum number of shared films.
  const activeProfile = ratingCount === 0 ? null : profile;
  const closestShared = activeProfile?.personaCandidates[0]?.sharedMovies ?? 0;
  const personaProgress = activeProfile?.persona
    ? 100
    : Math.min(100, Math.round((closestShared / PERSONA_MIN_SHARED) * 100));

  return (
    <AppShell
      title="Rate films"
      description="Your ratings are the only input the recommender has about you. They map this account onto the MovieLens 100K user space and drive the content-affinity layer."
      actions={
        <Button asChild className="gap-2">
          <Link to="/recommendations">
            <Sparkles className="size-4" />
            Get my Top-K
          </Link>
        </Button>
      }
    >
      {ml.loading ? (
        <LoadingCard label="Parsing MovieLens 100K…" />
      ) : (
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
          <div className="flex flex-col gap-6">
            <section className="rounded-xl border border-border/70 bg-card/50 p-5">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h2 className="text-sm font-semibold">Starter set</h2>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Popular films across {Math.min(genres.length, 18)} genres — enough to map your
                    taste without rating the whole catalogue.
                  </p>
                </div>
                <Badge variant="outline" className="border-border/70 tabular-nums">
                  {ratingCount} rated
                </Badge>
              </div>

              <div className="mt-5 grid grid-cols-2 gap-4 sm:grid-cols-3">
                {starter.map((movie, index) => (
                  <RatingCard
                    key={movie.id}
                    index={index}
                    movieId={movie.id}
                    title={movie.title}
                    year={movie.year}
                    genres={movie.genres}
                    support={stats.get(movie.id)?.count ?? 0}
                    value={draft[movie.id] ?? null}
                    onRate={handleRate}
                  />
                ))}
              </div>
            </section>

            {ratedIds.some((id) => !starter.some((m) => m.id === id)) ? (
              <section className="rounded-xl border border-border/70 bg-card/50 p-5">
                <h2 className="text-sm font-semibold">Other films you rated</h2>
                <div className="mt-4 flex flex-wrap gap-2">
                  {ratedIds
                    .filter((id) => !starter.some((m) => m.id === id))
                    .map((id) => {
                      const movie = moviesById.get(id);
                      if (!movie) return null;
                      return (
                        <div
                          key={id}
                          className="flex items-center gap-2 rounded-lg border border-border/70 px-3 py-2"
                        >
                          <span className="text-xs">{movie.title}</span>
                          <StarRating readOnly size="sm" value={draft[id]} />
                          <button
                            type="button"
                            aria-label={`Remove rating for ${movie.title}`}
                            onClick={() => handleClear(id)}
                            className="text-muted-foreground transition-colors hover:text-destructive"
                          >
                            <Trash2 className="size-3.5" />
                          </button>
                        </div>
                      );
                    })}
                </div>
              </section>
            ) : null}

            <section className="rounded-xl border border-border/70 bg-card/50 p-5">
              <h2 className="text-sm font-semibold">Find any film</h2>
              <div className="relative mt-3 max-w-md">
                <Search className="absolute left-3 top-2.5 size-4 text-muted-foreground" />
                <Input
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Search 1,682 titles…"
                  className="pl-9"
                />
              </div>
              {query.trim().length > 0 ? (
                <div className="mt-4 divide-y divide-border/60">
                  {results.length === 0 ? (
                    <p className="py-3 text-xs text-muted-foreground">No titles match that search.</p>
                  ) : (
                    results.slice(0, 12).map((movie) => (
                      <div
                        key={movie.id}
                        className="flex flex-wrap items-center justify-between gap-3 py-2.5"
                      >
                        <div className="min-w-0">
                          <p className="truncate text-sm">{movie.title}</p>
                          <p className="text-[11px] text-muted-foreground">
                            {movie.year ?? "year unknown"} · {movie.genres.filter((g) => g !== "unknown").slice(0, 3).join(", ")}
                          </p>
                        </div>
                        <StarRating
                          value={draft[movie.id] ?? null}
                          onChange={(rating) => handleRate(movie.id, rating)}
                          size="sm"
                        />
                      </div>
                    ))
                  )}
                </div>
              ) : null}
            </section>
          </div>

          <aside className="flex flex-col gap-4 lg:sticky lg:top-6 lg:self-start">
            <div className="rounded-xl border border-border/70 bg-card/60 p-5">
              <div className="flex items-center gap-2 text-primary">
                <Target className="size-4" />
                <h2 className="text-sm font-semibold text-foreground">Persona mapping</h2>
              </div>
              <Progress value={personaProgress} className="mt-4 h-1.5" />
              <p className="mt-3 text-xs leading-5 text-muted-foreground">
                {activeProfile?.persona
                  ? `Matched to MovieLens user ${activeProfile.persona.userId} — cosine ${activeProfile.persona.similarity.toFixed(2)} over ${activeProfile.persona.sharedMovies} co-rated films.`
                  : `A persona match needs at least ${PERSONA_MIN_SHARED} films you and one MovieLens user both rated. Below that, the recommender uses a labelled fallback instead of a model prediction.`}
              </p>
              {activeProfile && !activeProfile.persona && activeProfile.personaCandidates.length > 0 ? (
                <p className="mt-2 text-[11px] text-muted-foreground">
                  Closest candidate so far: user {activeProfile.personaCandidates[0].userId} with{" "}
                  {activeProfile.personaCandidates[0].sharedMovies} shared films.
                </p>
              ) : null}
            </div>

            {activeProfile && activeProfile.count > 0 ? (
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                className="rounded-xl border border-border/70 bg-card/60 p-5"
              >
                <div className="flex items-center gap-2 text-primary">
                  <UserRoundSearch className="size-4" />
                  <h2 className="text-sm font-semibold text-foreground">Your genre signals</h2>
                </div>
                <p className="mt-2 text-[11px] text-muted-foreground">
                  Your mean rating is {activeProfile.meanRating.toFixed(2)} against the dataset mean of{" "}
                  {activeProfile.globalMean.toFixed(2)}.
                </p>
                <ul className="mt-4 space-y-2.5">
                  {activeProfile.genreAffinity.slice(0, 6).map((entry) => (
                    <li key={entry.genre} className="flex items-center justify-between gap-3 text-xs">
                      <span>{entry.genre}</span>
                      <span className="flex items-center gap-2">
                        <span className="tabular-nums text-muted-foreground">
                          {entry.meanRating.toFixed(2)} · {entry.count}
                        </span>
                        <span
                          className={`w-12 text-right tabular-nums ${entry.lift >= 0 ? "text-chart-5" : "text-chart-3"}`}
                        >
                          {entry.lift >= 0 ? "+" : ""}
                          {entry.lift.toFixed(2)}
                        </span>
                      </span>
                    </li>
                  ))}
                </ul>
              </motion.div>
            ) : null}

            <div className="rounded-xl border border-border/70 bg-card/60 p-5">
              <h2 className="text-sm font-semibold">Where this goes next</h2>
              <p className="mt-2 text-xs leading-5 text-muted-foreground">
                Ratings feed candidate filtering (nothing you rated is recommended back to you) and
                the content-affinity layer. The model predictions themselves come from the trained
                stacking ensemble.
              </p>
              <Button asChild variant="outline" size="sm" className="mt-4 w-full">
                <Link to="/analytics">Train the ensemble</Link>
              </Button>
            </div>
          </aside>
        </div>
      )}
    </AppShell>
  );
}

function RatingCard({
  movieId,
  title,
  year,
  genres,
  support,
  value,
  onRate,
  index,
}: {
  movieId: number;
  title: string;
  year: number | null;
  genres: string[];
  support: number;
  value: number | null;
  onRate: (movieId: number, rating: number) => void;
  index: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: Math.min(index * 0.03, 0.4) }}
      className="flex flex-col gap-2.5"
    >
      <MoviePoster movieId={movieId} title={title} year={year} className="aspect-[2/3] w-full" />
      <div className="min-w-0">
        <p className="truncate text-xs font-medium" title={title}>
          {title}
        </p>
        <p className="text-[11px] text-muted-foreground">
          {genres.filter((g) => g !== "unknown").slice(0, 2).join(" · ")}
          {support > 0 ? ` · ${support} ratings` : ""}
        </p>
      </div>
      <StarRating value={value} onChange={(rating) => onRate(movieId, rating)} size="sm" />
    </motion.div>
  );
}

function LoadingCard({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-border/70 bg-card/50 p-6 text-sm text-muted-foreground">
      <Loader2 className="size-4 animate-spin" />
      {label}
    </div>
  );
}
