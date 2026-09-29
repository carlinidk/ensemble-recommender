import { AppShell } from "@/components/AppShell";
import { MoviePoster } from "@/components/MoviePoster";
import { useMl } from "@/components/MlProvider";
import { StarRating } from "@/components/StarRating";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { api } from "@/convex/_generated/api";
import { genreList, ratingTone, statMap } from "@/lib/catalog";
import type { MovieScore } from "@/ml/workerProtocol";
import { useMutation, useQuery } from "convex/react";
import { motion } from "framer-motion";
import { Info, Loader2, Search } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

const PAGE_SIZE = 36;

type SortKey = "popularity" | "year" | "title" | "predicted";

export default function Browse() {
  const ml = useMl();
  const myRatings = useQuery(api.recommender.myRatings);
  const rateMovie = useMutation(api.recommender.rateMovie);

  const [query, setQuery] = useState("");
  const [genre, setGenre] = useState("all");
  const [sort, setSort] = useState<SortKey>("popularity");
  const [visible, setVisible] = useState(PAGE_SIZE);
  const [scores, setScores] = useState<Map<number, MovieScore>>(new Map());
  const [scoring, setScoring] = useState(false);
  // Optimistic layer over the reactive `myRatings` query: a tap shows instantly
  // and the stored value takes over as soon as Convex pushes it back.
  const [pendingRatings, setPendingRatings] = useState<Record<number, number>>({});

  const movies = useMemo(() => ml.meta?.movies ?? [], [ml.meta]);
  const stats = useMemo(() => statMap(ml.meta?.movieStats ?? []), [ml.meta?.movieStats]);
  const genres = useMemo(() => genreList(movies), [movies]);

  const savedRatings = useMemo(
    () => Object.fromEntries((myRatings ?? []).map((row) => [row.movieId, row.rating])),
    [myRatings],
  );
  const ratingOf = (movieId: number): number | null =>
    pendingRatings[movieId] ?? savedRatings[movieId] ?? null;
  const ratings = useMemo(() => {
    const merged = new Map<number, number>();
    for (const [id, value] of Object.entries(savedRatings)) merged.set(Number(id), value);
    for (const [id, value] of Object.entries(pendingRatings)) merged.set(Number(id), value);
    return [...merged.entries()].map(([movieId, rating]) => ({ movieId, rating }));
  }, [savedRatings, pendingRatings]);

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const list = movies.filter((movie) => {
      if (needle && !movie.title.toLowerCase().includes(needle)) return false;
      if (genre !== "all" && !movie.genres.includes(genre)) return false;
      return true;
    });
    return list.sort((a, b) => {
      switch (sort) {
        case "year":
          return (b.year ?? 0) - (a.year ?? 0);
        case "title":
          return a.title.localeCompare(b.title);
        case "predicted": {
          const pa = scores.get(a.id)?.predictedRating ?? -1;
          const pb = scores.get(b.id)?.predictedRating ?? -1;
          return pb - pa;
        }
        case "popularity":
        default:
          return (stats.get(b.id)?.count ?? 0) - (stats.get(a.id)?.count ?? 0);
      }
    });
  }, [movies, query, genre, sort, scores, stats]);

  const page = useMemo(() => filtered.slice(0, visible), [filtered, visible]);
  const pageIds = useMemo(() => page.map((movie) => movie.id), [page]);
  const pageKey = pageIds.join(",");

  const hasTrainedModels = ml.hasTrainedModels;
  const scoreMovies = ml.scoreMovies;

  // Predictions are only requested when a stacking ensemble is resident and the
  // account has ratings; otherwise the catalogue stays descriptive. Every
  // dependency is memoized, so a settled response cannot re-trigger the effect.
  useEffect(() => {
    if (!hasTrainedModels || ratings.length === 0 || pageIds.length === 0) return;
    const missing = pageIds.filter((id) => !scores.has(id));
    if (missing.length === 0) return;
    let cancelled = false;
    setScoring(true);
    const timer = window.setTimeout(() => {
      scoreMovies(ratings, missing)
        .then((result) => {
          if (cancelled) return;
          setScores((prev) => {
            const next = new Map(prev);
            for (const score of result) next.set(score.movieId, score);
            return next;
          });
        })
        .catch(() => undefined)
        .finally(() => {
          if (!cancelled) setScoring(false);
        });
    }, 250);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [pageKey, pageIds, ratings, hasTrainedModels, scoreMovies, scores]);

  const handleRate = (movieId: number, rating: number) => {
    setPendingRatings((prev) => ({ ...prev, [movieId]: rating }));
    rateMovie({ movieId, rating }).catch(() => toast.error("Could not save that rating."));
  };

  const hasPredictions = ml.hasTrainedModels && ratings.length > 0;

  return (
    <AppShell
      title="Browse the catalogue"
      description="All 1,682 MovieLens 100K films with dataset support counts — and, once an ensemble is trained and your account has ratings, a predicted rating for each title."
      actions={
        scoring ? (
          <Badge variant="outline" className="gap-2 border-border/70">
            <Loader2 className="size-3 animate-spin" />
            Scoring
          </Badge>
        ) : null
      }
    >
      <div className="mb-5 grid gap-3 rounded-xl border border-border/70 bg-card/50 p-4 sm:grid-cols-[1fr_auto_auto]">
        <div className="relative">
          <Search className="absolute left-3 top-2.5 size-4 text-muted-foreground" />
          <Input
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setVisible(PAGE_SIZE);
            }}
            placeholder="Search titles…"
            className="pl-9"
          />
        </div>
        <select
          value={genre}
          onChange={(event) => {
            setGenre(event.target.value);
            setVisible(PAGE_SIZE);
          }}
          className="h-9 rounded-md border border-input bg-transparent px-3 text-sm outline-none focus-visible:ring-[2px] focus-visible:ring-ring/50"
        >
          <option value="all">All genres</option>
          {genres.map((g) => (
            <option key={g} value={g}>
              {g}
            </option>
          ))}
        </select>
        <select
          value={sort}
          onChange={(event) => setSort(event.target.value as SortKey)}
          className="h-9 rounded-md border border-input bg-transparent px-3 text-sm outline-none focus-visible:ring-[2px] focus-visible:ring-ring/50"
        >
          <option value="popularity">Most rated</option>
          <option value="predicted" disabled={!hasPredictions}>
            Predicted for you
          </option>
          <option value="year">Newest</option>
          <option value="title">Title A–Z</option>
        </select>
      </div>

      {!hasPredictions ? (
        <p className="mb-5 flex items-start gap-2 rounded-xl border border-border/70 bg-card/40 px-4 py-3 text-xs leading-5 text-muted-foreground">
          <Info className="mt-0.5 size-3.5 shrink-0 text-primary" />
          Predicted ratings appear once the stacking ensemble is trained and your account has
          ratings to match against the MovieLens user space. Until then this page shows dataset
          statistics only — no invented scores.
        </p>
      ) : null}

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6">
        {page.map((movie, index) => {
          const score = scores.get(movie.id);
          const stat = stats.get(movie.id);
          return (
            <motion.article
              key={movie.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: Math.min((index % PAGE_SIZE) * 0.012, 0.3) }}
              className="flex flex-col gap-2"
            >
              <div className="relative">
                <MoviePoster
                  movieId={movie.id}
                  title={movie.title}
                  year={movie.year}
                  className="aspect-[2/3] w-full"
                />
                {score?.predictedRating != null ? (
                  <span className="absolute right-1.5 top-1.5 rounded-md bg-background/85 px-1.5 py-0.5 text-[11px] font-semibold tabular-nums backdrop-blur">
                    <span className={ratingTone(score.predictedRating)}>
                      {score.predictedRating.toFixed(2)}
                    </span>
                  </span>
                ) : null}
              </div>
              <div className="min-w-0">
                <p className="truncate text-xs font-medium" title={movie.title}>
                  {movie.title}
                </p>
                <p className="text-[11px] text-muted-foreground">
                  {movie.year ?? "n/a"} ·{" "}
                  {movie.genres.filter((g) => g !== "unknown").slice(0, 2).join(", ")}
                </p>
                <p className="text-[11px] text-muted-foreground/80 tabular-nums">
                  {stat ? `${stat.count} ratings · ${stat.mean.toFixed(2)}` : "no ratings"}
                </p>
              </div>
              <StarRating
                value={ratingOf(movie.id)}
                onChange={(rating) => handleRate(movie.id, rating)}
                size="sm"
              />
            </motion.article>
          );
        })}
      </div>

      {filtered.length === 0 ? (
        <p className="py-10 text-center text-sm text-muted-foreground">No films match those filters.</p>
      ) : null}

      {visible < filtered.length ? (
        <div className="mt-8 flex justify-center">
          <Button variant="outline" onClick={() => setVisible((v) => v + PAGE_SIZE)}>
            Load {Math.min(PAGE_SIZE, filtered.length - visible)} more of {filtered.length}
          </Button>
        </div>
      ) : null}
    </AppShell>
  );
}
