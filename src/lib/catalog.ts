import type { Movie } from "@/ml/types";

export interface MovieStat {
  movieId: number;
  count: number;
  mean: number;
}

export function statMap(stats: MovieStat[]): Map<number, MovieStat> {
  return new Map(stats.map((s) => [s.movieId, s]));
}

/** Highest-rated filter used by the explorer and the starter rating grid. */
export function isWellKnown(stats: Map<number, MovieStat>, movieId: number, minCount = 80): boolean {
  return (stats.get(movieId)?.count ?? 0) >= minCount;
}

/**
 * Deterministic starter set: the most-rated films, round-robined across primary
 * genres so the first screen a user sees is not five action movies.
 */
export function starterSelection(
  movies: Movie[],
  stats: Map<number, MovieStat>,
  count: number,
): Movie[] {
  const ranked = movies
    .map((movie) => ({ movie, count: stats.get(movie.id)?.count ?? 0 }))
    .filter((entry) => entry.count >= 100)
    .sort((a, b) => b.count - a.count || a.movie.id - b.movie.id);

  const buckets = new Map<string, Movie[]>();
  for (const entry of ranked.slice(0, 400)) {
    const primary = entry.movie.genres.find((g) => g !== "unknown") ?? "other";
    const bucket = buckets.get(primary) ?? [];
    bucket.push(entry.movie);
    buckets.set(primary, bucket);
  }

  const ordered: Movie[] = [];
  let added = true;
  while (ordered.length < count && added) {
    added = false;
    for (const bucket of buckets.values()) {
      const next = bucket.shift();
      if (next) {
        ordered.push(next);
        added = true;
        if (ordered.length >= count) break;
      }
    }
  }
  return ordered.slice(0, count);
}

export function genreList(movies: Movie[], limit?: number): string[] {
  const counts = new Map<string, number>();
  for (const movie of movies) {
    for (const genre of movie.genres) {
      if (genre === "unknown") continue;
      counts.set(genre, (counts.get(genre) ?? 0) + 1);
    }
  }
  const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([genre]) => genre);
  return limit ? sorted.slice(0, limit) : sorted;
}

export function searchMovies(movies: Movie[], query: string, limit = 60): Movie[] {
  const needle = query.trim().toLowerCase();
  if (!needle) return [];
  return movies
    .filter((movie) => movie.title.toLowerCase().includes(needle))
    .slice(0, limit);
}

export function ratingTone(rating: number | null): string {
  if (rating === null) return "text-muted-foreground";
  if (rating >= 4.5) return "text-chart-5";
  if (rating >= 4) return "text-primary";
  if (rating >= 3.5) return "text-chart-2";
  return "text-chart-3";
}
