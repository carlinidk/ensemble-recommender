import type { Movie, Rating } from "../types";

/**
 * User taste profile and MovieLens persona matching.
 *
 * The paper has no new-user story: its models take a MovieLens `user_id` as an
 * input feature, so a brand-new app user simply has no row in that space.
 * Rather than pretending otherwise, the serving path maps the signed-in user's
 * own ratings onto the closest MovieLens 100K user ("persona") and uses that
 * user id for the paper pipeline. The mapping is a bridge we added, and is
 * reported as such everywhere it is used.
 */

export interface UserRating {
  movieId: number;
  rating: number;
}

export interface GenreAffinity {
  genre: string;
  count: number;
  meanRating: number;
  /** Mean rating minus the user's own mean rating. */
  lift: number;
  /** Shrunk lift used in scoring. */
  weight: number;
}

export interface DecadeAffinity {
  decade: number;
  count: number;
  meanRating: number;
  lift: number;
}

export interface PersonaCandidate {
  userId: number;
  similarity: number;
  sharedMovies: number;
}

export interface TasteProfile {
  count: number;
  meanRating: number;
  globalMean: number;
  genreAffinity: GenreAffinity[];
  decadeAffinity: DecadeAffinity[];
  persona: (PersonaCandidate & { method: string; considered: number }) | null;
  personaCandidates: PersonaCandidate[];
}

export interface ProfileArgs {
  ratings: UserRating[];
  moviesById: Map<number, Movie>;
  ratingsByMovie: Map<number, Rating[]>;
  globalMean: number;
  /** Minimum co-rated movies before a persona match is accepted. */
  minSharedMovies?: number;
  /** Bayesian smoothing strength for genre/era affinities. */
  priorStrength?: number;
}

export function buildTasteProfile({
  ratings,
  moviesById,
  ratingsByMovie,
  globalMean,
  minSharedMovies = 8,
  priorStrength = 5,
}: ProfileArgs): TasteProfile {
  const count = ratings.length;
  const meanRating = count > 0 ? ratings.reduce((a, r) => a + r.rating, 0) / count : globalMean;

  // --- genre affinity -------------------------------------------------------
  const genreSums = new Map<string, { sum: number; count: number }>();
  const decadeSums = new Map<number, { sum: number; count: number }>();
  for (const r of ratings) {
    const movie = moviesById.get(r.movieId);
    if (!movie) continue;
    for (const genre of movie.genres) {
      const entry = genreSums.get(genre) ?? { sum: 0, count: 0 };
      entry.sum += r.rating;
      entry.count++;
      genreSums.set(genre, entry);
    }
    if (movie.year !== null) {
      const decade = Math.floor(movie.year / 10) * 10;
      const entry = decadeSums.get(decade) ?? { sum: 0, count: 0 };
      entry.sum += r.rating;
      entry.count++;
      decadeSums.set(decade, entry);
    }
  }

  const genreAffinity: GenreAffinity[] = [...genreSums.entries()]
    .map(([genre, { sum, count: c }]) => {
      const mean = sum / c;
      const lift = mean - meanRating;
      return {
        genre,
        count: c,
        meanRating: mean,
        lift,
        weight: (lift * c) / (c + priorStrength),
      };
    })
    .sort((a, b) => b.count - a.count);

  const decadeAffinity: DecadeAffinity[] = [...decadeSums.entries()]
    .map(([decade, { sum, count: c }]) => {
      const mean = sum / c;
      return { decade, count: c, meanRating: mean, lift: mean - meanRating };
    })
    .sort((a, b) => b.count - a.count);

  // --- persona matching (cosine over co-rated movies) -----------------------
  const dot = new Map<number, number>();
  const shared = new Map<number, number>();
  const otherSq = new Map<number, number>();
  // Sum of MY squared ratings over the items each candidate also rated, so the
  // cosine is computed over the shared items only. A single coincidence can
  // never produce a perfect similarity because the shared count is required to
  // clear `minSharedMovies`.
  const sharedMineSq = new Map<number, number>();
  for (const r of ratings) {
    const others = ratingsByMovie.get(r.movieId);
    if (!others) continue;
    for (const other of others) {
      dot.set(other.userId, (dot.get(other.userId) ?? 0) + r.rating * other.rating);
      shared.set(other.userId, (shared.get(other.userId) ?? 0) + 1);
      otherSq.set(other.userId, (otherSq.get(other.userId) ?? 0) + other.rating * other.rating);
      sharedMineSq.set(other.userId, (sharedMineSq.get(other.userId) ?? 0) + r.rating * r.rating);
    }
  }

  const candidates: PersonaCandidate[] = [];
  for (const [userId, d] of dot) {
    const sharedCount = shared.get(userId) ?? 0;
    if (sharedCount < minSharedMovies) continue;
    const denom = Math.sqrt(sharedMineSq.get(userId) ?? 0) * Math.sqrt(otherSq.get(userId) ?? 0);
    if (denom <= 0) continue;
    candidates.push({ userId, similarity: d / denom, sharedMovies: sharedCount });
  }
  candidates.sort((a, b) => b.similarity - a.similarity || b.sharedMovies - a.sharedMovies);
  const best = candidates.length > 0 ? candidates[0] : null;

  return {
    count,
    meanRating,
    globalMean,
    genreAffinity,
    decadeAffinity,
    persona: best
      ? {
          ...best,
          method: "cosine similarity over co-rated MovieLens 100K movies",
          considered: dot.size,
        }
      : null,
    personaCandidates: candidates.slice(0, 5),
  };
}

/**
 * Affinity bonus in rating units. Zero-weight affinities contribute nothing,
 * so a user with no genre history gets no bonus.
 */
export function affinityBonus(
  movie: Movie,
  profile: TasteProfile,
): { bonus: number; drivers: { genre: string; weight: number }[] } {
  const byGenre = new Map(profile.genreAffinity.map((g) => [g.genre, g]));
  const drivers: { genre: string; weight: number }[] = [];
  let bonus = 0;
  for (const genre of movie.genres) {
    const entry = byGenre.get(genre);
    if (!entry) continue;
    bonus += entry.weight;
    if (entry.count >= 2) drivers.push({ genre, weight: entry.weight });
  }
  if (movie.year !== null) {
    const decade = Math.floor(movie.year / 10) * 10;
    const era = profile.decadeAffinity.find((d) => d.decade === decade);
    if (era && era.count >= 3) bonus += 0.5 * era.lift;
  }
  return { bonus, drivers };
}
