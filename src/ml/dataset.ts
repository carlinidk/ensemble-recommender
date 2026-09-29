import type {
  DataAudit,
  MergedRow,
  Movie,
  Rating,
  UserRecord,
} from "./types";

/**
 * PAPER MODE — dataset layer.
 *
 * The paper merges the MovieLens 100K movie, rating and user files into one
 * dataset. These parsers are the only place raw files are read.
 */

export const ML100K_VERSION = "ml-100k (October 1998 release)";
export const ML100K_BASE_PATH = "/data/ml-100k";

/** Genre list order matters: u.item stores 19 binary flags in this order. */
export function parseGenres(genreFile: string): string[] {
  return genreFile
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => line.split("|")[0].trim());
}

const MONTHS = [
  "jan",
  "feb",
  "mar",
  "apr",
  "may",
  "jun",
  "jul",
  "aug",
  "sep",
  "oct",
  "nov",
  "dec",
];

/** "01-Jan-1995" -> 1995, anything unparseable -> null. */
function parseReleaseDate(raw: string): number | null {
  const parts = raw.split("-");
  if (parts.length !== 3) return null;
  const monthIdx = MONTHS.indexOf(parts[1].toLowerCase());
  const year = Number.parseInt(parts[2], 10);
  if (monthIdx < 0 || !Number.isFinite(year)) return null;
  return year;
}

/** Most MovieLens titles end with the release year in parentheses. */
export function extractTitleYear(title: string): number | null {
  const match = /\((\d{4})\)\s*$/.exec(title.trim());
  if (match) {
    const year = Number.parseInt(match[1], 10);
    if (year >= 1880 && year <= 2100) return year;
  }
  const loose = /\((\d{4})\)/.exec(title);
  if (loose) return Number.parseInt(loose[1], 10);
  return null;
}

export function parseMovies(itemFile: string, genres: string[]): Movie[] {
  const movies: Movie[] = [];
  for (const line of itemFile.split("\n")) {
    if (!line.trim()) continue;
    const parts = line.split("|");
    if (parts.length < 6) continue;
    const id = Number.parseInt(parts[0], 10);
    if (!Number.isFinite(id)) continue;
    const title = parts[1].trim();
    const releaseYear = parseReleaseDate(parts[2] ?? "");
    const flags = parts.slice(5);
    const movieGenres = genres.filter((_, i) => flags[i] === "1");
    movies.push({
      id,
      title,
      year: extractTitleYear(title) ?? releaseYear,
      genres: movieGenres,
      imdbUrl: parts[4]?.trim() || null,
    });
  }
  return movies;
}

export function parseRatings(dataFile: string): Rating[] {
  const ratings: Rating[] = [];
  for (const line of dataFile.split("\n")) {
    if (!line.trim()) continue;
    const parts = line.split("\t");
    if (parts.length < 4) continue;
    const userId = Number.parseInt(parts[0], 10);
    const movieId = Number.parseInt(parts[1], 10);
    const rating = Number.parseInt(parts[2], 10);
    const timestamp = Number.parseInt(parts[3], 10);
    if (!Number.isFinite(userId) || !Number.isFinite(movieId)) continue;
    ratings.push({ userId, movieId, rating, timestamp });
  }
  return ratings;
}

export function parseUsers(userFile: string): UserRecord[] {
  const users: UserRecord[] = [];
  for (const line of userFile.split("\n")) {
    if (!line.trim()) continue;
    const parts = line.split("|");
    if (parts.length < 5) continue;
    const id = Number.parseInt(parts[0], 10);
    if (!Number.isFinite(id)) continue;
    users.push({
      id,
      age: Number.parseInt(parts[1], 10),
      gender: parts[2],
      occupation: parts[3],
      zip: parts[4],
    });
  }
  return users;
}

export interface MergedDataset {
  rows: MergedRow[];
  moviesById: Map<number, Movie>;
  usersById: Map<number, UserRecord>;
  /** ratings per movie, used by the recommender + popularity fallbacks */
  ratingsByMovie: Map<number, Rating[]>;
  /** ratings per user, used to exclude a matched persona's own history */
  ratingsByUser: Map<number, Rating[]>;
  /** movie popularity statistics over the whole dataset (descriptive, not modelling) */
  movieStats: Map<number, { count: number; mean: number }>;
  globalMean: number;
  audit: DataAudit;
}

/**
 * Merge the movie, rating and user files (the paper's step 1-2).
 *
 * Missing-value detection is reported in `audit`; the handling itself happens
 * in the feature layer so that every imputation is fitted on the training split
 * only (test-set statistics never influence preprocessing).
 */
export function mergeDataset(
  movies: Movie[],
  ratings: Rating[],
  users: UserRecord[],
): MergedDataset {
  const moviesById = new Map(movies.map((m) => [m.id, m]));
  const usersById = new Map(users.map((u) => [u.id, u]));
  const ratingsByMovie = new Map<number, Rating[]>();
  const ratingsByUser = new Map<number, Rating[]>();
  const seen = new Set<string>();
  let ratingTotal = 0;

  let duplicateUserMoviePairs = 0;
  let ratingsOutOfRange = 0;
  let missingMovieYear = 0;
  let missingMovieGenre = 0;
  let missingUserDemographics = 0;
  let droppedRows = 0;

  const rows: MergedRow[] = [];
  const years: number[] = [];

  for (const rating of ratings) {
    const movie = moviesById.get(rating.movieId);
    if (!movie) {
      droppedRows++;
      continue;
    }
    const pairKey = `${rating.userId}:${rating.movieId}`;
    if (seen.has(pairKey)) {
      duplicateUserMoviePairs++;
      continue;
    }
    seen.add(pairKey);

    if (rating.rating < 1 || rating.rating > 5) ratingsOutOfRange++;
    if (movie.year === null) missingMovieYear++;
    if (movie.genres.length === 0) missingMovieGenre++;
    if (!usersById.has(rating.userId)) missingUserDemographics++;

    if (movie.year !== null) years.push(movie.year);

    rows.push({
      userId: rating.userId,
      movieId: rating.movieId,
      rating: rating.rating,
      timestamp: rating.timestamp,
      year: movie.year ?? 0,
      yearMissing: movie.year === null,
      genreCount: movie.genres.length,
    });

    const list = ratingsByMovie.get(rating.movieId);
    if (list) list.push(rating);
    else ratingsByMovie.set(rating.movieId, [rating]);

    const userList = ratingsByUser.get(rating.userId);
    if (userList) userList.push(rating);
    else ratingsByUser.set(rating.userId, [rating]);
    ratingTotal += rating.rating;
  }

  const movieStats = new Map<number, { count: number; mean: number }>();
  for (const [movieId, list] of ratingsByMovie) {
    let sum = 0;
    for (const r of list) sum += r.rating;
    movieStats.set(movieId, { count: list.length, mean: sum / list.length });
  }

  years.sort((a, b) => a - b);
  const medianYear =
    years.length > 0 ? years[Math.floor(years.length / 2)] : 1990;

  return {
    rows,
    moviesById,
    usersById,
    ratingsByMovie,
    ratingsByUser,
    movieStats,
    globalMean: rows.length > 0 ? ratingTotal / rows.length : 3.5,
    audit: {
      rows: rows.length,
      users: new Set(rows.map((r) => r.userId)).size,
      movies: new Set(rows.map((r) => r.movieId)).size,
      duplicateUserMoviePairs,
      ratingsOutOfRange,
      missingMovieYear,
      missingMovieGenre,
      missingUserDemographics,
      droppedRows,
      yearFillValue: medianYear,
    },
  };
}

export interface RawMl100kFiles {
  movies: string;
  ratings: string;
  users: string;
  genres: string;
}

/** Browser loader for the bundled MovieLens 100K files. */
export async function fetchRawMl100k(
  basePath = ML100K_BASE_PATH,
): Promise<RawMl100kFiles> {
  const [movies, ratings, users, genres] = await Promise.all([
    fetch(`${basePath}/u.item`).then((r) => r.text()),
    fetch(`${basePath}/u.data`).then((r) => r.text()),
    fetch(`${basePath}/u.user`).then((r) => r.text()),
    fetch(`${basePath}/u.genre`).then((r) => r.text()),
  ]);
  return { movies, ratings, users, genres };
}

export interface LoadedDataset extends MergedDataset {
  movies: Movie[];
  genreNames: string[];
}

/** Parse + merge raw file text. Pure, so it can run in a test or a worker. */
export function buildDataset(files: RawMl100kFiles): LoadedDataset {
  const genreNames = parseGenres(files.genres);
  const movies = parseMovies(files.movies, genreNames);
  const ratings = parseRatings(files.ratings);
  const users = parseUsers(files.users);
  const merged = mergeDataset(movies, ratings, users);
  return { ...merged, movies, genreNames };
}

export function loadMl100k(basePath?: string): Promise<LoadedDataset> {
  return fetchRawMl100k(basePath).then(buildDataset);
}
