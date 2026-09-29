/**
 * Diversification of the Top-K list.
 *
 * The paper stresses balancing accuracy and diversity, so the serving path
 * supports two modes:
 *
 *  - ACCURACY MODE: rank strictly by the scoring function.
 *  - DIVERSITY MODE: re-rank with Maximum Marginal Relevance (Carbonell &
 *    Goldstein, 1998):
 *
 *        MMR = λ · relevance(m) − (1 − λ) · max_{s ∈ selected} similarity(m, s)
 *
 *    Similarity is the cosine between genre vectors, with a small release-year
 *    proximity term. λ is exposed to the UI (`diversityLambda`), so the
 *    accuracy/diversity trade-off is visible and configurable rather than
 *    hidden behind a fixed constant.
 */

export interface RankableItem {
  id: number;
  relevance: number;
  genreVector: Float64Array;
  year: number | null;
}

export interface MmrResult {
  ids: number[];
  /** Similarity penalty applied when each item was selected (0 for the first). */
  penalties: Map<number, number>;
}

export function genreVector(genres: string[], genreNames: string[]): Float64Array {
  const vec = new Float64Array(genreNames.length);
  for (const g of genres) {
    const i = genreNames.indexOf(g);
    if (i >= 0) vec[i] = 1;
  }
  return vec;
}

function cosine(a: Float64Array, b: Float64Array): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const den = Math.sqrt(na) * Math.sqrt(nb);
  return den === 0 ? 0 : dot / den;
}

function yearSimilarity(a: number | null, b: number | null): number {
  if (a === null || b === null) return 0;
  const diff = Math.abs(a - b);
  return Math.max(0, 1 - diff / 40);
}

export function similarity(a: RankableItem, b: RankableItem): number {
  return 0.85 * cosine(a.genreVector, b.genreVector) + 0.15 * yearSimilarity(a.year, b.year);
}

/** Min-max normalize relevance so it is on the same scale as similarity. */
function normalizeRelevance(items: RankableItem[]): Map<number, number> {
  let min = Infinity;
  let max = -Infinity;
  for (const item of items) {
    if (item.relevance < min) min = item.relevance;
    if (item.relevance > max) max = item.relevance;
  }
  const out = new Map<number, number>();
  const span = max - min;
  for (const item of items) {
    out.set(item.id, span === 0 ? 1 : (item.relevance - min) / span);
  }
  return out;
}

export function selectWithMmr(
  items: RankableItem[],
  k: number,
  lambda: number,
): MmrResult {
  const relevance = normalizeRelevance(items);
  const pool = [...items];
  const selected: RankableItem[] = [];
  const ids: number[] = [];
  const penalties = new Map<number, number>();
  const limit = Math.min(k, pool.length);

  while (ids.length < limit) {
    let bestIndex = 0;
    let bestScore = -Infinity;
    let bestPenalty = 0;
    for (let i = 0; i < pool.length; i++) {
      const item = pool[i];
      let maxSim = 0;
      for (const chosen of selected) {
        const sim = similarity(item, chosen);
        if (sim > maxSim) maxSim = sim;
      }
      const score = lambda * (relevance.get(item.id) ?? 0) - (1 - lambda) * maxSim;
      if (score > bestScore) {
        bestScore = score;
        bestIndex = i;
        bestPenalty = maxSim;
      }
    }
    const chosen = pool.splice(bestIndex, 1)[0];
    selected.push(chosen);
    ids.push(chosen.id);
    penalties.set(chosen.id, bestPenalty);
  }

  return { ids, penalties };
}

/** Distinct genres in a selection, used as a diversity read-out in the UI. */
export function genreCoverage(
  items: { genreVector: Float64Array }[],
  genreNames: string[],
): { genre: string; count: number }[] {
  const counts = new Map<string, number>();
  for (const item of items) {
    for (let i = 0; i < genreNames.length; i++) {
      if (item.genreVector[i] > 0) counts.set(genreNames[i], (counts.get(genreNames[i]) ?? 0) + 1);
    }
  }
  return [...counts.entries()]
    .map(([genre, count]) => ({ genre, count }))
    .sort((a, b) => b.count - a.count);
}

/** Mean pairwise similarity of a selection (lower = more diverse). */
export function intraListSimilarity(items: RankableItem[]): number {
  if (items.length < 2) return 0;
  let total = 0;
  let pairs = 0;
  for (let i = 0; i < items.length; i++) {
    for (let j = i + 1; j < items.length; j++) {
      total += similarity(items[i], items[j]);
      pairs++;
    }
  }
  return pairs === 0 ? 0 : total / pairs;
}
