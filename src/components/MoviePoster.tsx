import { cn } from "@/lib/utils";

/**
 * Movies in this project have no artwork: the MovieLens files ship titles and
 * IMDb links but no images, and hot-linking third-party posters is unreliable.
 * Each title therefore gets a deterministic generated "poster" derived from its
 * id, so the same movie always renders the same way and nothing is fabricated
 * or misattributed.
 */

const PALETTES: [string, string, string][] = [
  ["oklch(0.55 0.14 75)", "oklch(0.26 0.05 55)", "oklch(0.86 0.09 85)"],
  ["oklch(0.5 0.12 195)", "oklch(0.22 0.05 205)", "oklch(0.82 0.08 190)"],
  ["oklch(0.52 0.15 25)", "oklch(0.24 0.06 20)", "oklch(0.85 0.08 40)"],
  ["oklch(0.5 0.13 300)", "oklch(0.23 0.05 295)", "oklch(0.82 0.07 305)"],
  ["oklch(0.55 0.13 145)", "oklch(0.23 0.05 150)", "oklch(0.84 0.08 140)"],
  ["oklch(0.62 0.12 40)", "oklch(0.25 0.05 30)", "oklch(0.87 0.07 55)"],
];

function hash(value: number): number {
  let h = value * 2654435761;
  h ^= h >>> 13;
  return Math.abs(h);
}

function initials(title: string): string {
  const clean = title.replace(/\(.*?\)/g, "").trim();
  const words = clean.split(/[\s:]+/).filter((w) => w.length > 2);
  if (words.length === 0) return clean.slice(0, 2).toUpperCase();
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return (words[0][0] + words[1][0]).toUpperCase();
}

export function MoviePoster({
  movieId,
  title,
  year,
  className,
  compact = false,
}: {
  movieId: number;
  title: string;
  year: number | null;
  className?: string;
  compact?: boolean;
}) {
  const [from, mid, to] = PALETTES[hash(movieId) % PALETTES.length];
  const angle = 120 + (hash(movieId) % 60);

  return (
    <div
      className={cn(
        "relative flex select-none flex-col justify-between overflow-hidden rounded-md border border-white/10",
        className,
      )}
      style={{
        backgroundImage: `linear-gradient(${angle}deg, ${from} 0%, ${mid} 62%, ${mid} 100%)`,
      }}
      aria-hidden="true"
    >
      <div
        className="absolute inset-x-0 top-0 h-1/3 opacity-40"
        style={{ backgroundImage: `linear-gradient(180deg, ${to}, transparent)` }}
      />
      <div
        className="absolute -right-6 top-1/3 h-24 w-24 rotate-12 rounded-full opacity-25 blur-xl"
        style={{ backgroundColor: to }}
      />
      {compact ? (
        <span className="relative z-10 m-auto font-display text-xl tracking-tight text-white/90">
          {initials(title)}
        </span>
      ) : (
        <div className="relative z-10 flex h-full flex-col justify-end gap-1 p-3">
          <span className="font-display text-lg leading-tight text-white/95">{title}</span>
          {year !== null ? (
            <span className="text-[11px] font-medium uppercase tracking-[0.18em] text-white/60">
              {year}
            </span>
          ) : null}
        </div>
      )}
    </div>
  );
}
