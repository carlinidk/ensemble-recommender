import { cn } from "@/lib/utils";
import { Star } from "lucide-react";
import { useState } from "react";

export function StarRating({
  value,
  onChange,
  readOnly = false,
  size = "md",
  label,
}: {
  value: number | null;
  onChange?: (rating: number) => void;
  readOnly?: boolean;
  size?: "sm" | "md";
  label?: string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const active = hover ?? value ?? 0;
  const dimension = size === "sm" ? "size-3.5" : "size-5";

  return (
    <div
      className="flex items-center gap-0.5"
      role={readOnly ? "img" : "radiogroup"}
      aria-label={label ?? (readOnly ? `Rated ${value ?? 0} of 5` : "Rate this movie")}
      onMouseLeave={() => setHover(null)}
    >
      {[1, 2, 3, 4, 5].map((star) => (
        <button
          key={star}
          type="button"
          disabled={readOnly}
          aria-label={`${star} star${star > 1 ? "s" : ""}`}
          onMouseEnter={() => !readOnly && setHover(star)}
          onFocus={() => !readOnly && setHover(star)}
          onClick={() => !readOnly && onChange?.(star)}
          className={cn(
            "rounded p-0.5 transition-colors",
            readOnly ? "cursor-default" : "hover:scale-110",
            star <= active ? "text-primary" : "text-muted-foreground/40",
          )}
        >
          <Star className={cn(dimension)} fill={star <= active ? "currentColor" : "none"} />
        </button>
      ))}
    </div>
  );
}
