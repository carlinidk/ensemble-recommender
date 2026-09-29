import { ChevronDown } from "lucide-react";
import { useState, type ReactNode } from "react";
import { cn } from "@/lib/utils";

interface DisclosureProps {
  /** Short summary shown when the section is hidden — the only visible hint. */
  title: string;
  /** Optional supporting line under the title while collapsed. */
  hint?: string;
  /** Show the open/closed chevron; defaults to true. */
  withChevron?: boolean;
  /** Visual weight: "plain" for cards, "ghost" for inline links. */
  tone?: "plain" | "ghost";
  /** Start expanded (rarely wanted; detail stays hidden by default). */
  defaultOpen?: boolean;
  className?: string;
  children: ReactNode;
}

/**
 * Click-to-reveal container. Everything research- or model-flavoured lives
 * inside one of these so the movie experience stays front and centre.
 */
export function Disclosure({
  title,
  hint,
  withChevron = true,
  tone = "plain",
  defaultOpen = false,
  className,
  children,
}: DisclosureProps) {
  const [open, setOpen] = useState(defaultOpen);

  if (tone === "ghost") {
    return (
      <div className={className}>
        <button
          type="button"
          onClick={() => setOpen((prev) => !prev)}
          aria-expanded={open}
          className="group flex items-center gap-1.5 text-[11px] text-muted-foreground transition-colors hover:text-foreground"
        >
          {withChevron ? (
            <ChevronDown
              className={cn("size-3.5 transition-transform", open && "rotate-180")}
            />
          ) : null}
          {open ? "Hide details" : title}
        </button>
        {open ? <div className="mt-3">{children}</div> : null}
      </div>
    );
  }

  return (
    <div className={cn("rounded-xl border border-border/70 bg-card/60", className)}>
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-3 px-5 py-4 text-left"
      >
        <span className="min-w-0">
          <span className="block text-sm font-semibold">{title}</span>
          {hint ? (
            <span className="mt-0.5 block text-[11px] leading-4 text-muted-foreground">
              {hint}
            </span>
          ) : null}
        </span>
        {withChevron ? (
          <ChevronDown
            className={cn(
              "size-4 shrink-0 text-muted-foreground transition-transform",
              open && "rotate-180",
            )}
          />
        ) : null}
      </button>
      {open ? (
        <div className="border-t border-border/60 px-5 py-4">{children}</div>
      ) : null}
    </div>
  );
}
