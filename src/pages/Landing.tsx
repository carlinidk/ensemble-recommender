import { MoviePoster } from "@/components/MoviePoster";
import { useMl } from "@/components/MlProvider";
import { Disclosure } from "@/components/Disclosure";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/hooks/use-auth";
import { PAPER, PAPER_STANDALONE } from "@/ml/reference";
import { motion } from "framer-motion";
import {
  ArrowRight,
  ChevronDown,
  Film,
  LineChart,
  ShieldCheck,
  Sparkles,
  Star,
  Ticket,
  Workflow,
} from "lucide-react";
import { useState } from "react";
import { Link } from "react-router";

const MODEL_LABELS: Record<string, string> = {
  linear_regression: "Linear Regression",
  xgboost: "XGBoost",
  random_forest: "Random Forest",
  adaboost: "AdaBoost",
  gradient_boosting: "Gradient Boosting",
  knn: "K-Nearest Neighbours",
};

const fadeUp = {
  initial: { opacity: 0, y: 18 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, margin: "-80px" },
  transition: { duration: 0.5, ease: "easeOut" as const },
};

export default function Landing() {
  const { isAuthenticated } = useAuth();
  const { meta, result } = useMl();

  const showcase = (meta?.movies ?? []).filter((m) => m.title.length < 26).slice(0, 6);
  const primaryCta = isAuthenticated ? "/dashboard" : "/auth?returnTo=%2Frate";
  const [walkthrough, setWalkthrough] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState<null | "method" | "results" | "integrity">(null);

  const toggle = (key: "method" | "results" | "integrity") =>
    setDetailsOpen((prev) => (prev === key ? null : key));

  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/85 backdrop-blur">
        <div className="mx-auto flex w-full max-w-6xl items-center justify-between gap-4 px-4 py-3.5 sm:px-6">
          <Link to="/" className="flex items-center gap-2.5">
            <span className="flex size-8 items-center justify-center rounded-md bg-primary/15 text-primary">
              <Sparkles className="size-4" />
            </span>
            <span className="font-display text-xl leading-none">Ensemble</span>
          </Link>
          <nav className="hidden items-center gap-7 text-sm text-muted-foreground md:flex">
            <button
              type="button"
              className="transition-colors hover:text-foreground"
              onClick={() => toggle("method")}
            >
              Method
            </button>
            <button
              type="button"
              className="transition-colors hover:text-foreground"
              onClick={() => toggle("results")}
            >
              Results
            </button>
            <button
              type="button"
              className="transition-colors hover:text-foreground"
              onClick={() => toggle("integrity")}
            >
              Integrity
            </button>
          </nav>
          <div className="flex items-center gap-2">
            <Button asChild variant="ghost" size="sm" className="hidden sm:inline-flex">
              <Link to="/auth">Sign in</Link>
            </Button>
            <Button asChild size="sm" className="gap-2">
              <Link to={primaryCta}>
                {isAuthenticated ? "Open workspace" : "Start rating"}
                <ArrowRight className="size-3.5" />
              </Link>
            </Button>
          </div>
        </div>
      </header>

      {/* Hero — the movie experience comes first */}
      <section className="stage-glow relative overflow-hidden border-b border-border/60">
        <div className="mx-auto grid w-full max-w-6xl gap-12 px-4 py-16 sm:px-6 lg:grid-cols-[1.05fr_0.95fr] lg:py-24">
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="flex flex-col justify-center"
          >
            <Badge variant="outline" className="w-fit border-primary/40 text-primary">
              <Ticket className="mr-1.5 size-3" />
              Your private screening room
            </Badge>
            <h1 className="mt-5 font-display text-4xl leading-[1.05] tracking-tight sm:text-5xl lg:text-6xl">
              Rate a few films.
              <br />
              <span className="text-primary">Get your next</span>
              <br />
              five favourites.
            </h1>
            <p className="mt-5 max-w-xl text-base leading-7 text-muted-foreground">
              Ensemble turns a handful of star ratings into a personal Top-K list — ranked by
              machine-learnt taste, tuned for accuracy or diversity, and explained with data that
              actually exists. No ads, no noise, just your next watch.
            </p>
            <div className="mt-7 flex flex-wrap items-center gap-3">
              <Button asChild size="lg" className="gap-2">
                <Link to={primaryCta}>
                  <Star className="size-4" />
                  Start rating
                  <ArrowRight className="size-4" />
                </Link>
              </Button>
              <Button
                size="lg"
                variant="outline"
                className="gap-2"
                onClick={() => setWalkthrough((prev) => !prev)}
                aria-expanded={walkthrough}
              >
                <Film className="size-4" />
                How it works
                <ChevronDown
                  className={`size-3.5 transition-transform ${walkthrough ? "rotate-180" : ""}`}
                />
              </Button>
            </div>
            {walkthrough ? (
              <motion.ol
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.35 }}
                className="mt-6 max-w-xl space-y-2.5 rounded-xl border border-border/70 bg-card/60 p-5 text-sm"
              >
                {[
                  "Rate the starter films or search any of the 1,682 titles.",
                  "Your ratings map you to the closest MovieLens 100K taste profile.",
                  "Pick a list size and an accuracy-or-diversity ranking.",
                  "Your Top-K arrives with a predicted rating for every film.",
                ].map((line, index) => (
                  <li key={line} className="flex gap-2.5 text-muted-foreground">
                    <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-primary/15 text-[11px] font-semibold text-primary">
                      {index + 1}
                    </span>
                    {line}
                  </li>
                ))}
              </motion.ol>
            ) : null}
            <dl className="mt-9 grid max-w-lg grid-cols-3 gap-4 border-t border-border/60 pt-5">
              <Stat label="Films" value={meta ? meta.audit.movies.toLocaleString() : "1,682"} />
              <Stat label="Genres" value="19" />
              <Stat label="Your list" value="Top-K" />
            </dl>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.7, delay: 0.12 }}
            className="grid grid-cols-3 gap-3 self-center"
          >
            {(showcase.length > 0
              ? showcase
              : Array.from({ length: 6 }, (_, i) => ({ id: i + 1, title: "MovieLens 100K", year: 1998 }))
            ).map((movie, index) => (
              <MoviePoster
                key={movie.id}
                movieId={movie.id}
                title={movie.title}
                year={movie.year}
                className={`aspect-[2/3] ${index % 2 === 1 ? "translate-y-4" : ""}`}
              />
            ))}
          </motion.div>
        </div>
      </section>

      {/* Deep-dive disclosures: research content stays hidden until clicked */}
      <section className="border-b border-border/60">
        <div className="mx-auto flex w-full max-w-6xl flex-col gap-3 px-4 py-12 sm:px-6">
          <p className="text-center text-[11px] uppercase tracking-[0.2em] text-muted-foreground">
            Behind the curtain
          </p>
          <Disclosure
            tone="plain"
            title="The method behind the picks"
            hint="Two-level stacking, exactly as the paper describes it — KNN + XGBoost + Gradient Boosting feeding a Linear Regression meta-learner."
          >
            <MethodDetails />
          </Disclosure>
          <Disclosure
            tone="plain"
            title="Published reference vs our reproduction"
            hint="The paper's reported RMSE values, shown strictly as reference — never presented as our results."
          >
            <ResultsDetails result={result} />
          </Disclosure>
          <Disclosure
            tone="plain"
            title="Research integrity commitments"
            hint="What this project will not do with numbers it did not measure."
          >
            <IntegrityDetails />
          </Disclosure>
        </div>
      </section>

      <section className="stage-glow">
        <div className="mx-auto w-full max-w-6xl px-4 py-16 text-center sm:px-6 lg:py-20">
          <motion.div {...fadeUp}>
            <h2 className="font-display text-3xl tracking-tight sm:text-4xl">
              Rate five films, meet your Top-K
            </h2>
            <p className="mx-auto mt-3 max-w-xl text-sm leading-6 text-muted-foreground">
              The recommender needs your ratings to map you onto the MovieLens space. Fewer than
              eight co-rated movies and it says so, falling back to a labelled popularity list
              instead of pretending to know you.
            </p>
            <div className="mt-7 flex flex-wrap justify-center gap-3">
              <Button asChild size="lg" className="gap-2">
                <Link to={primaryCta}>
                  <Star className="size-4" />
                  Start rating
                </Link>
              </Button>
              <Button asChild size="lg" variant="outline">
                <Link to="/auth">Sign in</Link>
              </Button>
            </div>
          </motion.div>
        </div>
      </section>

      <footer className="border-t border-border/60">
        <div className="mx-auto flex w-full max-w-6xl flex-col gap-3 px-4 py-8 text-xs text-muted-foreground sm:px-6">
          <p className="max-w-3xl leading-5">
            Recommendation engine reproduced from: {PAPER.authors}, “{PAPER.title}”, {PAPER.venue},{" "}
            {PAPER.volume}, {PAPER.date}. Dataset: {PAPER.dataset}.
          </p>
          <p>
            Built with React, Vite and Convex. The stacking ensemble is implemented from scratch in
            TypeScript and trains in a web worker on your machine.
          </p>
        </div>
      </footer>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-[11px] uppercase tracking-[0.16em] text-muted-foreground">{label}</dt>
      <dd className="mt-1 font-display text-2xl">{value}</dd>
    </div>
  );
}

function MethodDetails() {
  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-2 text-primary">
        <Workflow className="size-4" />
        <h3 className="text-sm font-semibold">Two-level stacking</h3>
      </div>
      <p className="text-xs leading-5 text-muted-foreground">
        User, movie and rating files are merged into one table. Categorical ids are encoded
        numerically and the release year is extracted, giving the paper's four named columns. Base
        learners produce predictions that become the meta-learner's features.
      </p>
      <div className="grid gap-4 lg:grid-cols-3">
        {[
          {
            title: "Merged dataset",
            items: ["user_id → encoded", "movie_id → encoded", "year (imputed)", "rating → target"],
            note: "Demographics are merged but excluded from modelling, as the paper states.",
          },
          {
            title: "Level 1 · base learners",
            items: ["KNN regressor", "XGBoost regressor", "Gradient Boosting"],
            note: "Trained per fold so the meta learner never sees in-sample predictions.",
          },
          {
            title: "Level 2 · meta learner",
            items: ["Linear Regression", "on out-of-fold predictions"],
            note: "Base learners are then refit on the full training split for serving.",
          },
        ].map((stage) => (
          <div key={stage.title} className="rounded-xl border border-border/70 bg-background/40 p-4">
            <h4 className="text-sm font-semibold">{stage.title}</h4>
            <ul className="mt-3 space-y-1.5">
              {stage.items.map((item) => (
                <li key={item} className="flex items-center gap-2 text-xs text-muted-foreground">
                  <span className="size-1 rounded-full bg-primary/70" />
                  {item}
                </li>
              ))}
            </ul>
            <p className="mt-4 border-t border-border/60 pt-3 text-[11px] leading-4 text-muted-foreground/80">
              {stage.note}
            </p>
          </div>
        ))}
      </div>
      <div className="grid gap-3 sm:grid-cols-3">
        {[
          {
            title: "No stacking leakage",
            body: "Five-fold cross-validation produces out-of-fold base predictions. The meta-learner is trained on those, never on in-sample output.",
          },
          {
            title: "Six standalone baselines",
            body: "Linear Regression, KNN, Random Forest, AdaBoost, Gradient Boosting and XGBoost are each evaluated independently first.",
          },
          {
            title: "Seven stacking configurations",
            body: "Every meta-learner combination from the paper is re-run here and compared on the same held-out split.",
          },
        ].map((card) => (
          <div key={card.title} className="rounded-xl border border-border/70 bg-background/40 p-4">
            <h4 className="text-sm font-semibold">{card.title}</h4>
            <p className="mt-2 text-xs leading-5 text-muted-foreground">{card.body}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function ResultsDetails({ result }: { result: ReturnType<typeof useMl>["result"] }) {
  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-2 text-primary">
        <LineChart className="size-4" />
        <h3 className="text-sm font-semibold">Reference vs reproduction</h3>
      </div>
      <p className="text-xs leading-5 text-muted-foreground">
        These are the paper's reported standalone RMSE values. They are shown as reference only and
        are never mixed into our metric tables.
      </p>
      <div className="overflow-hidden rounded-xl border border-border/70">
        <table className="w-full text-sm">
          <thead className="bg-secondary/50 text-left text-xs uppercase tracking-wider text-muted-foreground">
            <tr>
              <th className="px-4 py-3 font-medium">Model</th>
              <th className="px-4 py-3 font-medium">Paper RMSE</th>
              <th className="px-4 py-3 font-medium">
                {result ? "Our reproduction RMSE" : "Our reproduction"}
              </th>
              <th className="px-4 py-3 font-medium">Difference</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(PAPER_STANDALONE).map(([key, reference]) => {
              const ours = result?.standalone.find((s) => s.model === key);
              return (
                <tr key={key} className="border-t border-border/60">
                  <td className="px-4 py-3 font-medium">{MODEL_LABELS[key] ?? key}</td>
                  <td className="px-4 py-3 tabular-nums">{reference.rmse.toFixed(2)}</td>
                  <td className="px-4 py-3 tabular-nums">
                    {ours ? ours.rmse.toFixed(4) : <span className="text-muted-foreground">computed on run</span>}
                  </td>
                  <td className="px-4 py-3 tabular-nums">
                    {ours ? (
                      <span className={ours.deltaVsReference && ours.deltaVsReference > 0 ? "text-chart-3" : "text-chart-5"}>
                        {ours.deltaVsReference && ours.deltaVsReference >= 0 ? "+" : ""}
                        {ours.deltaVsReference?.toFixed(4)}
                      </span>
                    ) : (
                      <span className="text-muted-foreground">—</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="text-xs leading-5 text-muted-foreground">
        Two of six positions reproduce closely and four do not — the analytics page explains the gap
        rather than hiding it. The full-dataset run trains all six models and seven stacking
        configurations in roughly 50 seconds, on your machine, in a web worker.
      </p>
    </div>
  );
}

function IntegrityDetails() {
  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-2 text-primary">
        <ShieldCheck className="size-4" />
        <h3 className="text-sm font-semibold">What this project will not do</h3>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        {[
          {
            title: "No borrowed numbers",
            body: "Published RMSE never appears as our result. Reference values live in one read-only file and are displayed under a separate heading.",
          },
          {
            title: "No invented explanations",
            body: "Every recommendation explanation is assembled from data that exists: base-learner outputs, persona match strength, the account's own ratings, and MovieLens support counts.",
          },
          {
            title: "Deviations documented",
            body: "Seeds, splits, fold counts and hyperparameters are unpublished by the paper, so ours are recorded per run and listed with the result.",
          },
          {
            title: "Leakage audited",
            body: "Test statistics never enter preprocessing, supervised features are refit inside each fold, and the meta learner only ever sees out-of-fold predictions.",
          },
        ].map((card) => (
          <div key={card.title} className="rounded-xl border border-border/70 bg-background/40 p-4">
            <h4 className="text-sm font-semibold">{card.title}</h4>
            <p className="mt-2 text-xs leading-5 text-muted-foreground">{card.body}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
