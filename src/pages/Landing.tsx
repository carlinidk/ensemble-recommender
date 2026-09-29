import { MoviePoster } from "@/components/MoviePoster";
import { useMl } from "@/components/MlProvider";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/hooks/use-auth";
import { PAPER, PAPER_STANDALONE } from "@/ml/reference";
import { motion } from "framer-motion";
import {
  ArrowRight,
  Boxes,
  Database,
  GitBranch,
  LineChart,
  ShieldCheck,
  Sparkles,
  Star,
  Workflow,
} from "lucide-react";
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
            <a className="transition-colors hover:text-foreground" href="#method">
              Method
            </a>
            <a className="transition-colors hover:text-foreground" href="#results">
              Results
            </a>
            <a className="transition-colors hover:text-foreground" href="#integrity">
              Integrity
            </a>
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

      {/* Hero */}
      <section className="stage-glow relative overflow-hidden border-b border-border/60">
        <div className="mx-auto grid w-full max-w-6xl gap-12 px-4 py-16 sm:px-6 lg:grid-cols-[1.05fr_0.95fr] lg:py-24">
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="flex flex-col justify-center"
          >
            <Badge variant="outline" className="w-fit border-primary/40 text-primary">
              Reproduction study · JATIT Vol. 101 No. 18
            </Badge>
            <h1 className="mt-5 font-display text-4xl leading-[1.05] tracking-tight sm:text-5xl lg:text-6xl">
              Six base learners.
              <br />
              One meta-learner.
              <br />
              <span className="text-primary">Every number measured here.</span>
            </h1>
            <p className="mt-5 max-w-xl text-base leading-7 text-muted-foreground">
              A working reconstruction of Sharma &amp; Dutta's stacking ensemble for movie
              recommendation. Rate a handful of films, get a Top-K list ranked by a real
              KNN + XGBoost + Gradient Boosting → Linear Regression stack — with the published
              reference numbers kept strictly separate from ours.
            </p>
            <div className="mt-7 flex flex-wrap items-center gap-3">
              <Button asChild size="lg" className="gap-2">
                <Link to={primaryCta}>
                  Build my Top-K
                  <ArrowRight className="size-4" />
                </Link>
              </Button>
              <Button asChild size="lg" variant="outline">
                <a href="#method">See the architecture</a>
              </Button>
            </div>
            <dl className="mt-9 grid max-w-lg grid-cols-3 gap-4 border-t border-border/60 pt-5">
              <Stat label="Ratings" value={meta ? meta.audit.rows.toLocaleString() : "100,000"} />
              <Stat label="Users" value={meta ? String(meta.audit.users) : "943"} />
              <Stat label="Films" value={meta ? String(meta.audit.movies) : "1,682"} />
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

      {/* Method / architecture */}
      <section id="method" className="border-b border-border/60">
        <div className="mx-auto w-full max-w-6xl px-4 py-16 sm:px-6 lg:py-20">
          <motion.div {...fadeUp}>
            <Badge variant="outline" className="border-border/70">
              <Workflow className="mr-1.5 size-3" />
              Two-level stacking
            </Badge>
            <h2 className="mt-4 max-w-2xl font-display text-3xl tracking-tight sm:text-4xl">
              The pipeline, exactly as the paper describes it
            </h2>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-muted-foreground">
              User, movie and rating files are merged into one table. Categorical ids are encoded
              numerically and the release year is extracted, giving the paper's four named columns.
              Base learners produce predictions that become the meta-learner's features.
            </p>
          </motion.div>

          <motion.div {...fadeUp} className="mt-10 grid gap-4 lg:grid-cols-[1fr_auto_1fr_auto_1fr]">
            <Stage
              icon={<Database className="size-4" />}
              title="Merged dataset"
              items={["user_id → encoded", "movie_id → encoded", "year (imputed)", "rating → target"]}
              note="Demographics are merged but excluded from modelling, as the paper states."
            />
            <Arrow />
            <Stage
              icon={<Boxes className="size-4" />}
              title="Level 1 · base learners"
              items={["KNN regressor", "XGBoost regressor", "Gradient Boosting"]}
              note="Trained per fold so the meta learner never sees in-sample predictions."
            />
            <Arrow />
            <Stage
              icon={<GitBranch className="size-4" />}
              title="Level 2 · meta learner"
              items={["Linear Regression", "on out-of-fold predictions"]}
              note="Base learners are then refit on the full training split for serving."
            />
          </motion.div>

          <motion.div {...fadeUp} className="mt-4 grid gap-4 sm:grid-cols-3">
            <MiniCard
              title="No stacking leakage"
              body="Five-fold cross-validation produces out-of-fold base predictions. The meta-learner is trained on those, never on in-sample output."
            />
            <MiniCard
              title="Six standalone baselines"
              body="Linear Regression, KNN, Random Forest, AdaBoost, Gradient Boosting and XGBoost are each evaluated independently first."
            />
            <MiniCard
              title="Seven stacking configurations"
              body="Every meta-learner combination from the paper is re-run here and compared on the same held-out split."
            />
          </motion.div>
        </div>
      </section>

      {/* Results */}
      <section id="results" className="border-b border-border/60 bg-card/40">
        <div className="mx-auto w-full max-w-6xl px-4 py-16 sm:px-6 lg:py-20">
          <motion.div {...fadeUp} className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <Badge variant="outline" className="border-border/70">
                <LineChart className="mr-1.5 size-3" />
                Reference vs reproduction
              </Badge>
              <h2 className="mt-4 font-display text-3xl tracking-tight sm:text-4xl">
                Published reference results
              </h2>
              <p className="mt-3 max-w-2xl text-sm leading-6 text-muted-foreground">
                These are the paper's reported standalone RMSE values. They are shown as reference
                only and are never mixed into our metric tables.
              </p>
            </div>
            <Button asChild variant="outline" size="sm" className="gap-2">
              <Link to={isAuthenticated ? "/analytics" : "/auth?returnTo=%2Fanalytics"}>
                Run the full experiment
                <ArrowRight className="size-3.5" />
              </Link>
            </Button>
          </motion.div>

          <motion.div {...fadeUp} className="mt-8 overflow-hidden rounded-xl border border-border/70">
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
          </motion.div>
          <p className="mt-3 text-xs leading-5 text-muted-foreground">
            Two of six positions reproduce closely and four do not — the analytics page explains the
            gap rather than hiding it. The full-dataset run trains all six models and seven stacking
            configurations in roughly 50 seconds, on your machine, in a web worker.
          </p>
        </div>
      </section>

      {/* Integrity */}
      <section id="integrity" className="border-b border-border/60">
        <div className="mx-auto w-full max-w-6xl px-4 py-16 sm:px-6 lg:py-20">
          <motion.div {...fadeUp}>
            <Badge variant="outline" className="border-border/70">
              <ShieldCheck className="mr-1.5 size-3" />
              Research integrity
            </Badge>
            <h2 className="mt-4 font-display text-3xl tracking-tight sm:text-4xl">
              What this project will not do
            </h2>
          </motion.div>
          <div className="mt-9 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
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
              <motion.div
                key={card.title}
                {...fadeUp}
                className="rounded-xl border border-border/70 bg-card/60 p-5"
              >
                <h3 className="text-sm font-semibold">{card.title}</h3>
                <p className="mt-2 text-xs leading-5 text-muted-foreground">{card.body}</p>
              </motion.div>
            ))}
          </div>
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
            Paper reproduced: {PAPER.authors}, “{PAPER.title}”, {PAPER.venue}, {PAPER.volume},{" "}
            {PAPER.date}. Dataset: {PAPER.dataset}.
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

function Stage({
  icon,
  title,
  items,
  note,
}: {
  icon: React.ReactNode;
  title: string;
  items: string[];
  note: string;
}) {
  return (
    <div className="rounded-xl border border-border/70 bg-card/60 p-5">
      <div className="flex items-center gap-2 text-primary">{icon}</div>
      <h3 className="mt-3 text-sm font-semibold">{title}</h3>
      <ul className="mt-3 space-y-1.5">
        {items.map((item) => (
          <li key={item} className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className="size-1 rounded-full bg-primary/70" />
            {item}
          </li>
        ))}
      </ul>
      <p className="mt-4 border-t border-border/60 pt-3 text-[11px] leading-4 text-muted-foreground/80">
        {note}
      </p>
    </div>
  );
}

function Arrow() {
  return (
    <div className="hidden items-center justify-center lg:flex">
      <ArrowRight className="size-5 text-primary/60" />
    </div>
  );
}

function MiniCard({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-xl border border-border/70 bg-card/40 p-5">
      <h3 className="text-sm font-semibold">{title}</h3>
      <p className="mt-2 text-xs leading-5 text-muted-foreground">{body}</p>
    </div>
  );
}
