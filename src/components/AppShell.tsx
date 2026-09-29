import { useAuth } from "@/hooks/use-auth";
import { cn } from "@/lib/utils";
import { LogOut, Menu, Sparkles } from "lucide-react";
import { useState, type ReactNode } from "react";
import { Link, useLocation, useNavigate } from "react-router";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTitle, SheetTrigger } from "@/components/ui/sheet";

interface NavItem {
  to: string;
  label: string;
  hint: string;
}

const NAV: NavItem[] = [
  { to: "/dashboard", label: "Overview", hint: "Your account at a glance" },
  { to: "/rate", label: "Rate films", hint: "Build your taste profile" },
  { to: "/recommendations", label: "Top-K picks", hint: "Ranked recommendations" },
  { to: "/browse", label: "Browse catalogue", hint: "All 1,682 films" },
  { to: "/analytics", label: "Models & experiments", hint: "Run and inspect the ensemble" },
];

function NavLinks({ onNavigate }: { onNavigate?: () => void }) {
  const location = useLocation();
  return (
    <nav className="flex flex-col gap-1">
      {NAV.map((item) => {
        const active = location.pathname === item.to;
        return (
          <Link
            key={item.to}
            to={item.to}
            onClick={onNavigate}
            className={cn(
              "group rounded-lg px-3 py-2.5 transition-colors",
              active
                ? "bg-primary/12 text-foreground"
                : "text-muted-foreground hover:bg-white/5 hover:text-foreground",
            )}
          >
            <span className="flex items-center gap-2 text-sm font-medium">
              {active ? <span className="size-1.5 rounded-full bg-primary" /> : null}
              {item.label}
            </span>
            <span className="mt-0.5 block text-[11px] text-muted-foreground/70">{item.hint}</span>
          </Link>
        );
      })}
    </nav>
  );
}

function Brand() {
  return (
    <Link to="/" className="flex items-center gap-2.5">
      <span className="flex size-8 items-center justify-center rounded-md bg-primary/15 text-primary">
        <Sparkles className="size-4" />
      </span>
      <span className="font-display text-xl leading-none">Ensemble</span>
    </Link>
  );
}

export function AppShell({
  title,
  description,
  children,
  actions,
}: {
  title: string;
  description?: string;
  children: ReactNode;
  actions?: ReactNode;
}) {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);

  const handleSignOut = async () => {
    await signOut();
    navigate("/");
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="mx-auto flex w-full max-w-[1400px]">
        {/* Desktop sidebar */}
        <aside className="sticky top-0 hidden h-screen w-64 shrink-0 flex-col justify-between border-r border-border/70 px-4 py-6 lg:flex">
          <div className="flex flex-col gap-8">
            <Brand />
            <NavLinks />
          </div>
          <div className="flex flex-col gap-3 border-t border-border/70 pt-4">
            <div className="px-1">
              <p className="truncate text-sm font-medium">{user?.name ?? user?.email ?? "Guest"}</p>
              <p className="text-[11px] text-muted-foreground">Signed in</p>
            </div>
            <Button variant="outline" size="sm" className="gap-2" onClick={handleSignOut}>
              <LogOut className="size-3.5" />
              Sign out
            </Button>
          </div>
        </aside>

        <div className="flex min-w-0 flex-1 flex-col">
          {/* Mobile top bar */}
          <header className="sticky top-0 z-30 flex items-center justify-between gap-3 border-b border-border/70 bg-background/90 px-4 py-3 backdrop-blur lg:hidden">
            <Brand />
            <Sheet open={open} onOpenChange={setOpen}>
              <SheetTrigger asChild>
                <Button variant="outline" size="icon" aria-label="Open navigation">
                  <Menu className="size-4" />
                </Button>
              </SheetTrigger>
              <SheetContent side="right" className="w-72 p-5">
                <SheetTitle className="mb-6 font-display text-xl">Ensemble</SheetTitle>
                <NavLinks onNavigate={() => setOpen(false)} />
                <div className="mt-6 border-t border-border/70 pt-4">
                  <Button variant="outline" size="sm" className="w-full gap-2" onClick={handleSignOut}>
                    <LogOut className="size-3.5" />
                    Sign out
                  </Button>
                </div>
              </SheetContent>
            </Sheet>
          </header>

          <main className="min-w-0 flex-1 px-4 py-6 sm:px-6 lg:px-8 lg:py-10">
            <div className="mb-6 flex flex-col gap-3 sm:mb-8 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <h1 className="font-display text-3xl tracking-tight sm:text-4xl">{title}</h1>
                {description ? (
                  <p className="mt-1.5 max-w-2xl text-sm leading-6 text-muted-foreground">
                    {description}
                  </p>
                ) : null}
              </div>
              {actions ? <div className="flex flex-wrap gap-2">{actions}</div> : null}
            </div>
            {children}
          </main>
        </div>
      </div>
    </div>
  );
}
