import { Link, useLocation } from "@tanstack/react-router";
import {
  Activity,
  HeartPulse,
  Laptop,
  Moon,
  Network,
  Route as RouteIcon,
  ShieldAlert,
  Sun,
} from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";

import { Toaster } from "@/components/ui/sonner";
import { useTheme, type ThemeMode } from "@/lib/theme";

const navigation = [
  { label: "Analyze", path: "/explore", icon: Activity },
  { label: "Network", path: "/network", icon: Network },
  { label: "Disrupt", path: "/disruption", icon: ShieldAlert },
  { label: "Reroute", path: "/routing", icon: RouteIcon },
  { label: "Recover", path: "/recovery", icon: HeartPulse },
] as const;

const SAVED_ANALYSIS_KEY = "route-resilience.analysis-id";

export function WorkflowHeader({ children }: { children: ReactNode }) {
  const { theme, setTheme } = useTheme();
  const location = useLocation();
  const queryAnalysisId = new URLSearchParams(location.searchStr).get("analysis");
  const [savedAnalysisId, setSavedAnalysisId] = useState<string | null>(null);
  const currentPath = location.pathname;
  const activePage = navigation.find((item) => item.path === currentPath);
  const analysisId = queryAnalysisId ?? savedAnalysisId;

  useEffect(() => {
    setSavedAnalysisId(window.localStorage.getItem(SAVED_ANALYSIS_KEY));
  }, [currentPath, queryAnalysisId]);

  return (
    <div className="app-shell">
      <aside className="app-sidebar" aria-label="Application sidebar">
        <Link to="/explore" className="app-brand" aria-label="Route Resilience home">
          <span className="app-brand__mark" aria-hidden="true">
            <Network size={18} strokeWidth={2} />
          </span>
          <span className="app-brand__copy">
            <span className="app-brand__name">ROUTE</span>
            <span className="app-brand__name">RESILIENCE</span>
          </span>
        </Link>

        <div className="app-sidebar__section-label">Workspace</div>
        <nav className="app-navigation" aria-label="Main navigation">
          {navigation.map(({ label, path, icon: Icon }) => {
            const active = path === currentPath;
            return (
              <Link
                key={path}
                to={path}
                search={{ analysis: analysisId ?? undefined }}
                aria-current={active ? "page" : undefined}
                className={`app-navigation__link${active ? " is-active" : ""}`}
              >
                <Icon size={16} strokeWidth={1.8} aria-hidden="true" />
                <span>{label}</span>
              </Link>
            );
          })}
        </nav>

        <div className="app-sidebar__footer">
          <span className="app-sidebar__section-label">Platform</span>
          <span className="app-sidebar__footer-copy">Road network intelligence</span>
        </div>
      </aside>

      <div className="app-main">
        <header className="app-topbar">
          <div className="app-topbar__title" aria-live="polite">
            {activePage?.label ?? "Workspace"}
          </div>

          <div className="flex items-center gap-3">
            {analysisId ? (
              <div className="flex items-center gap-2">
                <div className="app-analysis" title={analysisId}>
                  <span className="app-analysis__indicator" aria-hidden="true" />
                  <span className="app-analysis__label hidden sm:inline">Active</span>
                  <span className="app-analysis__id">#{analysisId.slice(0, 8)}</span>
                </div>
                <Link
                  to="/explore"
                  className="text-[0.68rem] font-medium text-muted-foreground hover:text-foreground transition px-2 py-1 rounded-md border border-border bg-card shadow-2xs hover:border-primary/50"
                >
                  Switch / New
                </Link>
              </div>
            ) : (
              <span className="app-topbar__meta hidden sm:inline">Network resilience workspace</span>
            )}

            {/* Scientific Theme Mode Switcher */}
            <div
              className="flex items-center rounded-lg border border-border bg-muted/40 p-0.5 text-muted-foreground"
              role="group"
              aria-label="Theme mode toggle"
            >
              <button
                type="button"
                onClick={() => setTheme("light")}
                className={`flex h-6 w-6 items-center justify-center rounded-md transition-all ${
                  theme === "light"
                    ? "bg-card text-foreground shadow-2xs border border-border/80 font-semibold"
                    : "hover:text-foreground"
                }`}
                title="Light mode"
                aria-pressed={theme === "light"}
              >
                <Sun size={13} strokeWidth={2} />
              </button>
              <button
                type="button"
                onClick={() => setTheme("dark")}
                className={`flex h-6 w-6 items-center justify-center rounded-md transition-all ${
                  theme === "dark"
                    ? "bg-card text-foreground shadow-2xs border border-border/80 font-semibold"
                    : "hover:text-foreground"
                }`}
                title="Dark mode"
                aria-pressed={theme === "dark"}
              >
                <Moon size={13} strokeWidth={2} />
              </button>
              <button
                type="button"
                onClick={() => setTheme("system")}
                className={`flex h-6 w-6 items-center justify-center rounded-md transition-all ${
                  theme === "system"
                    ? "bg-card text-foreground shadow-2xs border border-border/80 font-semibold"
                    : "hover:text-foreground"
                }`}
                title="System theme"
                aria-pressed={theme === "system"}
              >
                <Laptop size={13} strokeWidth={2} />
              </button>
            </div>
          </div>
        </header>
        <main className="app-content">{children}</main>
      </div>
      <Toaster position="bottom-right" />
    </div>
  );
}
