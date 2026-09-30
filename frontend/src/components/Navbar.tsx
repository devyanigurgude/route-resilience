import { Link, useLocation } from "@tanstack/react-router";
import {
  Activity,
  BarChart3,
  CheckCircle2,
  CircleGauge,
  FileImage,
  Home,
  Menu,
  Network,
  Route as RouteIcon,
  Settings,
  ShieldAlert,
  X,
} from "lucide-react";
import { useState } from "react";

export function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();
  const pathname = location.pathname;

  const navLinks = [
    { label: "Home", to: "/", exact: true, icon: Home },
    { label: "Analyze", to: "/explore", icon: FileImage },
    { label: "Network", to: "/network", icon: Network },
    { label: "Criticality", to: "/network", icon: BarChart3 },
    { label: "Disruption", to: "/disruption", icon: ShieldAlert },
    { label: "Routing", to: "/routing", icon: RouteIcon },
    { label: "Recovery", to: "/recovery", icon: CheckCircle2 },
  ];

  const isLinkActive = (to: string, exact?: boolean) => {
    if (exact) {
      return pathname === to;
    }
    return pathname === to || pathname.startsWith(to + "/");
  };

  return (
    <>
      <aside
        className="route-resilience-nav fixed inset-y-0 left-0 z-40 hidden w-64 flex-col border-r border-sidebar-border bg-sidebar md:flex"
        aria-label="Primary navigation"
      >
        <div className="border-b border-sidebar-border px-5 py-5">
          <Link to="/" className="flex items-center gap-3 text-sidebar-foreground">
            <span className="flex h-8 w-8 items-center justify-center border border-sidebar-primary/60 bg-sidebar-primary/10 text-sidebar-primary">
              <CircleGauge size={17} />
            </span>
            <span>
              <strong className="block font-display text-sm tracking-[0.12em]">ROUTE</strong>
              <span className="block font-display text-[0.62rem] tracking-[0.2em] text-sidebar-foreground/55">
                RESILIENCE
              </span>
            </span>
          </Link>
        </div>
        <div className="px-3 pt-6">
          <p className="px-3 pb-2 font-display text-[0.56rem] tracking-[0.2em] text-sidebar-foreground/35 uppercase">
            WORKSPACE
          </p>
          <div className="space-y-1">
            {navLinks.map((item) => {
              const active = isLinkActive(item.to, item.exact);
              const Icon = item.icon;
              return (
                <Link
                  key={item.label}
                  to={item.to}
                  className={`group flex items-center gap-3 border-l-2 px-3 py-2.5 font-display text-[0.66rem] tracking-[0.08em] transition-colors ${active ? "border-sidebar-primary bg-sidebar-accent text-sidebar-primary" : "border-transparent text-sidebar-foreground/58 hover:bg-sidebar-accent/60 hover:text-sidebar-foreground"}`}
                >
                  <Icon size={15} strokeWidth={1.7} />
                  <span>{item.label}</span>
                  {active && <span className="ml-auto h-1.5 w-1.5 bg-sidebar-primary" />}
                </Link>
              );
            })}
          </div>
        </div>
        <div className="mt-auto border-t border-sidebar-border p-4">
          <Link
            to="/"
            className="flex items-center gap-3 border border-sidebar-border bg-sidebar-accent/40 p-3 text-sidebar-foreground/65 hover:text-sidebar-foreground"
          >
            <span className="flex h-7 w-7 items-center justify-center bg-emerald-500/15 text-emerald-400">
              <Activity size={14} />
            </span>
            <span className="min-w-0">
              <span className="block truncate font-display text-[0.62rem] text-sidebar-foreground">
                SYSTEM ONLINE
              </span>
              <span className="block truncate text-[0.58rem] text-sidebar-foreground/40">
                Local analysis workspace
              </span>
            </span>
            <Settings size={14} className="ml-auto text-sidebar-foreground/35" />
          </Link>
        </div>
      </aside>
      <nav
        className="route-resilience-mobile-nav relative z-30 flex items-center justify-between border-b border-border/50 bg-background/95 px-5 py-4 backdrop-blur-md md:hidden"
        aria-label="Primary navigation"
      >
        <Link to="/" className="flex items-center gap-2 text-primary">
          <CircleGauge size={18} />
          <span className="font-display text-[0.7rem] font-semibold tracking-[0.15em]">
            ROUTE RESILIENCE
          </span>
        </Link>
        <button
          type="button"
          aria-label={menuOpen ? "Close navigation menu" : "Open navigation menu"}
          aria-expanded={menuOpen}
          onClick={() => setMenuOpen((open) => !open)}
          className="flex h-9 w-9 items-center justify-center border border-border/60 text-primary"
        >
          {menuOpen ? <X size={17} /> : <Menu size={17} />}
        </button>
        {menuOpen && (
          <div className="absolute left-0 right-0 top-full border-b border-border/60 bg-background p-3 shadow-xl">
            {navLinks.map((item) => {
              const active = isLinkActive(item.to, item.exact);
              const Icon = item.icon;
              return (
                <Link
                  key={item.label}
                  to={item.to}
                  onClick={() => setMenuOpen(false)}
                  className={`flex items-center gap-3 px-3 py-3 font-display text-[0.65rem] tracking-[0.1em] ${active ? "bg-primary/10 text-primary" : "text-foreground/65"}`}
                >
                  <Icon size={15} />
                  {item.label}
                </Link>
              );
            })}
          </div>
        )}
      </nav>
    </>
  );
}
