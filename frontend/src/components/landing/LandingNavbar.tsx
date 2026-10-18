import { Link } from "@tanstack/react-router";
import { Network } from "lucide-react";

export function LandingNavbar() {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 py-5 sm:py-6 pointer-events-none">
      <div className="max-w-7xl mx-auto px-6 sm:px-8 lg:px-12 flex items-center justify-between">
        {/* Minimal Brand & Logo only */}
        <Link
          to="/"
          className="pointer-events-auto inline-flex items-center gap-3 text-left group cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 rounded-lg py-1 px-1.5 transition-all duration-200"
          aria-label="Route Resilience home"
        >
          <div className="w-8 h-8 rounded-lg bg-[#0066FF] flex items-center justify-center text-white shadow-lg shadow-blue-500/35 group-hover:scale-105 group-hover:bg-[#0052cc] transition-all duration-200">
            <Network size={18} strokeWidth={2.2} />
          </div>
          <div>
            <span className="block font-display text-sm font-bold tracking-tight text-white leading-tight drop-shadow-[0_2px_8px_rgba(0,0,0,0.85)]">
              ROUTE RESILIENCE
            </span>
            <span className="block text-[0.62rem] font-medium tracking-widest text-slate-300 uppercase drop-shadow-[0_1px_4px_rgba(0,0,0,0.85)]">
              Network Intelligence
            </span>
          </div>
        </Link>
      </div>
    </header>
  );
}
