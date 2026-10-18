import { Link } from "@tanstack/react-router";
import { ArrowUpRight, Circle, Network } from "lucide-react";

export function LandingFooter() {
  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <footer className="bg-white border-t border-slate-200/80 py-16 text-slate-600 text-xs">
      <div className="max-w-7xl mx-auto px-6 sm:px-8 lg:px-12">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-10 pb-12 border-b border-slate-100">
          {/* Brand Info */}
          <div className="col-span-2 space-y-4">
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-lg bg-[#0066FF] flex items-center justify-center text-white">
                <Network size={16} strokeWidth={2.2} />
              </div>
              <span className="font-display text-sm font-bold tracking-tight text-[#0A192F]">
                ROUTE RESILIENCE
              </span>
            </div>
            <p className="text-slate-500 max-w-sm text-xs leading-relaxed">
              Transforming raw satellite remote sensing imagery into intelligent, stress-tested
              road network topologies. See what breaks before it does.
            </p>
            <div className="inline-flex items-center gap-2 text-[0.7rem] text-slate-500 bg-slate-50 px-3 py-1.5 rounded-full border border-slate-200/70">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span>Inference Core: Standby & Ready</span>
            </div>
          </div>

          {/* Quick Navigation */}
          <div>
            <h4 className="font-display text-xs font-bold text-[#0A192F] uppercase tracking-wider mb-4">
              Overview
            </h4>
            <ul className="space-y-2.5">
              <li>
                <button
                  type="button"
                  onClick={scrollToTop}
                  className="hover:text-[#0066FF] transition-colors cursor-pointer"
                >
                  Home
                </button>
              </li>
              <li>
                <a href="#technology" className="hover:text-[#0066FF] transition-colors">
                  Technology
                </a>
              </li>
              <li>
                <a href="#scenarios" className="hover:text-[#0066FF] transition-colors">
                  Scenarios
                </a>
              </li>
              <li>
                <a href="#about" className="hover:text-[#0066FF] transition-colors">
                  About
                </a>
              </li>
            </ul>
          </div>

          {/* Workspace Routes */}
          <div>
            <h4 className="font-display text-xs font-bold text-[#0A192F] uppercase tracking-wider mb-4">
              Workspace
            </h4>
            <ul className="space-y-2.5">
              <li>
                <Link to="/explore" className="hover:text-[#0066FF] transition-colors inline-flex items-center gap-1">
                  New Analysis <ArrowUpRight size={11} />
                </Link>
              </li>
              <li>
                <Link to="/network" className="hover:text-[#0066FF] transition-colors">
                  Network Graph
                </Link>
              </li>
              <li>
                <Link to="/disruption" className="hover:text-[#0066FF] transition-colors">
                  Disruption Sim
                </Link>
              </li>
              <li>
                <Link to="/routing" className="hover:text-[#0066FF] transition-colors">
                  Dynamic Routing
                </Link>
              </li>
              <li>
                <Link to="/recovery" className="hover:text-[#0066FF] transition-colors">
                  Recovery Planner
                </Link>
              </li>
            </ul>
          </div>

          {/* Tech Stack */}
          <div>
            <h4 className="font-display text-xs font-bold text-[#0A192F] uppercase tracking-wider mb-4">
              Stack
            </h4>
            <ul className="space-y-2.5 text-slate-500">
              <li>PyTorch & CUDA</li>
              <li>SegFormer & UNet</li>
              <li>NetworkX & Topology</li>
              <li>TanStack Start & Vite</li>
              <li>Tailwind CSS</li>
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="pt-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-[0.72rem] text-slate-400">
          <p>© 2026 Route Resilience. All rights reserved.</p>
          <div className="flex items-center gap-6">
            <span>ISRO BAH 2026 Benchmark</span>
            <span>•</span>
            <button
              type="button"
              onClick={scrollToTop}
              className="hover:text-slate-600 transition-colors cursor-pointer"
            >
              Back to top ↑
            </button>
          </div>
        </div>
      </div>
    </footer>
  );
}
