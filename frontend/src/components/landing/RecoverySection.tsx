import { ArrowRight, CheckCircle2, Compass, GitCommit, Navigation, ShieldCheck, Sparkles } from "lucide-react";
import { Link } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";

export function RecoverySection() {
  const [detourActive, setDetourActive] = useState(true);
  const [isVisible, setIsVisible] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.25 }
    );
    if (containerRef.current) observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  return (
    <section ref={containerRef} className="py-24 sm:py-32 bg-white relative overflow-hidden border-b border-slate-100">
      <div className="max-w-7xl mx-auto px-6 sm:px-8 lg:px-12">
        <div className="grid lg:grid-cols-12 gap-12 lg:gap-16 items-center">
          {/* Left Column: Narrative & Metrics */}
          <div className="lg:col-span-6 space-y-6">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-50 border border-emerald-200/50 text-emerald-600 font-display text-[0.66rem] font-bold tracking-widest uppercase">
              <span>RECOVERY & RESILIENCE</span>
            </div>

            <h2 className="font-display text-3xl sm:text-5xl font-bold tracking-tight text-[#0A192F] leading-tight">
              Dynamic alternate routes & automated recovery.
            </h2>

            <p className="text-base sm:text-lg text-slate-600 font-normal leading-relaxed">
              When a lifeline link collapses, emergency responders cannot wait hours for manual surveys.
              Route Resilience instantly computes viable alternate detours, calculates detour penalties,
              and generates greedy restoration sequences within maintenance budgets.
            </p>

            {/* Quick Metrics Bar */}
            <div className="grid grid-cols-2 gap-4 pt-2">
              <div className="p-4 rounded-2xl bg-[#FAFBFD] border border-slate-200/80">
                <span className="text-xs text-slate-500 font-medium block">Resilience Score:</span>
                <span className="font-display text-2xl font-bold text-[#0066FF] mt-1 block">
                  88.6 <span className="text-sm font-normal text-slate-400">/ 100</span>
                </span>
                <span className="text-[0.68rem] text-emerald-600 font-medium mt-1 inline-flex items-center gap-1">
                  <CheckCircle2 size={11} /> High Adaptability
                </span>
              </div>

              <div className="p-4 rounded-2xl bg-[#FAFBFD] border border-slate-200/80">
                <span className="text-xs text-slate-500 font-medium block">Computed Detour:</span>
                <span className="font-display text-2xl font-bold text-[#0A192F] mt-1 block">
                  +24.1% <span className="text-sm font-normal text-slate-400">dist</span>
                </span>
                <span className="text-[0.68rem] text-slate-500 mt-1 block">
                  12.4 km → 15.4 km bypass
                </span>
              </div>
            </div>

            {/* Restoration Sequence Steps */}
            <div className="space-y-3 pt-2">
              <div className="flex items-center justify-between p-3.5 rounded-xl border border-slate-200 bg-white">
                <div className="flex items-center gap-3">
                  <span className="w-6 h-6 rounded-full bg-blue-50 text-[#0066FF] text-xs font-bold flex items-center justify-center">
                    1
                  </span>
                  <div>
                    <strong className="text-xs font-semibold text-[#0A192F] block">
                      Clear Primary Valley Pass
                    </strong>
                    <span className="text-[0.68rem] text-slate-500">
                      Edge e(104, 109) • Restores 86% of disconnected service
                    </span>
                  </div>
                </div>
                <span className="text-xs font-bold text-emerald-600 bg-emerald-50 px-2 py-1 rounded">
                  +0.28 Eff
                </span>
              </div>

              <div className="flex items-center justify-between p-3.5 rounded-xl border border-slate-200 bg-white">
                <div className="flex items-center gap-3">
                  <span className="w-6 h-6 rounded-full bg-slate-100 text-slate-600 text-xs font-bold flex items-center justify-center">
                    2
                  </span>
                  <div>
                    <strong className="text-xs font-semibold text-[#0A192F] block">
                      Reinforce Southern Bypass Culvert
                    </strong>
                    <span className="text-[0.68rem] text-slate-500">
                      Edge e(180, 210) • Secondary emergency corridor
                    </span>
                  </div>
                </div>
                <span className="text-xs font-bold text-slate-600 bg-slate-100 px-2 py-1 rounded">
                  +0.03 Eff
                </span>
              </div>
            </div>

            <div className="pt-2">
              <Link
                to="/recovery"
                className="inline-flex items-center gap-2 text-xs font-semibold text-[#0066FF] hover:text-blue-700 tracking-wide group"
              >
                <span>Explore multi-step budget recovery planner</span>
                <ArrowRight size={14} className="group-hover:translate-x-1 transition-transform" />
              </Link>
            </div>
          </div>

          {/* Right Column: Visual Routing Canvas with Drawing Detour */}
          <div className="lg:col-span-6">
            <div className="rounded-3xl border border-slate-200/90 bg-[#FAFBFD] p-6 sm:p-8 shadow-sm">
              <div className="flex items-center justify-between pb-4 border-b border-slate-200/70 mb-5">
                <div>
                  <span className="font-display text-xs font-bold text-slate-400 uppercase tracking-wider block">
                    DYNAMIC ROUTING ENGINE
                  </span>
                  <h4 className="font-display text-base font-bold text-[#0A192F]">
                    Alternate Pathfinding Simulation
                  </h4>
                </div>
                <button
                  type="button"
                  onClick={() => setDetourActive(!detourActive)}
                  className="px-3 py-1 rounded-full text-xs font-semibold bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 cursor-pointer shadow-2xs"
                >
                  {detourActive ? "Hide Detour" : "Show Detour"}
                </button>
              </div>

              {/* Dynamic Route SVG Map */}
              <div className="relative aspect-[16/11] w-full rounded-2xl bg-white border border-slate-200/80 p-6 flex flex-col justify-between overflow-hidden">
                <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 500 320">
                  {/* Background Road Grid */}
                  <g stroke="#E2E8F0" strokeWidth="2" strokeLinecap="round">
                    <line x1="80" y1="160" x2="160" y2="90" />
                    <line x1="80" y1="160" x2="160" y2="230" />
                    <line x1="160" y1="90" x2="340" y2="90" />
                    <line x1="160" y1="230" x2="340" y2="230" />
                    <line x1="340" y1="90" x2="420" y2="160" />
                    <line x1="340" y1="230" x2="420" y2="160" />
                  </g>

                  {/* Severed Central Corridor */}
                  <g>
                    <line
                      x1="160"
                      y1="160"
                      x2="340"
                      y2="160"
                      stroke="#EF4444"
                      strokeWidth="2.5"
                      strokeDasharray="4 6"
                      opacity="0.4"
                    />
                    <circle cx="250" cy="160" r="12" fill="#FEF2F2" stroke="#EF4444" strokeWidth="1.5" />
                    <text x="250" y="164" textAnchor="middle" fill="#EF4444" fontSize="10" fontWeight="bold">
                      ✕
                    </text>
                  </g>

                  {/* Alternate Detour Route (Draws itself in Emerald/Cyan!) */}
                  {detourActive && isVisible && (
                    <g>
                      <path
                        d="M 80 160 L 160 230 L 340 230 L 420 160"
                        stroke="#10B981"
                        strokeWidth="4.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        fill="none"
                        className="animate-road-draw"
                      />
                      {/* Pulse marker along the detour */}
                      <circle cx="250" cy="230" r="16" fill="#10B981" opacity="0.15" className="animate-radar" />
                    </g>
                  )}

                  {/* Node Indicators */}
                  {/* Origin */}
                  <g transform="translate(80, 160)">
                    <circle r="9" fill="#0066FF" stroke="#FFFFFF" strokeWidth="2.5" />
                    <text y="-14" textAnchor="middle" fill="#0066FF" fontSize="10" fontWeight="bold">
                      ORIGIN
                    </text>
                  </g>

                  {/* Destination */}
                  <g transform="translate(420, 160)">
                    <circle r="9" fill="#10B981" stroke="#FFFFFF" strokeWidth="2.5" />
                    <text y="-14" textAnchor="middle" fill="#10B981" fontSize="10" fontWeight="bold">
                      DESTINATION
                    </text>
                  </g>

                  {/* Intermediate Waypoints */}
                  <circle cx="160" cy="90" r="5" fill="#94A3B8" />
                  <circle cx="340" cy="90" r="5" fill="#94A3B8" />
                  <circle cx="160" cy="230" r="7" fill={detourActive ? "#10B981" : "#94A3B8"} stroke="#FFFFFF" strokeWidth="2" />
                  <circle cx="340" cy="230" r="7" fill={detourActive ? "#10B981" : "#94A3B8"} stroke="#FFFFFF" strokeWidth="2" />
                </svg>

                {/* Overlaid Route Info Cards */}
                <div className="relative z-10 flex items-center justify-between">
                  <span className="text-[0.68rem] font-bold text-slate-500 uppercase">
                    Route Query: N-42 → N-145
                  </span>
                  <span className="text-[0.68rem] font-semibold text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded">
                    Bypass Verified
                  </span>
                </div>

                <div className="relative z-10 grid grid-cols-2 gap-3 bg-white/95 backdrop-blur-md p-3 rounded-xl border border-slate-200/90 shadow-2xs">
                  <div>
                    <span className="text-[0.62rem] text-slate-400 block uppercase">Blocked Route:</span>
                    <span className="text-xs font-semibold text-slate-700 line-through">12.4 km (Direct)</span>
                  </div>
                  <div>
                    <span className="text-[0.62rem] text-emerald-600 block uppercase font-bold">Rerouted:</span>
                    <span className="text-xs font-bold text-emerald-600">15.4 km (+24.1%)</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
