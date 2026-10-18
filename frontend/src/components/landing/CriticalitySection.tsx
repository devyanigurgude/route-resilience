import { AlertTriangle, ArrowRight, Check, Flame, ShieldAlert, Sparkles } from "lucide-react";
import { Link } from "@tanstack/react-router";
import { useState } from "react";

export function CriticalitySection() {
  const [selectedEdge, setSelectedEdge] = useState<"critical" | "secondary">("critical");

  return (
    <section id="scenarios" className="py-24 sm:py-32 bg-white relative overflow-hidden border-b border-slate-100">
      <div className="max-w-7xl mx-auto px-6 sm:px-8 lg:px-12">
        <div className="grid lg:grid-cols-12 gap-12 lg:gap-16 items-center">
          {/* Left Column: Visual Highlight of Critical Road */}
          <div className="lg:col-span-6 order-2 lg:order-1">
            <div className="relative rounded-3xl border border-slate-200/90 bg-[#FAFBFD] p-6 sm:p-8 shadow-sm">
              <div className="flex items-center justify-between pb-4 border-b border-slate-200/70 mb-6">
                <div>
                  <span className="font-display text-xs font-bold text-slate-400 uppercase tracking-wider block">
                    TOPOLOGICAL STRESS TEST
                  </span>
                  <h4 className="font-display text-base font-bold text-[#0A192F]">
                    Betweenness Centrality Heatmap
                  </h4>
                </div>
                <div className="flex items-center gap-2">
                  <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-orange-100/70 text-[#FF5A36] text-xs font-bold">
                    <span className="w-2 h-2 rounded-full bg-[#FF5A36] animate-pulse" />
                    Critical Bottleneck
                  </span>
                </div>
              </div>

              {/* Graphical representation of the critical corridor */}
              <div className="relative aspect-[16/11] w-full rounded-2xl bg-white border border-slate-200/80 p-6 flex flex-col justify-between overflow-hidden">
                <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 450 300">
                  {/* Western cluster */}
                  <line x1="60" y1="80" x2="140" y2="150" stroke="#CBD5E1" strokeWidth="2.5" />
                  <line x1="60" y1="220" x2="140" y2="150" stroke="#CBD5E1" strokeWidth="2.5" />
                  <line x1="60" y1="80" x2="60" y2="220" stroke="#CBD5E1" strokeWidth="2" />

                  {/* Critical Arterial Link e(104, 109) */}
                  <g>
                    <line
                      x1="140"
                      y1="150"
                      x2="310"
                      y2="150"
                      stroke="#FF5A36"
                      strokeWidth="5"
                      strokeLinecap="round"
                      className="animate-road-pulse"
                    />
                    {/* Pulsing glow aura */}
                    <circle cx="225" cy="150" r="28" fill="#FF5A36" opacity="0.12" className="animate-radar" />
                  </g>

                  {/* Eastern cluster */}
                  <line x1="310" y1="150" x2="390" y2="80" stroke="#CBD5E1" strokeWidth="2.5" />
                  <line x1="310" y1="150" x2="390" y2="220" stroke="#CBD5E1" strokeWidth="2.5" />
                  <line x1="390" y1="80" x2="390" y2="220" stroke="#CBD5E1" strokeWidth="2" />

                  {/* Nodes */}
                  <circle cx="60" cy="80" r="6" fill="#0066FF" />
                  <circle cx="60" cy="220" r="6" fill="#0066FF" />
                  <circle cx="140" cy="150" r="8" fill="#FF5A36" stroke="#FFFFFF" strokeWidth="2" />
                  <circle cx="310" cy="150" r="8" fill="#FF5A36" stroke="#FFFFFF" strokeWidth="2" />
                  <circle cx="390" cy="80" r="6" fill="#0066FF" />
                  <circle cx="390" cy="220" r="6" fill="#0066FF" />
                </svg>

                {/* Overlaid labels */}
                <div className="relative z-10 flex justify-between items-start">
                  <div className="bg-slate-50 border border-slate-200/80 px-2.5 py-1.5 rounded-lg text-[0.68rem] font-bold text-slate-700">
                    Western Region (18 Nodes)
                  </div>
                  <div className="bg-slate-50 border border-slate-200/80 px-2.5 py-1.5 rounded-lg text-[0.68rem] font-bold text-slate-700">
                    Eastern Valley (24 Nodes)
                  </div>
                </div>

                <div className="relative z-10 mx-auto bg-white/95 backdrop-blur-md px-4 py-2 rounded-xl border border-orange-200 shadow-md text-center max-w-[280px]">
                  <p className="font-display text-[0.68rem] font-bold text-[#FF5A36] uppercase tracking-wider">
                    Edge e(104, 109) • Lifeline Bridge
                  </p>
                  <p className="text-[0.72rem] text-slate-600 mt-0.5">
                    Centrality Index: <strong>0.894</strong> (Top 1% Critical)
                  </p>
                </div>

                <div className="relative z-10 flex items-center justify-between text-[0.7rem] text-slate-500 pt-2 border-t border-slate-100">
                  <span>Severance Impact: <strong>38% Network Disconnection</strong></span>
                  <span className="text-[#FF5A36] font-semibold">Priority 1 Reinforcement</span>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Narrative Storytelling */}
          <div className="lg:col-span-6 order-1 lg:order-2 space-y-6">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-orange-50 border border-orange-200/50 text-[#FF5A36] font-display text-[0.66rem] font-bold tracking-widest uppercase">
              <span>CRITICALITY ANALYSIS</span>
            </div>

            <h2 className="font-display text-3xl sm:text-5xl font-bold tracking-tight text-[#0A192F] leading-tight">
              Not all roads are created equal.
            </h2>

            <p className="text-base sm:text-lg text-slate-600 font-normal leading-relaxed">
              In disaster zones, 5% of network corridors often carry over 60% of regional accessibility.
              Route Resilience automatically identifies the structural bridges and mountain passes that,
              if severed by floods or landslides, isolate entire hospitals, towns, and evacuation routes.
            </p>

            {/* Value bullets */}
            <div className="space-y-3.5 pt-2">
              <div className="flex items-start gap-3">
                <div className="w-5 h-5 rounded-full bg-blue-50 text-[#0066FF] flex items-center justify-center shrink-0 mt-0.5">
                  <Check size={13} strokeWidth={2.5} />
                </div>
                <div>
                  <strong className="text-sm font-semibold text-[#0A192F] block">
                    Betweenness Centrality Computation
                  </strong>
                  <span className="text-xs text-slate-600">
                    Measures how many shortest path journeys pass through each individual road segment.
                  </span>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <div className="w-5 h-5 rounded-full bg-orange-50 text-[#FF5A36] flex items-center justify-center shrink-0 mt-0.5">
                  <Check size={13} strokeWidth={2.5} />
                </div>
                <div>
                  <strong className="text-sm font-semibold text-[#0A192F] block">
                    Single-Point-of-Failure Isolation
                  </strong>
                  <span className="text-xs text-slate-600">
                    Flags bridges and narrow valley corridors that lack immediate redundant paths.
                  </span>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <div className="w-5 h-5 rounded-full bg-blue-50 text-[#0066FF] flex items-center justify-center shrink-0 mt-0.5">
                  <Check size={13} strokeWidth={2.5} />
                </div>
                <div>
                  <strong className="text-sm font-semibold text-[#0A192F] block">
                    Proactive Infrastructure Hardening
                  </strong>
                  <span className="text-xs text-slate-600">
                    Provides municipal engineers with quantitative justification for culvert reinforcement and slope stabilization.
                  </span>
                </div>
              </div>
            </div>

            <div className="pt-4">
              <Link
                to="/disruption"
                className="inline-flex items-center gap-2 text-xs font-semibold text-[#0066FF] hover:text-blue-700 tracking-wide group"
              >
                <span>Simulate a failure scenario in the workspace</span>
                <ArrowRight size={14} className="group-hover:translate-x-1 transition-transform" />
              </Link>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
