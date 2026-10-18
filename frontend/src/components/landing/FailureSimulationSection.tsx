import { AlertOctagon, AlertTriangle, ArrowRight, CheckCircle2, RotateCcw, ShieldAlert, Zap } from "lucide-react";
import { Link } from "@tanstack/react-router";
import { useState } from "react";

type SimPhase = "before" | "failure" | "after";

export function FailureSimulationSection() {
  const [phase, setPhase] = useState<SimPhase>("before");

  return (
    <section className="py-24 sm:py-32 bg-[#FAFBFD] relative overflow-hidden border-b border-slate-100">
      <div className="max-w-7xl mx-auto px-6 sm:px-8 lg:px-12">
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto mb-16 sm:mb-20">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-orange-50 border border-orange-200/50 text-[#FF5A36] font-display text-[0.66rem] font-bold tracking-widest uppercase mb-4">
            <span>PERCOLATION MODELING</span>
          </div>
          <h2 className="font-display text-3xl sm:text-5xl font-bold tracking-tight text-[#0A192F] leading-tight mb-4">
            Stress-testing under catastrophic events.
          </h2>
          <p className="text-base sm:text-lg text-slate-600 font-normal leading-relaxed">
            Simulate flash floods, slope failures, or earthquake disruptions. Observe how targeted edge
            severances trigger cascading network degradation across entire districts.
          </p>
        </div>

        {/* Interactive 3-Phase Controller Bar */}
        <div className="max-w-xl mx-auto mb-10 p-1.5 rounded-full bg-white border border-slate-200 shadow-xs flex items-center justify-between">
          <button
            type="button"
            onClick={() => setPhase("before")}
            className={`flex-1 py-2.5 px-4 rounded-full text-xs font-semibold transition-all cursor-pointer ${
              phase === "before"
                ? "bg-[#0066FF] text-white shadow-xs"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            1. Before (Baseline)
          </button>
          <button
            type="button"
            onClick={() => setPhase("failure")}
            className={`flex-1 py-2.5 px-4 rounded-full text-xs font-semibold transition-all cursor-pointer ${
              phase === "failure"
                ? "bg-[#FF5A36] text-white shadow-xs"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            2. Failure Event
          </button>
          <button
            type="button"
            onClick={() => setPhase("after")}
            className={`flex-1 py-2.5 px-4 rounded-full text-xs font-semibold transition-all cursor-pointer ${
              phase === "after"
                ? "bg-slate-900 text-white shadow-xs"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            3. Aftermath Impact
          </button>
        </div>

        {/* Simulation Interactive Card */}
        <div className="grid lg:grid-cols-12 gap-8 items-center bg-white rounded-3xl border border-slate-200/90 p-6 sm:p-10 shadow-sm">
          {/* Visual Canvas (8 cols) */}
          <div className="lg:col-span-8">
            <div className="relative aspect-[16/10] w-full rounded-2xl bg-slate-950 p-6 flex flex-col justify-between overflow-hidden shadow-inner">
              {/* Background Map Grid */}
              <div className="absolute inset-0 bg-grid-subtle opacity-5 pointer-events-none" />

              {/* Status Header inside canvas */}
              <div className="relative z-10 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span
                    className={`w-2.5 h-2.5 rounded-full ${
                      phase === "before"
                        ? "bg-emerald-400"
                        : phase === "failure"
                        ? "bg-[#FF5A36] animate-ping"
                        : "bg-amber-400"
                    }`}
                  />
                  <span className="text-xs font-display font-bold tracking-wider text-slate-300 uppercase">
                    {phase === "before" && "STATUS: BASELINE ROAD NETWORK INTACT"}
                    {phase === "failure" && "ALERT: LANDSLIDE IMPACT AT ARTERIAL PASS"}
                    {phase === "after" && "DISRUPTED: NETWORK FRAGMENTED INTO 2 COMPONENTS"}
                  </span>
                </div>

                <button
                  type="button"
                  onClick={() => setPhase(phase === "before" ? "failure" : phase === "failure" ? "after" : "before")}
                  className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md bg-slate-800 text-slate-300 hover:text-white text-xs font-medium cursor-pointer"
                >
                  <RotateCcw size={12} />
                  <span>Next State</span>
                </button>
              </div>

              {/* Interactive SVG Network */}
              <div className="relative z-10 my-auto">
                <svg viewBox="0 0 700 320" className="w-full h-auto">
                  {/* Intact Western Subgraph */}
                  <g stroke="#38BDF8" strokeWidth="3" strokeLinecap="round">
                    <line x1="100" y1="80" x2="220" y2="160" />
                    <line x1="100" y1="240" x2="220" y2="160" />
                    <line x1="100" y1="80" x2="100" y2="240" stroke="#1E293B" strokeWidth="2" />
                    <line x1="60" y1="160" x2="220" y2="160" stroke="#1E293B" strokeWidth="2" />
                  </g>

                  {/* Intact Eastern Subgraph */}
                  <g
                    stroke={phase === "after" ? "#F59E0B" : "#38BDF8"}
                    strokeWidth="3"
                    strokeLinecap="round"
                    className="transition-colors duration-500"
                  >
                    <line x1="480" y1="160" x2="600" y2="80" />
                    <line x1="480" y1="160" x2="600" y2="240" />
                    <line x1="600" y1="80" x2="600" y2="240" stroke="#1E293B" strokeWidth="2" />
                    <line x1="480" y1="160" x2="640" y2="160" stroke="#1E293B" strokeWidth="2" />
                  </g>

                  {/* Central Critical Edge (The one that breaks!) */}
                  {phase === "before" && (
                    <g>
                      <line
                        x1="220"
                        y1="160"
                        x2="480"
                        y2="160"
                        stroke="#0066FF"
                        strokeWidth="5"
                        strokeLinecap="round"
                      />
                      <circle cx="350" cy="160" r="10" fill="#0066FF" opacity="0.3" className="animate-radar" />
                    </g>
                  )}

                  {phase === "failure" && (
                    <g>
                      {/* Broken fracture lines */}
                      <line
                        x1="220"
                        y1="160"
                        x2="320"
                        y2="155"
                        stroke="#FF5A36"
                        strokeWidth="4"
                        strokeLinecap="round"
                        className="animate-road-broken"
                      />
                      <line
                        x1="380"
                        y1="165"
                        x2="480"
                        y2="160"
                        stroke="#FF5A36"
                        strokeWidth="4"
                        strokeLinecap="round"
                        className="animate-road-broken"
                      />
                      {/* Blast / rupture indicator */}
                      <circle cx="350" cy="160" r="18" fill="#FF5A36" opacity="0.2" className="animate-radar" />
                      <line x1="335" y1="145" x2="365" y2="175" stroke="#FF5A36" strokeWidth="3" />
                      <line x1="365" y1="145" x2="335" y2="175" stroke="#FF5A36" strokeWidth="3" />
                    </g>
                  )}

                  {phase === "after" && (
                    <g>
                      {/* Severed gap */}
                      <line
                        x1="220"
                        y1="160"
                        x2="480"
                        y2="160"
                        stroke="#EF4444"
                        strokeWidth="2"
                        strokeDasharray="4 8"
                        opacity="0.3"
                      />
                      {/* Warning mark */}
                      <circle cx="350" cy="160" r="14" fill="#1E293B" stroke="#EF4444" strokeWidth="2" />
                      <text x="350" y="165" textAnchor="middle" fill="#EF4444" fontSize="12" fontWeight="bold">
                        ✕
                      </text>
                    </g>
                  )}

                  {/* Nodes */}
                  <circle cx="100" cy="80" r="6" fill="#38BDF8" />
                  <circle cx="100" cy="240" r="6" fill="#38BDF8" />
                  <circle cx="60" cy="160" r="5" fill="#38BDF8" />
                  <circle
                    cx="220"
                    cy="160"
                    r="8"
                    fill={phase === "before" ? "#0066FF" : "#FF5A36"}
                    stroke="#FFFFFF"
                    strokeWidth="2"
                  />

                  <circle
                    cx="480"
                    cy="160"
                    r="8"
                    fill={phase === "before" ? "#0066FF" : phase === "failure" ? "#FF5A36" : "#F59E0B"}
                    stroke="#FFFFFF"
                    strokeWidth="2"
                  />
                  <circle cx="600" cy="80" r="6" fill={phase === "after" ? "#F59E0B" : "#38BDF8"} />
                  <circle cx="600" cy="240" r="6" fill={phase === "after" ? "#F59E0B" : "#38BDF8"} />
                  <circle cx="640" cy="160" r="5" fill={phase === "after" ? "#F59E0B" : "#38BDF8"} />
                </svg>
              </div>

              {/* Bottom Canvas Telemetry Strip */}
              <div className="relative z-10 flex items-center justify-between text-xs text-slate-400 pt-3 border-t border-slate-900">
                <span>Target Corridor: <strong>Valley Pass Route 4</strong></span>
                <span className={phase === "before" ? "text-emerald-400" : "text-[#FF5A36]"}>
                  {phase === "before" && "Flow Capacity: 100% (No Impediments)"}
                  {phase === "failure" && "Structural Fracture in Progress..."}
                  {phase === "after" && "Flow Capacity: 58% (-42% Drop)"}
                </span>
              </div>
            </div>
          </div>

          {/* Right Metrics Panel (4 cols) */}
          <div className="lg:col-span-4 space-y-6">
            <div>
              <p className="font-display text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">
                DISRUPTION TELEMETRY
              </p>
              <h3 className="font-display text-2xl font-bold text-[#0A192F]">
                {phase === "before" && "Baseline Network State"}
                {phase === "failure" && "Catastrophic Severance"}
                {phase === "after" && "Systemic Vulnerability"}
              </h3>
            </div>

            <div className="space-y-4">
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-slate-500 font-medium">Global Efficiency:</span>
                  <span
                    className={`font-display text-base font-bold ${
                      phase === "before"
                        ? "text-emerald-600"
                        : phase === "failure"
                        ? "text-[#FF5A36]"
                        : "text-amber-600"
                    }`}
                  >
                    {phase === "before" && "0.742 (Optimal)"}
                    {phase === "failure" && "0.431 (-41.9%)"}
                    {phase === "after" && "0.431 (-41.9%)"}
                  </span>
                </div>
                <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-500 ${
                      phase === "before" ? "bg-emerald-500 w-full" : "bg-[#FF5A36] w-[58%]"
                    }`}
                  />
                </div>
              </div>

              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-500 font-medium">Connected Components:</span>
                  <span className="font-display text-sm font-bold text-[#0A192F]">
                    {phase === "before" ? "1 (Fully Unified)" : "2 (Isolated District)"}
                  </span>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-500 font-medium">Severed OD Trips:</span>
                  <span
                    className={`font-display text-sm font-bold ${
                      phase === "before" ? "text-slate-400" : "text-[#FF5A36]"
                    }`}
                  >
                    {phase === "before" ? "0 pairs" : "38.2% of routes"}
                  </span>
                </div>
              </div>
            </div>

            <Link
              to="/disruption"
              className="inline-flex items-center gap-2 text-xs font-semibold text-[#FF5A36] hover:text-[#EA4A26] transition-colors"
            >
              <span>Open live Disruption Simulator</span>
              <ArrowRight size={13} />
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}
