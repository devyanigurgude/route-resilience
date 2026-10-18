import { ArrowRight, CheckCircle2, FileImage, ShieldCheck, Sparkles } from "lucide-react";
import { Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { SAVED_ANALYSIS_KEY } from "@/lib/backend";

export function FinalCTASection() {
  const [savedId, setSavedId] = useState<string | null>(null);

  useEffect(() => {
    const id = window.localStorage.getItem(SAVED_ANALYSIS_KEY);
    if (id) setSavedId(id);
  }, []);

  return (
    <section className="py-24 sm:py-32 bg-gradient-to-b from-white to-[#F4F6F9] relative overflow-hidden">
      {/* Background soft ambient glows */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[400px] bg-gradient-to-r from-blue-100/40 via-orange-50/40 to-transparent rounded-full blur-3xl pointer-events-none -z-10" />

      <div className="max-w-5xl mx-auto px-6 sm:px-8 text-center">
        {/* Pill Badge */}
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-white border border-slate-200/80 shadow-2xs mb-6">
          <span className="w-2 h-2 rounded-full bg-[#FF5A36]" />
          <span className="font-display text-[0.68rem] font-bold tracking-widest text-slate-800 uppercase">
            OPERATIONAL READINESS
          </span>
        </div>

        {/* Main CTA Heading */}
        <h2 className="font-display text-4xl sm:text-6xl font-bold tracking-tight text-[#0A192F] leading-[1.1] mb-6">
          Build more resilient networks.
        </h2>

        {/* Subtitle */}
        <p className="text-base sm:text-lg text-slate-600 font-normal leading-relaxed max-w-2xl mx-auto mb-10">
          Upload satellite tiles, extract mathematical road topologies, test disaster severance
          scenarios, and safeguard critical transit arteries before extreme weather strikes.
        </p>

        {/* CTA Buttons */}
        <div className="flex flex-wrap items-center justify-center gap-4 mb-12">
          <Link
            to="/explore"
            className="inline-flex items-center gap-2.5 bg-[#FF5A36] hover:bg-[#EA4A26] text-white text-sm font-semibold px-8 py-4 rounded-full shadow-md hover:shadow-lg transition-all duration-200 transform hover:-translate-y-0.5"
          >
            <span>Launch Analysis</span>
            <ArrowRight size={16} strokeWidth={2.2} />
          </Link>
          <Link
            to="/disruption"
            className="inline-flex items-center gap-2 bg-white hover:bg-slate-50 text-slate-800 text-sm font-semibold px-7 py-4 rounded-full border border-slate-200/90 shadow-2xs hover:shadow-sm transition-all duration-200"
          >
            <span>Simulate Disruption</span>
          </Link>
        </div>

        {/* Reassurance Features */}
        <div className="flex flex-wrap items-center justify-center gap-6 sm:gap-10 text-xs font-semibold text-slate-600">
          <div className="flex items-center gap-2">
            <CheckCircle2 size={16} className="text-[#0066FF]" />
            <span>ISRO BAH 2026 Remote Sensing Tested</span>
          </div>
          <div className="flex items-center gap-2">
            <CheckCircle2 size={16} className="text-[#0066FF]" />
            <span>High-Fidelity Graph Topology Engine</span>
          </div>
          <div className="flex items-center gap-2">
            <CheckCircle2 size={16} className="text-[#0066FF]" />
            <span>Zero-Dependency Local Inference</span>
          </div>
        </div>

        {/* Saved Analysis Quick Jump (if user already ran an analysis) */}
        {savedId && (
          <div className="mt-12 inline-flex items-center gap-3 bg-white/95 px-5 py-3 rounded-2xl border border-slate-200/90 shadow-sm text-xs text-slate-600">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            <span>Active session found: <strong className="font-mono text-slate-900">{savedId}</strong></span>
            <Link
              to="/network"
              search={{ analysis: savedId }}
              className="text-[#0066FF] hover:underline font-semibold ml-2 inline-flex items-center gap-1"
            >
              Resume Workspace →
            </Link>
          </div>
        )}
      </div>
    </section>
  );
}
