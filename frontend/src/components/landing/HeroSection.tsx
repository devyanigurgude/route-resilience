import { Link } from "@tanstack/react-router";
import { ArrowRight, Eye, Layers, Network, Play, Sparkles } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import terrainAsset from "@/assets/route-resilience-terrain.jpg";

interface HeroSectionProps {
  onWatchDemo: () => void;
}

export function HeroSection({ onWatchDemo }: HeroSectionProps) {
  const [activeLayer, setActiveLayer] = useState<"both" | "satellite" | "network">("both");
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width - 0.5;
    const y = (e.clientY - rect.top) / rect.height - 0.5;
    setMousePos({ x, y });
  };

  const handleMouseLeave = () => {
    setMousePos({ x: 0, y: 0 });
  };

  return (
    <section
      id="top"
      className="relative pt-32 pb-20 sm:pt-36 sm:pb-28 lg:pt-40 lg:pb-32 overflow-hidden bg-[#FAFBFD]"
    >
      {/* Subtle background radial glows */}
      <div className="absolute top-10 left-1/2 -translate-x-1/2 w-[1000px] h-[500px] bg-gradient-to-b from-blue-50/70 via-indigo-50/30 to-transparent rounded-full blur-3xl pointer-events-none -z-10" />
      <div className="absolute top-40 right-10 w-96 h-96 bg-orange-50/40 rounded-full blur-3xl pointer-events-none -z-10" />

      <div className="max-w-7xl mx-auto px-6 sm:px-8 lg:px-12">
        {/* Top Tagline Badge */}
        <div className="flex flex-col items-center text-center max-w-4xl mx-auto">
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-blue-50/90 border border-blue-200/60 shadow-2xs mb-6 transition-all hover:bg-blue-100/70">
            <span className="w-2 h-2 rounded-full bg-[#0066FF] animate-pulse" />
            <span className="font-display text-[0.68rem] font-bold tracking-widest text-[#0066FF] uppercase">
              ROUTE RESILIENCE
            </span>
            <span className="text-slate-300">|</span>
            <span className="text-[0.72rem] font-medium text-slate-600">
              Satellite Network Topology Engine
            </span>
          </div>

          {/* Main Headline */}
          <h1 className="font-display text-4xl sm:text-6xl lg:text-7xl font-bold tracking-tight text-[#0A192F] leading-[1.08] mb-6">
            See what breaks <br className="hidden sm:inline" />
            <span className="bg-gradient-to-r from-[#0A192F] via-[#0066FF] to-[#0A192F] bg-clip-text text-transparent">
              before it does.
            </span>
          </h1>

          {/* Subtitle */}
          <p className="text-lg sm:text-xl text-slate-600 font-normal leading-relaxed max-w-2xl mb-10">
            Ingest satellite imagery, extract deep learning road topologies, and stress-test
            critical lifelines against climate disruptions before disasters sever vital communities.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-wrap items-center justify-center gap-4 mb-16 sm:mb-20">
            <Link
              to="/explore"
              className="inline-flex items-center gap-2.5 bg-[#FF5A36] hover:bg-[#EA4A26] text-white text-sm font-semibold px-7 py-3.5 rounded-full shadow-md hover:shadow-lg transition-all duration-200 transform hover:-translate-y-0.5"
            >
              <span>Explore the Platform</span>
              <ArrowRight size={16} strokeWidth={2.2} />
            </Link>
            <button
              type="button"
              onClick={onWatchDemo}
              className="inline-flex items-center gap-2.5 bg-white hover:bg-slate-50 text-slate-800 text-sm font-semibold px-6 py-3.5 rounded-full border border-slate-200/90 shadow-2xs hover:shadow-sm transition-all duration-200 cursor-pointer"
            >
              <div className="w-5 h-5 rounded-full bg-blue-50 text-[#0066FF] flex items-center justify-center">
                <Play size={10} className="ml-0.5 fill-[#0066FF]" />
              </div>
              <span>Watch Demo</span>
            </button>
          </div>
        </div>

        {/* Hero Visual Panel: Rounded Satellite Panel with Animated Road Overlay */}
        <div
          ref={containerRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          className="relative max-w-5xl mx-auto rounded-3xl p-2.5 sm:p-3 bg-white/80 border border-slate-200/90 shadow-2xl shadow-slate-200/70 backdrop-blur-md transition-transform duration-300 ease-out"
          style={{
            transform: `perspective(1000px) rotateX(${mousePos.y * -4}deg) rotateY(${mousePos.x * 4}deg)`,
          }}
        >
          {/* Card Top Utility Bar */}
          <div className="flex items-center justify-between px-4 py-2.5 border-b border-slate-100 bg-slate-50/70 rounded-t-2xl mb-2">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-slate-300" />
              <span className="w-2.5 h-2.5 rounded-full bg-slate-300" />
              <span className="w-2.5 h-2.5 rounded-full bg-slate-300" />
              <span className="ml-2 font-display text-[0.66rem] font-semibold text-slate-500 uppercase tracking-wider">
                Earth Observation Tile • Lat 18.5204° N / Long 73.8567° E
              </span>
            </div>

            {/* Layer Controller */}
            <div className="flex items-center bg-white rounded-lg p-1 border border-slate-200 shadow-2xs">
              <button
                type="button"
                onClick={() => setActiveLayer("satellite")}
                className={`px-2.5 py-1 text-[0.66rem] font-semibold rounded transition-colors cursor-pointer ${
                  activeLayer === "satellite"
                    ? "bg-slate-100 text-[#0A192F]"
                    : "text-slate-500 hover:text-slate-800"
                }`}
              >
                Satellite
              </button>
              <button
                type="button"
                onClick={() => setActiveLayer("both")}
                className={`px-2.5 py-1 text-[0.66rem] font-semibold rounded transition-colors cursor-pointer ${
                  activeLayer === "both"
                    ? "bg-[#0066FF] text-white"
                    : "text-slate-500 hover:text-slate-800"
                }`}
              >
                Detected Roads
              </button>
              <button
                type="button"
                onClick={() => setActiveLayer("network")}
                className={`px-2.5 py-1 text-[0.66rem] font-semibold rounded transition-colors cursor-pointer ${
                  activeLayer === "network"
                    ? "bg-slate-900 text-white"
                    : "text-slate-500 hover:text-slate-800"
                }`}
              >
                Graph Only
              </button>
            </div>
          </div>

          {/* Main Visual Frame */}
          <div className="relative aspect-[16/10] sm:aspect-[16/9] w-full rounded-2xl overflow-hidden bg-slate-950">
            {/* Satellite Image Layer */}
            <img
              src={terrainAsset}
              alt="Satellite terrain imagery with road network"
              className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-500 ${
                activeLayer === "network" ? "opacity-15 grayscale" : "opacity-100"
              }`}
            />

            {/* Gradient subtle vignette to soften satellite borders */}
            <div className="absolute inset-0 bg-gradient-to-t from-slate-950/60 via-transparent to-slate-950/20 pointer-events-none" />

            {/* SVG Animated Road Network Layer */}
            {(activeLayer === "both" || activeLayer === "network") && (
              <svg
                viewBox="0 0 1000 625"
                preserveAspectRatio="xMidYMid slice"
                className="absolute inset-0 w-full h-full pointer-events-none"
              >
                <defs>
                  <linearGradient id="primaryRoadGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#38BDF8" />
                    <stop offset="50%" stopColor="#0066FF" />
                    <stop offset="100%" stopColor="#60A5FA" />
                  </linearGradient>

                  <linearGradient id="criticalRoadGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#FF7A59" />
                    <stop offset="100%" stopColor="#FF5A36" />
                  </linearGradient>

                  <filter id="roadGlow" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="3" result="blur" />
                    <feComposite in="SourceGraphic" in2="blur" operator="over" />
                  </filter>
                </defs>

                {/* Background Secondary Road Arteries */}
                <g stroke="#93C5FD" strokeWidth="2.5" strokeOpacity="0.45" fill="none">
                  <path d="M 60 180 Q 220 220 380 200 T 700 240 T 940 210" className="animate-road-draw" />
                  <path d="M 120 540 Q 280 460 480 490 T 820 450 T 960 500" className="animate-road-draw-delay-1" />
                  <path d="M 380 200 Q 420 350 480 490" className="animate-road-draw-delay-2" />
                  <path d="M 700 240 Q 660 360 820 450" className="animate-road-draw" />
                  <path d="M 150 120 Q 300 280 240 430" className="animate-road-draw-delay-1" />
                </g>

                {/* Primary High-Capacity Corridors */}
                <g stroke="url(#primaryRoadGrad)" strokeWidth="4.5" strokeLinecap="round" strokeLinejoin="round" fill="none" filter="url(#roadGlow)">
                  <path
                    d="M 50 340 C 180 320, 260 270, 420 290 S 640 380, 800 320 S 910 240, 960 230"
                    className="animate-road-draw"
                  />
                  <path
                    d="M 320 60 C 340 180, 410 230, 420 290 S 450 440, 520 580"
                    className="animate-road-draw-delay-1"
                  />
                  <path
                    d="M 800 320 C 780 430, 820 520, 870 590"
                    className="animate-road-draw-delay-2"
                  />
                </g>

                {/* Critical Bottleneck Segment (Softly pulsing in Coral) */}
                <g>
                  <path
                    d="M 420 290 C 510 300, 560 325, 630 335"
                    stroke="url(#criticalRoadGrad)"
                    strokeWidth="5.5"
                    strokeLinecap="round"
                    fill="none"
                    className="animate-road-pulse"
                  />
                </g>

                {/* Topological Nodes */}
                {[
                  { cx: 50, cy: 340, id: "N-1" },
                  { cx: 220, cy: 305, id: "N-2" },
                  { cx: 380, cy: 200, id: "N-3" },
                  { cx: 420, cy: 290, id: "N-4 (Junction)" },
                  { cx: 630, cy: 335, id: "N-5 (Bottleneck)" },
                  { cx: 800, cy: 320, id: "N-6" },
                  { cx: 960, cy: 230, id: "N-7" },
                  { cx: 480, cy: 490, id: "N-8" },
                  { cx: 700, cy: 240, id: "N-9" },
                  { cx: 820, cy: 450, id: "N-10" },
                ].map((node) => (
                  <g key={node.id} transform={`translate(${node.cx}, ${node.cy})`}>
                    <circle r="4" fill="#FFFFFF" stroke="#0066FF" strokeWidth="2.5" />
                    <circle r="12" fill="none" stroke="#0066FF" strokeWidth="1" opacity="0.4" className="animate-radar" />
                  </g>
                ))}

                {/* Critical Node Highlight */}
                <g transform="translate(630, 335)">
                  <circle r="6" fill="#FF5A36" stroke="#FFFFFF" strokeWidth="2" />
                  <circle r="16" fill="none" stroke="#FF5A36" strokeWidth="1.5" className="animate-radar" />
                </g>
              </svg>
            )}

            {/* Floating Telemetry Badges */}
            <div className="absolute top-4 left-4 hidden sm:flex items-center gap-2.5 bg-white/90 backdrop-blur-md px-3.5 py-2 rounded-xl border border-slate-200/80 shadow-md">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-xs font-semibold text-slate-800">
                AI Model: SegFormer-B4
              </span>
              <span className="text-xs text-slate-400">|</span>
              <span className="text-xs font-medium text-emerald-600">IoU 93.8%</span>
            </div>

            <div className="absolute bottom-4 left-4 hidden sm:flex items-center gap-3 bg-slate-900/90 backdrop-blur-md px-4 py-2.5 rounded-xl border border-slate-700/80 text-white shadow-lg">
              <div className="flex items-center gap-1.5 text-xs">
                <Network size={14} className="text-[#0066FF]" />
                <span className="text-slate-300">Graph:</span>
                <span className="font-semibold text-white">248 Nodes / 312 Edges</span>
              </div>
              <span className="text-slate-600">|</span>
              <div className="flex items-center gap-1.5 text-xs">
                <span className="w-2 h-2 rounded-full bg-[#FF5A36] animate-ping" />
                <span className="text-slate-300">Critical Edge:</span>
                <span className="font-semibold text-[#FF5A36]">e(104, 109)</span>
              </div>
            </div>

            <div className="absolute bottom-4 right-4 hidden md:flex items-center gap-2 bg-white/95 backdrop-blur-md px-3.5 py-2 rounded-xl border border-slate-200 shadow-md">
              <Sparkles size={14} className="text-[#FF5A36]" />
              <span className="text-xs font-semibold text-slate-800">
                Resilience Index:
              </span>
              <span className="text-xs font-bold text-[#0066FF]">88.4 / 100</span>
            </div>
          </div>
        </div>

        {/* Floating Sub-indicator below visual */}
        <div className="mt-8 flex flex-wrap items-center justify-center gap-6 sm:gap-10 text-slate-500 text-xs font-medium">
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-slate-400" />
            <span>High-Resolution Remote Sensing</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-[#0066FF]" />
            <span>Automated Topology Healing</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-[#FF5A36]" />
            <span>Percolation Disruption Stress-Test</span>
          </div>
        </div>
      </div>
    </section>
  );
}
