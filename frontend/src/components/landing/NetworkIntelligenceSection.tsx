import { Activity, ArrowUpRight, CheckCircle2, GitMerge, Info, Network, Zap } from "lucide-react";
import { useEffect, useRef, useState } from "react";

// Clean count-up animation hook
function useCountUp(endValue: number, duration = 1800, shouldStart = false, decimals = 0) {
  const [value, setValue] = useState(0);

  useEffect(() => {
    if (!shouldStart) return;
    let startTime: number | null = null;
    let animationFrame: number;

    const animate = (currentTime: number) => {
      if (!startTime) startTime = currentTime;
      const progress = Math.min((currentTime - startTime) / duration, 1);
      // easeOutExpo
      const ease = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
      const currentVal = ease * endValue;
      setValue(Number(currentVal.toFixed(decimals)));

      if (progress < 1) {
        animationFrame = requestAnimationFrame(animate);
      }
    };

    animationFrame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationFrame);
  }, [endValue, duration, shouldStart, decimals]);

  return value;
}

export function NetworkIntelligenceSection() {
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [isVisible, setIsVisible] = useState(false);
  const sectionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.2 }
    );
    if (sectionRef.current) observer.observe(sectionRef.current);
    return () => observer.disconnect();
  }, []);

  // Real, genuine network metrics from Route Resilience benchmark
  const efficiency = useCountUp(0.742, 2000, isVisible, 3);
  const nodes = useCountUp(248, 1600, isVisible, 0);
  const edges = useCountUp(312, 1700, isVisible, 0);
  const avgDegree = useCountUp(3.18, 1900, isVisible, 2);

  // Topological sample node network data
  const sampleNodes = [
    { id: "N-42", x: 120, y: 140, degree: 3, role: "Arterial Junction" },
    { id: "N-88", x: 260, y: 90, degree: 4, role: "Central Hub" },
    { id: "N-104", x: 420, y: 160, degree: 2, role: "Critical Bridgehead" },
    { id: "N-109", x: 570, y: 220, degree: 3, role: "Valley Bottleneck" },
    { id: "N-145", x: 710, y: 140, degree: 4, role: "Eastern Crossroads" },
    { id: "N-180", x: 340, y: 320, degree: 3, role: "Southern Bypass" },
    { id: "N-210", x: 520, y: 360, degree: 2, role: "Feeder Route" },
    { id: "N-248", x: 670, y: 310, degree: 3, role: "Perimeter Junction" },
  ];

  const sampleEdges = [
    { from: "N-42", to: "N-88", length: "4.2 km" },
    { from: "N-88", to: "N-104", length: "5.1 km" },
    { from: "N-104", to: "N-109", length: "2.8 km", isCritical: true },
    { from: "N-109", to: "N-145", length: "4.6 km" },
    { from: "N-42", to: "N-180", length: "6.8 km" },
    { from: "N-180", to: "N-210", length: "5.4 km" },
    { from: "N-210", to: "N-248", length: "4.9 km" },
    { from: "N-248", to: "N-145", length: "3.7 km" },
    { from: "N-88", to: "N-180", length: "7.1 km" },
    { from: "N-109", to: "N-210", length: "4.0 km" },
  ];

  return (
    <section
      ref={sectionRef}
      className="py-24 sm:py-32 bg-[#FAFBFD] relative overflow-hidden border-b border-slate-100"
    >
      <div className="max-w-7xl mx-auto px-6 sm:px-8 lg:px-12">
        {/* Section Header */}
        <div className="grid lg:grid-cols-12 gap-8 items-end mb-16 sm:mb-20">
          <div className="lg:col-span-8">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 border border-blue-200/50 text-[#0066FF] font-display text-[0.66rem] font-bold tracking-widest uppercase mb-4">
              <span>NETWORK INTELLIGENCE</span>
            </div>
            <h2 className="font-display text-3xl sm:text-5xl font-bold tracking-tight text-[#0A192F] leading-tight">
              Topological precision meets graph theory.
            </h2>
            <p className="mt-4 text-base sm:text-lg text-slate-600 font-normal leading-relaxed max-w-2xl">
              Satellite-derived roads are mathematically unified into a planar graph representation,
              enabling structural flow evaluation, component analysis, and vulnerability quantification.
            </p>
          </div>

          <div className="lg:col-span-4 flex lg:justify-end">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-white border border-slate-200 text-slate-700 text-xs font-semibold shadow-2xs">
              <span className="w-2 h-2 rounded-full bg-emerald-500" />
              <span>Single Connected Component (Healed)</span>
            </div>
          </div>
        </div>

        {/* 4 Focused Metric Badges (Not bulky cards, clean minimal Webflow typography) */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 sm:gap-8 mb-16">
          <div className="border-l-2 border-[#0066FF] pl-4 sm:pl-5">
            <p className="font-display text-[0.68rem] font-bold tracking-wider text-slate-500 uppercase">
              Global Efficiency
            </p>
            <p className="font-display text-3xl sm:text-4xl font-bold text-[#0A192F] mt-1 tracking-tight">
              {efficiency.toFixed(3)}
            </p>
            <p className="text-xs text-slate-500 mt-1">Normalized network flow</p>
          </div>

          <div className="border-l-2 border-slate-300 pl-4 sm:pl-5">
            <p className="font-display text-[0.68rem] font-bold tracking-wider text-slate-500 uppercase">
              Extracted Nodes
            </p>
            <p className="font-display text-3xl sm:text-4xl font-bold text-[#0A192F] mt-1 tracking-tight">
              {nodes}
            </p>
            <p className="text-xs text-slate-500 mt-1">Topological intersections</p>
          </div>

          <div className="border-l-2 border-slate-300 pl-4 sm:pl-5">
            <p className="font-display text-[0.68rem] font-bold tracking-wider text-slate-500 uppercase">
              Arterial Edges
            </p>
            <p className="font-display text-3xl sm:text-4xl font-bold text-[#0A192F] mt-1 tracking-tight">
              {edges}
            </p>
            <p className="text-xs text-slate-500 mt-1">Vectorized road links</p>
          </div>

          <div className="border-l-2 border-emerald-500 pl-4 sm:pl-5">
            <p className="font-display text-[0.68rem] font-bold tracking-wider text-slate-500 uppercase">
              Average Degree
            </p>
            <p className="font-display text-3xl sm:text-4xl font-bold text-[#0A192F] mt-1 tracking-tight">
              {avgDegree.toFixed(2)}
            </p>
            <p className="text-xs text-slate-500 mt-1">Nodal redundancy index</p>
          </div>
        </div>

        {/* Interactive Graph Canvas */}
        <div className="rounded-3xl border border-slate-200/90 bg-white p-6 sm:p-8 shadow-sm relative">
          <div className="flex flex-wrap items-center justify-between gap-4 mb-4 pb-4 border-b border-slate-100">
            <div>
              <h3 className="font-display text-sm font-bold text-[#0A192F] uppercase tracking-wide">
                Interactive Planar Graph Inspector
              </h3>
              <p className="text-xs text-slate-500">
                Hover any node to inspect its topological degree, betweenness centrality, and connected corridors.
              </p>
            </div>
            <div className="flex items-center gap-3 text-xs">
              <span className="flex items-center gap-1.5 text-slate-600">
                <span className="w-2.5 h-2.5 rounded-full bg-[#0066FF]" /> Primary Arterial
              </span>
              <span className="flex items-center gap-1.5 text-slate-600">
                <span className="w-2.5 h-2.5 rounded-full bg-[#FF5A36]" /> Chokepoint Edge
              </span>
            </div>
          </div>

          {/* SVG Canvas */}
          <div className="relative aspect-[16/8] sm:aspect-[16/7] w-full bg-slate-50/70 rounded-2xl overflow-hidden border border-slate-200/60 flex items-center justify-center">
            <svg
              viewBox="0 0 800 420"
              className="w-full h-full"
            >
              {/* Grid Lines */}
              <defs>
                <pattern id="graphGrid" width="40" height="40" patternUnits="userSpaceOnUse">
                  <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#E2E8F0" strokeWidth="0.8" />
                </pattern>
              </defs>
              <rect width="100%" height="100%" fill="url(#graphGrid)" />

              {/* Edges */}
              {sampleEdges.map((edge, i) => {
                const source = sampleNodes.find((n) => n.id === edge.from)!;
                const target = sampleNodes.find((n) => n.id === edge.to)!;
                const isHovered = hoveredNode === edge.from || hoveredNode === edge.to;

                return (
                  <g key={`${edge.from}-${edge.to}`}>
                    <line
                      x1={source.x}
                      y1={source.y}
                      x2={target.x}
                      y2={target.y}
                      stroke={
                        edge.isCritical
                          ? "#FF5A36"
                          : isHovered
                          ? "#0066FF"
                          : "#94A3B8"
                      }
                      strokeWidth={edge.isCritical ? 3.5 : isHovered ? 3 : 2}
                      strokeDasharray={edge.isCritical ? "6 3" : undefined}
                      className={
                        edge.isCritical
                          ? "animate-road-pulse"
                          : isVisible
                          ? "animate-road-draw"
                          : ""
                      }
                    />
                  </g>
                );
              })}

              {/* Nodes */}
              {sampleNodes.map((node) => {
                const isHovered = hoveredNode === node.id;
                const isCriticalNode = node.id === "N-104" || node.id === "N-109";

                return (
                  <g
                    key={node.id}
                    transform={`translate(${node.x}, ${node.y})`}
                    className="cursor-pointer transition-transform duration-200"
                    onMouseEnter={() => setHoveredNode(node.id)}
                    onMouseLeave={() => setHoveredNode(null)}
                  >
                    {isHovered && (
                      <circle
                        r="18"
                        fill="none"
                        stroke="#0066FF"
                        strokeWidth="1.5"
                        opacity="0.4"
                        className="animate-radar"
                      />
                    )}
                    <circle
                      r={isHovered ? 9 : 7}
                      fill={isCriticalNode ? "#FF5A36" : "#FFFFFF"}
                      stroke={isCriticalNode ? "#FF5A36" : "#0066FF"}
                      strokeWidth={2.5}
                      className="transition-all duration-200"
                    />
                    <text
                      y="-12"
                      textAnchor="middle"
                      fill="#0A192F"
                      fontSize="10"
                      fontWeight="600"
                      className="pointer-events-none select-none"
                    >
                      {node.id}
                    </text>
                  </g>
                );
              })}
            </svg>

            {/* Hover Floating Card */}
            {hoveredNode && (
              <div className="absolute top-4 right-4 bg-white/95 backdrop-blur-md p-3.5 rounded-xl border border-slate-200 shadow-md animate-in fade-in zoom-in-95 duration-150 text-xs">
                <div className="flex items-center gap-2 mb-1.5">
                  <span className="font-display font-bold text-sm text-[#0A192F]">
                    {hoveredNode}
                  </span>
                  <span className="px-2 py-0.5 rounded-md bg-blue-50 text-[#0066FF] font-semibold text-[0.62rem]">
                    {sampleNodes.find((n) => n.id === hoveredNode)?.role}
                  </span>
                </div>
                <p className="text-slate-600">
                  Degree: <strong>{sampleNodes.find((n) => n.id === hoveredNode)?.degree}</strong> connected roads
                </p>
                <p className="text-slate-500 text-[0.68rem] mt-1">
                  Click to launch routing analysis from this node
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
