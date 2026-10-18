import { ArrowRight, CheckCircle2, ChevronLeft, ChevronRight, Play, X } from "lucide-react";
import { Link } from "@tanstack/react-router";
import { useState } from "react";
import terrainAsset from "@/assets/route-resilience-terrain.jpg";

interface DemoModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const DEMO_SLIDES = [
  {
    title: "1. Satellite Image Acquisition",
    subtitle: "High-Resolution Remote Sensing",
    description:
      "Route Resilience processes optical & radar earth observation tiles, handling diverse ground textures, cloud shadows, and rural topographies.",
    image: terrainAsset,
    tag: "Input Acquisition",
  },
  {
    title: "2. Deep Learning Segmentation",
    subtitle: "Pixel Probability Road Masking",
    description:
      "Deep neural networks identify continuous road surfaces, unpaved corridors, and obscured arterial links with 93.8% IoU precision.",
    image: "/scene_10013_mask.png",
    tag: "Neural Segmentation",
  },
  {
    title: "3. Topological Graph Extraction",
    subtitle: "Thinning, Junctions & Gap Healing",
    description:
      "Skeletonization algorithms prune road masks into mathematical graphs with 248 nodes and 312 edges, healing occlusions automatically.",
    image: "/scene_10013_skeleton.png",
    tag: "Topology Synthesis",
  },
  {
    title: "4. Percolation Disruption Stress-Test",
    subtitle: "Betweenness Centrality & Fracture Analysis",
    description:
      "Edge percolation reveals that severing bottleneck e(104, 109) causes a 41.9% drop in regional network efficiency.",
    image: "/scene_10013_visualization.png",
    tag: "Vulnerability Modeling",
  },
];

export function DemoModal({ isOpen, onClose }: DemoModalProps) {
  const [slideIndex, setSlideIndex] = useState(0);

  if (!isOpen) return null;

  const current = DEMO_SLIDES[slideIndex];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-slate-950/60 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="relative w-full max-w-4xl bg-white rounded-3xl shadow-2xl border border-slate-200 overflow-hidden flex flex-col">
        {/* Modal Top Bar */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100 bg-slate-50/70">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-[#0066FF]" />
            <span className="font-display text-xs font-bold text-[#0A192F] uppercase tracking-wider">
              Interactive Platform Walkthrough
            </span>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-full text-slate-400 hover:text-slate-700 hover:bg-slate-200/60 transition-colors cursor-pointer"
            aria-label="Close walkthrough"
          >
            <X size={18} />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 sm:p-8 grid lg:grid-cols-12 gap-8 items-center">
          {/* Visual Panel */}
          <div className="lg:col-span-7">
            <div className="relative aspect-[16/10] rounded-2xl overflow-hidden bg-slate-900 border border-slate-200 shadow-md">
              <img
                src={current.image}
                alt={current.title}
                className="w-full h-full object-cover transition-all duration-300"
              />
              <div className="absolute bottom-3 left-3 bg-white/90 backdrop-blur-md px-3 py-1 rounded-lg text-slate-800 text-xs font-semibold">
                {current.tag}
              </div>
            </div>
          </div>

          {/* Text & Steps Panel */}
          <div className="lg:col-span-5 space-y-4">
            <div className="inline-flex items-center gap-1.5 text-xs font-bold text-[#0066FF] uppercase tracking-wider">
              <span>Step {slideIndex + 1} of {DEMO_SLIDES.length}</span>
            </div>

            <h3 className="font-display text-xl sm:text-2xl font-bold text-[#0A192F] leading-snug">
              {current.title}
            </h3>

            <p className="text-slate-600 text-sm leading-relaxed">
              {current.description}
            </p>

            {/* Stepper Dots */}
            <div className="flex items-center gap-2 pt-2">
              {DEMO_SLIDES.map((_, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => setSlideIndex(idx)}
                  className={`h-2 rounded-full transition-all cursor-pointer ${
                    idx === slideIndex ? "w-8 bg-[#0066FF]" : "w-2 bg-slate-200 hover:bg-slate-300"
                  }`}
                  aria-label={`Go to slide ${idx + 1}`}
                />
              ))}
            </div>

            {/* Action Buttons */}
            <div className="flex items-center justify-between pt-4 border-t border-slate-100">
              <button
                type="button"
                disabled={slideIndex === 0}
                onClick={() => setSlideIndex((i) => Math.max(0, i - 1))}
                className="inline-flex items-center gap-1 px-3 py-1.5 rounded-full border border-slate-200 text-xs font-semibold text-slate-600 hover:bg-slate-50 disabled:opacity-30 disabled:pointer-events-none cursor-pointer"
              >
                <ChevronLeft size={14} /> Back
              </button>

              {slideIndex < DEMO_SLIDES.length - 1 ? (
                <button
                  type="button"
                  onClick={() => setSlideIndex((i) => i + 1)}
                  className="inline-flex items-center gap-1.5 px-4 py-2 rounded-full bg-[#0066FF] text-white text-xs font-semibold hover:bg-blue-600 transition-colors shadow-xs cursor-pointer"
                >
                  Next Step <ChevronRight size={14} />
                </button>
              ) : (
                <Link
                  to="/explore"
                  onClick={onClose}
                  className="inline-flex items-center gap-1.5 px-5 py-2 rounded-full bg-[#FF5A36] text-white text-xs font-semibold hover:bg-[#EA4A26] transition-colors shadow-xs"
                >
                  Launch Analysis <ArrowRight size={14} />
                </Link>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
