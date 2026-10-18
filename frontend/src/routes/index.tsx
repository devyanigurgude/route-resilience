import { createFileRoute, Link } from "@tanstack/react-router";
import { ArrowRight } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { LandingNavbar } from "@/components/landing/LandingNavbar";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      {
        title: "Route Resilience",
      },
      {
        name: "description",
        content:
          "Autonomous road network intelligence and topology resilience platform.",
      },
    ],
    links: [
      { rel: "preconnect", href: "https://fonts.googleapis.com" },
      {
        rel: "preconnect",
        href: "https://fonts.gstatic.com",
        crossOrigin: "anonymous",
      },
      {
        rel: "stylesheet",
        href: "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700;800&family=Manrope:wght@400;500;600;700&display=swap",
      },
    ],
  }),
  component: LandingPage,
});

function LandingPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

  useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    setPrefersReducedMotion(mediaQuery.matches);

    const handleMotionChange = (e: MediaQueryListEvent) => {
      setPrefersReducedMotion(e.matches);
      if (videoRef.current) {
        if (e.matches) {
          videoRef.current.pause();
        } else {
          videoRef.current.play().catch(() => {});
        }
      }
    };

    if (mediaQuery.matches && videoRef.current) {
      videoRef.current.pause();
    }

    mediaQuery.addEventListener("change", handleMotionChange);
    return () => mediaQuery.removeEventListener("change", handleMotionChange);
  }, []);

  return (
    <div className="landing-container">
      {/* Existing navigation overlay */}
      <LandingNavbar />

      {/* Full-screen, autoplaying, muted, seamlessly looping Earth-and-satellite video */}
      <div className="landing-video-wrapper" aria-hidden="true">
        <video
          ref={videoRef}
          className="landing-bg-video"
          src="/DL_Landing_Page.mp4"
          autoPlay={!prefersReducedMotion}
          muted
          loop
          playsInline
          preload="auto"
          poster="/earth.png"
        />
      </div>

      {/* Subtle dark-blue overlay for readability */}
      <div className="landing-blue-overlay" aria-hidden="true" />

      {/* Center hero overlay content */}
      <main className="landing-hero-overlay">
        <div className="flex flex-col items-center max-w-4xl mx-auto">
          {/* Subtle live indicator badge */}
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-slate-950/50 border border-sky-400/35 backdrop-blur-md mb-6 shadow-md shadow-black/40">
            <span className="w-2 h-2 rounded-full bg-sky-400 animate-pulse" />
            <span className="text-[0.66rem] sm:text-[0.7rem] font-semibold tracking-widest text-sky-200 uppercase font-display drop-shadow-[0_1px_2px_rgba(0,0,0,0.8)]">
              Satellite Network Topology Engine
            </span>
          </div>

          {/* Centered ROUTE RESILIENCE Title in crisp white typography */}
          <h1 className="landing-hero-title" id="page-title">
            ROUTE RESILIENCE
          </h1>

          {/* Short Tagline */}
          <p className="landing-hero-tagline">
            Understand the network. Anticipate critical failure before disasters sever vital communities.
          </p>

          {/* Clean Call-to-Action button */}
          <div className="landing-hero-cta">
            <Link
              to="/explore"
              className="landing-cta-btn"
              aria-label="Enter the Route Resilience application"
            >
              <span>ENTER SYSTEM</span>
              <ArrowRight size={16} strokeWidth={2.4} aria-hidden="true" />
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
}
