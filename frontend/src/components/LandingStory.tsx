import { Link } from "@tanstack/react-router";
import { ArrowRight } from "lucide-react";
import { type CSSProperties, useEffect, useRef, useState } from "react";

/* ───────────────────────── scene data ───────────────────────── */

type Road = { d: string; x: number; w: number; react?: boolean; detail?: boolean };

// x / w = horizontal extent, used so each road draws as the scan beam passes over it.
const ROADS: Road[] = [
  { d: "M-120 745 C20 735 150 715 260 700", x: -120, w: 380 },
  { d: "M260 700 C330 660 400 640 470 620", x: 260, w: 210 },
  { d: "M260 700 C320 760 400 790 470 800", x: 260, w: 210 },
  { d: "M470 620 C450 690 455 750 470 800", x: 450, w: 30 },
  { d: "M470 620 C550 610 620 660 700 660", x: 470, w: 230, react: true },
  { d: "M700 660 C705 700 690 750 700 800", x: 690, w: 20 },
  { d: "M470 800 C540 830 630 815 700 800", x: 470, w: 230 },
  { d: "M700 800 C780 780 860 810 930 790", x: 700, w: 230 },
  { d: "M930 790 C945 720 925 660 930 600", x: 925, w: 10 },
  { d: "M930 600 C1030 600 1100 650 1180 680", x: 930, w: 250, react: true },
  { d: "M930 790 C1030 780 1120 740 1180 680", x: 930, w: 250 },
  { d: "M1180 680 C1300 690 1420 720 1580 745", x: 1180, w: 400 },
  { d: "M930 600 C950 580 975 565 1010 555", x: 930, w: 80, detail: true },
  { d: "M470 620 C490 600 520 590 560 585", x: 470, w: 90, detail: true },
  { d: "M260 700 C250 650 235 620 215 595", x: 215, w: 45, detail: true },
  { d: "M470 800 C460 840 455 870 450 910", x: 450, w: 20, detail: true },
  { d: "M700 800 C705 840 710 880 712 910", x: 700, w: 12, detail: true },
  { d: "M930 790 C935 830 945 870 955 910", x: 930, w: 25, detail: true },
];

// The critical road (C → D) and the alternate route (C → G → H → D).
// The alternate route reuses the exact curves of three existing roads.
const CRIT = { d: "M700 660 C780 650 850 640 930 600", x: 700, w: 230 };
const ALT = "M700 660 C705 700 690 750 700 800 C780 780 860 810 930 790 C945 720 925 660 930 600";

//                A          B          C          D          E           F          G          H
const NODES: [number, number][] = [
  [260, 700], [470, 620], [700, 660], [930, 600], [1180, 680], [470, 800], [700, 800], [930, 790],
];
const [C, D, G, H] = [2, 3, 6, 7];

/* ───────────────────────── scroll → state ───────────────────────── */

const clamp = (v: number, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const smooth = (t: number) => {
  const x = clamp(t);
  return x * x * (3 - 2 * x);
};
const sweep = (t: number) => 0.5 - 0.5 * Math.cos(Math.PI * clamp(t));
const bump = (a: number, from: number, len = 0.22) =>
  a <= from ? 0 : Math.sin(Math.PI * clamp((a - from) / len));
const f = (n: number) => n.toFixed(4);

function compute(p: number, mobile: boolean) {
  const v: Record<string, string> = {};

  // ── Stage 1: satellite scans, network draws itself (p 0.03 → 0.30)
  const s = sweep((p - 0.03) / 0.27);
  const x0 = mobile ? 100 : -100;
  const satX = x0 + (1720 - x0) * s;
  v["--sx"] = f(satX);
  v["--sy"] = f(372 - 26 * Math.sin(Math.PI * s) - p * 30);
  v["--bo"] = f(Math.min(1, Math.sin(Math.PI * clamp((p - 0.03) / 0.27)) * 2) * 0.55);

  ROADS.forEach((r, i) => {
    v[`--r${i}`] = f(smooth((satX - r.x + 40) / (r.w + 160)));
  });
  const rc = smooth((satX - CRIT.x + 40) / (CRIT.w + 160));
  v["--rc"] = f(rc);
  NODES.forEach(([x], i) => {
    v[`--n${i}`] = f(smooth((satX - x + 20) / 260));
  });

  // ── Stage 2: highlight → pulse → break (p 0.33 → 0.62)
  const warn = smooth((p - 0.33) / 0.1);
  const coral = warn * (1 - smooth((p - 0.56) / 0.07));
  const pulse = Math.sin(Math.PI * clamp((p - 0.43) / 0.1));
  const ring = clamp((p - 0.44) / 0.08);
  const brk = smooth((p - 0.5) / 0.1);

  // ── Stage 3: alternate route → reconnect → failed segment settles back
  const alt = sweep((p - 0.64) / 0.22);
  const heal = smooth((p - 0.86) / 0.09);
  const gap = brk * (1 - heal);
  const gapLen = 46 * gap;
  const seg = 50 - gapLen / 2;

  v["--co"] = f(coral);
  v["--pulse"] = f(pulse);
  v["--ring"] = f(ring);
  v["--ringo"] = f(ring > 0 && ring < 1 ? (1 - ring) * smooth(ring * 8) * 0.85 : 0);
  v["--d3"] = rc >= 0.999 ? `${f(seg)} ${f(gapLen)} ${f(seg)} 200` : "100 100";
  v["--o3"] = f(1 - 0.55 * gap - 0.12 * heal);
  v["--react"] = f(brk * (1 - alt));

  v["--alt"] = f(alt);
  v["--hC"] = f(bump(alt, 0));
  v["--hG"] = f(bump(alt, 0.24));
  v["--hH"] = f(bump(alt, 0.66));
  v["--hD"] = f(Math.sin(Math.PI * clamp((p - 0.855) / 0.07)));
  v["--vC"] = f(smooth(alt / 0.05));
  v["--vG"] = f(smooth((alt - 0.24) / 0.05));
  v["--vH"] = f(smooth((alt - 0.66) / 0.05));
  v["--vD"] = f(smooth((alt - 0.95) / 0.05));

  // ── Parallax (clouds, Earth, ridge)
  v["--cx1"] = f(-p * 40);
  v["--cx2"] = f(-p * 90);
  v["--cx3"] = f(-p * 160);
  v["--ey"] = f(-p * 18);
  v["--hy"] = f(-p * 8);

  // ── Copy
  const o1 = smooth((p - 0.25) / 0.06);
  const i2 = smooth((p - 0.32) / 0.06);
  const o2 = smooth((p - 0.57) / 0.06);
  const i3 = smooth((p - 0.63) / 0.06);
  v["--t1o"] = f(1 - o1);
  v["--t1y"] = f(-o1 * 16);
  v["--t2o"] = f(i2 * (1 - o2));
  v["--t2y"] = f((1 - i2) * 18 - o2 * 16);
  v["--t3o"] = f(i3);
  v["--t3y"] = f((1 - i3) * 18);
  const cta = smooth((p - 0.92) / 0.06);
  v["--cta"] = f(cta);

  return { v, stage: p < 0.31 ? "1" : p < 0.62 ? "2" : "3", cta: cta > 0.4 ? "on" : "off" };
}

/* ───────────────────────── small pieces ───────────────────────── */

const kv = (value: string) => ({ ["--k" as string]: value }) as CSSProperties;
const hv = (value: string) => ({ ["--h" as string]: value }) as CSSProperties;
const vv = (value: string) => ({ ["--v" as string]: value }) as CSSProperties;

function Cloud({ x, y, s = 1, o = 1 }: { x: number; y: number; s?: number; o?: number }) {
  return (
    <g transform={`translate(${x} ${y}) scale(${s})`} opacity={o}>
      <ellipse cx="0" cy="0" rx="92" ry="26" fill="url(#rs-cloud)" />
      <ellipse cx="-52" cy="-10" rx="54" ry="24" fill="url(#rs-cloud)" />
      <ellipse cx="46" cy="-15" rx="62" ry="28" fill="url(#rs-cloud)" />
    </g>
  );
}

function Satellite() {
  return (
    <g transform="rotate(-8) scale(.95)">
      <rect x="-46" y="-6" width="30" height="14" rx="2" fill="#2f78a8" stroke="#f1fbff" strokeWidth="1.6" />
      <rect x="16" y="-6" width="30" height="14" rx="2" fill="#2f78a8" stroke="#f1fbff" strokeWidth="1.6" />
      <path d="M-36 -6V8M-26 -6V8M26 -6V8M36 -6V8" stroke="#9bd6ec" strokeWidth="1.2" />
      <rect x="-16" y="-9" width="32" height="20" rx="4" fill="#f6f9fa" stroke="#cddde3" strokeWidth="1.6" />
      <rect x="-9" y="-3" width="18" height="8" rx="1.5" fill="#e2603f" />
      <circle cy="17" r="4.5" fill="#243b4a" stroke="#f1fbff" strokeWidth="1.5" />
    </g>
  );
}

/* ───────────────────────── component ───────────────────────── */

export function LandingStory() {
  const rootRef = useRef<HTMLElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const [mobile, setMobile] = useState(false);

  useEffect(() => {
    const root = rootRef.current;
    const stage = stageRef.current;
    if (!root || !stage) return;

    const mq = window.matchMedia("(max-width: 640px)");
    const rm = window.matchMedia("(prefers-reduced-motion: reduce)");
    let target = 0;
    let current = 0;
    let raf = 0;
    let last = 0;

    const apply = () => {
      const { v, stage: st, cta } = compute(current, mq.matches);
      for (const k in v) root.style.setProperty(k, v[k] ?? "");
      root.dataset["stage"] = st;
      root.dataset["cta"] = cta;
    };
    const measure = () => {
      const rect = root.getBoundingClientRect();
      const total = rect.height - stage.offsetHeight;
      target = total > 0 ? clamp(-rect.top / total) : 0;
    };
    const tick = (now: number) => {
      const dt = Math.min(0.064, (now - last) / 1000);
      last = now;
      current += (target - current) * (1 - Math.exp(-dt * 4.5));
      if (Math.abs(target - current) < 0.0005) {
        current = target;
        raf = 0;
      } else {
        raf = requestAnimationFrame(tick);
      }
      apply();
    };
    const onScroll = () => {
      measure();
      if (rm.matches) {
        current = target;
        apply();
        return;
      }
      if (!raf) {
        last = performance.now();
        raf = requestAnimationFrame(tick);
      }
    };
    const onResize = () => {
      measure();
      current = target;
      apply();
    };
    const onMq = () => {
      setMobile(mq.matches);
      apply();
    };

    setMobile(mq.matches);
    measure();
    current = target;
    apply();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onResize);
    mq.addEventListener("change", onMq);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onResize);
      mq.removeEventListener("change", onMq);
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);

  return (
    <section ref={rootRef} className="rs" aria-label="How Route Resilience works" data-stage="1">
      <div className="rs-mist" aria-hidden="true" />

      <div ref={stageRef} className="rs-stage">
        <div className="rs-scene" aria-hidden="true">
          <svg
            className="rs-svg"
            viewBox={mobile ? "225 230 990 670" : "0 0 1440 900"}
            preserveAspectRatio={mobile ? "xMidYMax meet" : "xMidYMax slice"}
          >
            <defs>
              <radialGradient id="rs-cloud">
                <stop offset="0" stopColor="#fff" stopOpacity=".95" />
                <stop offset="1" stopColor="#fff" stopOpacity="0" />
              </radialGradient>
              <linearGradient id="rs-land" gradientUnits="userSpaceOnUse" x1="0" y1="520" x2="0" y2="1000">
                <stop offset="0" stopColor="#d3efc8" />
                <stop offset=".45" stopColor="#a2d89f" />
                <stop offset="1" stopColor="#6dba8d" />
              </linearGradient>
              <linearGradient id="rs-haze" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0" stopColor="#fff" stopOpacity=".75" />
                <stop offset="1" stopColor="#fff" stopOpacity="0" />
              </linearGradient>
              <linearGradient id="rs-shade" gradientUnits="userSpaceOnUse" x1="0" y1="520" x2="0" y2="1000">
                <stop offset="0" stopColor="#fff" stopOpacity=".3" />
                <stop offset=".25" stopColor="#fff" stopOpacity="0" />
                <stop offset="1" stopColor="#0b4a86" stopOpacity=".2" />
              </linearGradient>
              <linearGradient id="rs-beam" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0" stopColor="#fff" stopOpacity=".7" />
                <stop offset="1" stopColor="#fff" stopOpacity=".1" />
              </linearGradient>
              <clipPath id="rs-clip">
                <circle cx="720" cy="3120" r="2600" />
              </clipPath>
            </defs>

            {/* far + mid clouds */}
            <g className="rs-layer rs-cx1">
              <g className="rs-drift rs-drift--a">
                <Cloud x={200} y={300} s={0.7} o={0.5} />
                <Cloud x={760} y={250} s={0.6} o={0.45} />
                <Cloud x={1250} y={330} s={0.8} o={0.5} />
                <Cloud x={1650} y={280} s={0.7} o={0.45} />
              </g>
            </g>
            <g className="rs-layer rs-cx2">
              <g className="rs-drift rs-drift--b">
                <Cloud x={420} y={430} s={1} o={0.7} />
                <Cloud x={1000} y={400} s={1.1} o={0.65} />
                <Cloud x={1500} y={450} s={1} o={0.7} />
              </g>
            </g>

            {/* Earth + terrain + network */}
            <g className="rs-layer rs-earth">
              <circle cx="720" cy="3120" r="2600" fill="none" stroke="#fff" strokeOpacity=".14" strokeWidth="44" />
              <circle cx="720" cy="3120" r="2600" fill="none" stroke="#e6f7ff" strokeOpacity=".3" strokeWidth="16" />
              <circle cx="720" cy="3120" r="2600" fill="none" stroke="#fff" strokeOpacity=".95" strokeWidth="3" />

              <g clipPath="url(#rs-clip)">
                <circle cx="720" cy="3120" r="2600" fill="url(#rs-land)" />
                <g className="rs-hills">
                  <path
                    d="M-300 620 C0 560 200 585 420 575 S800 545 1000 570 S1400 560 1800 600 V700 H-300Z"
                    fill="#a6dcc2"
                    opacity=".5"
                  />
                </g>
                <rect x="-300" y="508" width="2040" height="140" fill="url(#rs-haze)" />

                <g className="rs-detail">
                  <ellipse cx="380" cy="740" rx="120" ry="26" fill="#fff" opacity=".14" />
                  <ellipse cx="580" cy="722" rx="90" ry="18" fill="#3d9a78" opacity=".1" />
                  <ellipse cx="820" cy="722" rx="110" ry="22" fill="#fff" opacity=".12" />
                  <ellipse cx="1060" cy="742" rx="140" ry="26" fill="#3d9a78" opacity=".1" />
                  <ellipse cx="330" cy="860" rx="160" ry="30" fill="#3d9a78" opacity=".1" />
                  <ellipse cx="820" cy="862" rx="200" ry="34" fill="#fff" opacity=".12" />
                  <ellipse cx="1100" cy="852" rx="150" ry="28" fill="#fff" opacity=".1" />
                  <ellipse cx="160" cy="780" rx="90" ry="20" fill="#fff" opacity=".12" />
                  <ellipse cx="1320" cy="792" rx="100" ry="20" fill="#3d9a78" opacity=".1" />
                  <ellipse cx="1090" cy="640" rx="70" ry="12" fill="#8fd3ee" opacity=".85" />
                  <path
                    d="M610 548 C640 590 590 640 620 690 C650 740 560 770 590 830 C600 870 590 900 585 930"
                    fill="none"
                    stroke="#8fd3ee"
                    strokeWidth="12"
                    strokeLinecap="round"
                    opacity=".7"
                  />
                </g>

                <circle cx="720" cy="3120" r="2600" fill="url(#rs-shade)" />

                {/* ── road network ── */}
                <g className="rs-net">
                  {ROADS.map((r, i) => (
                    <path
                      key={r.d}
                      d={r.d}
                      pathLength={100}
                      className={`rs-road${r.react ? " rs-road--react" : ""}${r.detail ? " rs-detail" : ""}`}
                      style={kv(`var(--r${i},0)`)}
                    />
                  ))}

                  {/* critical road: navy base → coral highlight → break */}
                  <path d={CRIT.d} pathLength={100} className="rs-crit rs-crit--base" />
                  <path d={CRIT.d} pathLength={100} className="rs-crit rs-crit--glow" />
                  <path d={CRIT.d} pathLength={100} className="rs-crit rs-crit--core" />
                  <circle cx="815" cy="641" r="12" className="rs-ring" />

                  {/* alternate route */}
                  <path d={ALT} pathLength={100} className="rs-alt rs-alt--case" />
                  <path d={ALT} pathLength={100} className="rs-alt rs-alt--line" />

                  {/* soft halos while the route is discovered / network reacts */}
                  <circle cx={NODES[C]![0]} cy={NODES[C]![1]} r="11" className="rs-halo" style={hv("var(--hC,0)")} />
                  <circle cx={NODES[G]![0]} cy={NODES[G]![1]} r="11" className="rs-halo" style={hv("var(--hG,0)")} />
                  <circle cx={NODES[H]![0]} cy={NODES[H]![1]} r="11" className="rs-halo" style={hv("var(--hH,0)")} />
                  <circle cx={NODES[D]![0]} cy={NODES[D]![1]} r="11" className="rs-halo" style={hv("var(--hD,0)")} />
                  <circle cx={NODES[C]![0]} cy={NODES[C]![1]} r="11" className="rs-halo rs-halo--coral" />
                  <circle cx={NODES[D]![0]} cy={NODES[D]![1]} r="11" className="rs-halo rs-halo--coral" />

                  {NODES.map(([x, y], i) => (
                    <circle key={i} cx={x} cy={y} r="7" className="rs-node" style={kv(`var(--n${i},0)`)} />
                  ))}
                  <circle cx={NODES[C]![0]} cy={NODES[C]![1]} r="12" className="rs-vring" style={vv("var(--vC,0)")} />
                  <circle cx={NODES[G]![0]} cy={NODES[G]![1]} r="12" className="rs-vring" style={vv("var(--vG,0)")} />
                  <circle cx={NODES[H]![0]} cy={NODES[H]![1]} r="12" className="rs-vring" style={vv("var(--vH,0)")} />
                  <circle cx={NODES[D]![0]} cy={NODES[D]![1]} r="12" className="rs-vring" style={vv("var(--vD,0)")} />
                </g>
              </g>
            </g>

            {/* satellite + scan beam */}
            <g className="rs-sat">
              <g className="rs-bob">
                <polygon points="0,16 -150,340 150,340" fill="url(#rs-beam)" className="rs-beam" />
                <ellipse cx="0" cy="340" rx="150" ry="24" className="rs-footprint" />
                <Satellite />
              </g>
            </g>

            {/* near clouds, in front of the horizon */}
            <g className="rs-layer rs-cx3">
              <g className="rs-drift rs-drift--c">
                <Cloud x={120} y={492} s={1.6} o={0.55} />
                <Cloud x={900} y={505} s={1.8} o={0.5} />
                <Cloud x={1400} y={495} s={1.5} o={0.55} />
              </g>
            </g>
          </svg>
        </div>

        <div className="rs-copy">
          <div className="rs-panel rs-panel--1">
            <h2>See the network</h2>
            <p>Turn satellite imagery into a connected road network.</p>
          </div>
          <div className="rs-panel rs-panel--2">
            <h2>What if it fails?</h2>
            <p>Test the network when a critical road disappears.</p>
          </div>
          <div className="rs-panel rs-panel--3">
            <h2>Find a way through.</h2>
            <p>Discover an alternate route and measure recovery.</p>
          </div>
          <div className="rs-cta">
            <Link to="/explore" className="rs-btn">
              <span>Analyze your network</span>
              <ArrowRight size={16} strokeWidth={2.4} aria-hidden="true" />
            </Link>
          </div>
        </div>

        <div className="rs-dots" aria-hidden="true">
          <i />
          <i />
          <i />
        </div>
      </div>
    </section>
  );
}