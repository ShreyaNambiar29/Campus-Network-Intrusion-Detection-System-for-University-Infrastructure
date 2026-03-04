import { useState, useEffect, useRef, useCallback } from "react";
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, RadialBarChart, RadialBar, Legend
} from "recharts";

/* ═══════════════════════════════════════════════════════════════
   CAMPUS NETWORK IDS — CYBERSECURITY DASHBOARD
   Aesthetic: Tactical Dark / Military-Grade Cyber Console
   Fonts: Rajdhani (display) + Share Tech Mono (data)
═══════════════════════════════════════════════════════════════ */

const STYLE = `
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg-void:    #040608;
    --bg-deep:    #070b0f;
    --bg-panel:   #0a0f15;
    --bg-card:    #0d1520;
    --bg-hover:   #111d2e;
    --border:     #1a2d42;
    --border-hi:  #1e3a52;
    --cyan:       #00d4ff;
    --cyan-dim:   #007a94;
    --cyan-glow:  rgba(0,212,255,0.15);
    --green:      #00ff88;
    --green-dim:  #00994d;
    --amber:      #ffaa00;
    --red:        #ff3355;
    --red-dim:    #991f33;
    --purple:     #aa44ff;
    --text-hi:    #e8f4ff;
    --text-mid:   #7a9bb5;
    --text-dim:   #3d5970;
    --scan-line:  rgba(0,212,255,0.03);
  }

  html, body { height: 100%; background: var(--bg-void); color: var(--text-hi);
    font-family: 'Exo 2', sans-serif; overflow-x: hidden; }

  /* Scan-line overlay */
  body::before {
    content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 9999;
    background: repeating-linear-gradient(
      0deg, transparent, transparent 2px, var(--scan-line) 2px, var(--scan-line) 4px
    );
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: var(--bg-deep); }
  ::-webkit-scrollbar-thumb { background: var(--cyan-dim); border-radius: 2px; }

  .ids-root { display: flex; height: 100vh; overflow: hidden; }

  /* ── Sidebar ── */
  .sidebar {
    width: 72px; background: var(--bg-deep);
    border-right: 1px solid var(--border);
    display: flex; flex-direction: column; align-items: center;
    padding: 20px 0; gap: 8px; flex-shrink: 0; z-index: 100;
    position: relative;
  }
  .sidebar::after {
    content: ''; position: absolute; right: 0; top: 0; bottom: 0; width: 1px;
    background: linear-gradient(180deg, transparent, var(--cyan), transparent);
    opacity: 0.4;
  }
  .sidebar-logo {
    width: 44px; height: 44px; background: var(--cyan-glow);
    border: 1px solid var(--cyan-dim); border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 16px; font-size: 20px;
    box-shadow: 0 0 20px var(--cyan-glow), inset 0 0 10px rgba(0,212,255,0.05);
  }
  .nav-item {
    width: 44px; height: 44px; border-radius: 10px; border: 1px solid transparent;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer; transition: all 0.2s; font-size: 18px; position: relative;
    color: var(--text-dim);
  }
  .nav-item:hover { border-color: var(--border-hi); background: var(--bg-hover);
    color: var(--cyan); }
  .nav-item.active {
    border-color: var(--cyan-dim); background: var(--cyan-glow);
    color: var(--cyan); box-shadow: 0 0 12px var(--cyan-glow);
  }
  .nav-badge {
    position: absolute; top: 4px; right: 4px; width: 8px; height: 8px;
    background: var(--red); border-radius: 50%; border: 1px solid var(--bg-deep);
    animation: pulse-red 2s infinite;
  }
  @keyframes pulse-red {
    0%,100% { box-shadow: 0 0 0 0 rgba(255,51,85,0.7); }
    50% { box-shadow: 0 0 0 4px rgba(255,51,85,0); }
  }

  /* ── Main area ── */
  .main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

  /* ── Topbar ── */
  .topbar {
    height: 56px; background: var(--bg-deep); border-bottom: 1px solid var(--border);
    display: flex; align-items: center; padding: 0 24px; gap: 20px; flex-shrink: 0;
    position: relative;
  }
  .topbar::before {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, var(--cyan-dim), transparent);
    opacity: 0.5;
  }
  .topbar-title {
    font-family: 'Rajdhani', sans-serif; font-weight: 700; font-size: 18px;
    letter-spacing: 3px; text-transform: uppercase; color: var(--cyan);
    text-shadow: 0 0 20px var(--cyan-dim);
  }
  .topbar-sub { font-size: 11px; color: var(--text-dim); letter-spacing: 2px;
    text-transform: uppercase; font-family: 'Share Tech Mono', monospace; }
  .topbar-spacer { flex: 1; }
  .topbar-stat {
    display: flex; flex-direction: column; align-items: flex-end;
    padding: 0 16px; border-left: 1px solid var(--border);
  }
  .topbar-stat-val { font-family: 'Share Tech Mono', monospace; font-size: 14px;
    color: var(--text-hi); }
  .topbar-stat-lbl { font-size: 9px; color: var(--text-dim); text-transform: uppercase;
    letter-spacing: 1.5px; }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green);
    box-shadow: 0 0 8px var(--green); animation: pulse-green 3s infinite; }
  @keyframes pulse-green {
    0%,100% { opacity: 1; } 50% { opacity: 0.4; }
  }

  /* ── Content ── */
  .content {
    flex: 1; overflow-y: auto; padding: 20px 24px;
    background: var(--bg-void);
    background-image:
      radial-gradient(ellipse at 20% 20%, rgba(0,212,255,0.03) 0%, transparent 60%),
      radial-gradient(ellipse at 80% 80%, rgba(0,255,136,0.02) 0%, transparent 60%);
  }

  /* ── Grid layouts ── */
  .grid-5 { display: grid; grid-template-columns: repeat(5,1fr); gap: 12px; margin-bottom: 20px; }
  .grid-3 { display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 16px; margin-bottom: 20px; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
  .grid-half { display: grid; grid-template-columns: 3fr 2fr; gap: 16px; margin-bottom: 20px; }

  /* ── Cards ── */
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px; padding: 16px;
    position: relative; overflow: hidden;
    transition: border-color 0.2s;
  }
  .card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-hi), transparent);
  }
  .card:hover { border-color: var(--border-hi); }

  /* Stat card */
  .stat-card { padding: 14px 16px; }
  .stat-card .s-icon { font-size: 22px; margin-bottom: 10px; }
  .stat-card .s-val {
    font-family: 'Rajdhani', sans-serif; font-size: 30px; font-weight: 700;
    line-height: 1; margin-bottom: 4px;
  }
  .stat-card .s-label { font-size: 10px; text-transform: uppercase; letter-spacing: 2px;
    color: var(--text-dim); }
  .stat-card .s-delta { font-size: 11px; margin-top: 8px; font-family: 'Share Tech Mono', monospace; }

  .stat-cyan  { border-top: 2px solid var(--cyan);  }
  .stat-green { border-top: 2px solid var(--green); }
  .stat-amber { border-top: 2px solid var(--amber); }
  .stat-red   { border-top: 2px solid var(--red);   }
  .stat-purple{ border-top: 2px solid var(--purple); }

  /* Card header */
  .card-hdr { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
  .card-hdr-icon { font-size: 14px; }
  .card-hdr-title { font-family: 'Rajdhani', sans-serif; font-weight: 600;
    letter-spacing: 2px; text-transform: uppercase; font-size: 13px; color: var(--text-mid); }
  .card-hdr-badge { margin-left: auto; font-size: 10px; padding: 2px 8px;
    border-radius: 3px; font-family: 'Share Tech Mono', monospace; }
  .badge-live { background: rgba(0,255,136,0.1); color: var(--green);
    border: 1px solid var(--green-dim); animation: blink 2s infinite; }
  @keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
  .badge-num { background: rgba(0,212,255,0.1); color: var(--cyan);
    border: 1px solid var(--cyan-dim); }

  /* Security score ring */
  .score-wrap {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: 8px 0;
  }
  .score-ring { position: relative; width: 140px; height: 140px; }
  .score-num {
    position: absolute; inset: 0; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
  }
  .score-num .big { font-family: 'Rajdhani', sans-serif; font-size: 38px;
    font-weight: 700; line-height: 1; }
  .score-num .lbl { font-size: 9px; text-transform: uppercase; letter-spacing: 2px;
    color: var(--text-dim); margin-top: 2px; }
  .score-grade {
    font-family: 'Rajdhani', sans-serif; font-size: 22px; font-weight: 700;
    margin-top: 10px; letter-spacing: 3px;
  }
  .score-status { font-size: 10px; text-transform: uppercase; letter-spacing: 2px;
    color: var(--text-dim); margin-top: 4px; }

  /* Alert feed */
  .alert-feed { display: flex; flex-direction: column; gap: 6px; max-height: 280px;
    overflow-y: auto; }
  .alert-item {
    display: flex; align-items: flex-start; gap: 10px; padding: 9px 12px;
    background: var(--bg-panel); border-radius: 5px;
    border-left: 3px solid var(--border);
    animation: slide-in 0.3s ease-out; transition: background 0.2s;
  }
  .alert-item:hover { background: var(--bg-hover); }
  @keyframes slide-in { from { opacity: 0; transform: translateX(-8px); } to { opacity: 1; } }
  .alert-item.critical { border-left-color: var(--red); }
  .alert-item.high     { border-left-color: var(--amber); }
  .alert-item.medium   { border-left-color: var(--cyan); }
  .alert-item.low      { border-left-color: var(--green); }
  .alert-sev { font-size: 9px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; padding: 2px 6px; border-radius: 2px;
    font-family: 'Share Tech Mono', monospace; white-space: nowrap; }
  .sev-critical { background: rgba(255,51,85,0.15); color: var(--red); }
  .sev-high     { background: rgba(255,170,0,0.15); color: var(--amber); }
  .sev-medium   { background: rgba(0,212,255,0.1);  color: var(--cyan); }
  .sev-low      { background: rgba(0,255,136,0.1);  color: var(--green); }
  .alert-body { flex: 1; min-width: 0; }
  .alert-type { font-size: 12px; font-weight: 600; color: var(--text-hi);
    font-family: 'Rajdhani', sans-serif; letter-spacing: 1px; }
  .alert-ip { font-size: 11px; color: var(--text-dim);
    font-family: 'Share Tech Mono', monospace; margin-top: 1px; }
  .alert-time { font-size: 10px; color: var(--text-dim);
    font-family: 'Share Tech Mono', monospace; white-space: nowrap; }

  /* IP table */
  .ip-table { width: 100%; border-collapse: collapse; }
  .ip-table th { font-size: 9px; text-transform: uppercase; letter-spacing: 2px;
    color: var(--text-dim); text-align: left; padding: 6px 8px;
    border-bottom: 1px solid var(--border); }
  .ip-table td { font-size: 12px; padding: 7px 8px;
    border-bottom: 1px solid rgba(26,45,66,0.5); }
  .ip-table tr:hover td { background: var(--bg-hover); }
  .ip-mono { font-family: 'Share Tech Mono', monospace; color: var(--cyan); }
  .threat-bar-wrap { display: flex; align-items: center; gap: 8px; }
  .threat-bar { height: 4px; border-radius: 2px; flex: 1; background: var(--bg-panel); overflow: hidden; }
  .threat-fill { height: 100%; border-radius: 2px; transition: width 0.6s ease; }

  /* Incident table */
  .incident-table { width: 100%; border-collapse: collapse; }
  .incident-table th { font-size: 9px; text-transform: uppercase; letter-spacing: 2px;
    color: var(--text-dim); padding: 8px 12px; border-bottom: 1px solid var(--border);
    text-align: left; }
  .incident-table td { font-size: 12px; padding: 9px 12px;
    border-bottom: 1px solid rgba(26,45,66,0.4); }
  .incident-table tr { transition: background 0.15s; }
  .incident-table tr:hover td { background: var(--bg-hover); }
  .status-open     { color: var(--red);   font-size: 10px; text-transform: uppercase;
    letter-spacing: 1px; font-family: 'Share Tech Mono', monospace; }
  .status-resolved { color: var(--green); font-size: 10px; text-transform: uppercase;
    letter-spacing: 1px; font-family: 'Share Tech Mono', monospace; }
  .resolve-btn {
    font-size: 10px; padding: 3px 10px; border-radius: 3px;
    border: 1px solid var(--green-dim); background: rgba(0,255,136,0.05);
    color: var(--green); cursor: pointer; transition: all 0.15s;
    font-family: 'Share Tech Mono', monospace;
  }
  .resolve-btn:hover { background: rgba(0,255,136,0.15); }

  /* Tooltip */
  .recharts-tooltip-wrapper { outline: none; }
  .custom-tt { background: var(--bg-card); border: 1px solid var(--border-hi);
    border-radius: 5px; padding: 8px 12px; font-size: 11px;
    font-family: 'Share Tech Mono', monospace; color: var(--text-hi); }

  /* Section label */
  .section-lbl { font-family: 'Rajdhani', sans-serif; font-weight: 600;
    font-size: 11px; text-transform: uppercase; letter-spacing: 3px;
    color: var(--text-dim); margin-bottom: 10px; display: flex; align-items: center; gap: 8px; }
  .section-lbl::after { content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--border), transparent); }

  /* Tab bar */
  .tab-bar { display: flex; gap: 4px; margin-bottom: 20px; }
  .tab-btn {
    padding: 7px 16px; border-radius: 5px; border: 1px solid var(--border);
    background: transparent; color: var(--text-dim); cursor: pointer;
    font-family: 'Rajdhani', sans-serif; font-size: 13px; font-weight: 600;
    letter-spacing: 1px; text-transform: uppercase; transition: all 0.2s;
  }
  .tab-btn:hover { border-color: var(--border-hi); color: var(--text-hi); }
  .tab-btn.active { border-color: var(--cyan-dim); background: var(--cyan-glow);
    color: var(--cyan); }

  /* Ticker */
  .ticker-wrap { background: var(--bg-deep); border-top: 1px solid var(--border);
    height: 30px; overflow: hidden; display: flex; align-items: center;
    flex-shrink: 0; padding: 0 16px; gap: 8px; }
  .ticker-lbl { font-size: 9px; text-transform: uppercase; letter-spacing: 2px;
    color: var(--cyan); font-family: 'Share Tech Mono', monospace; white-space: nowrap; }
  .ticker-scroll { flex: 1; overflow: hidden; position: relative; }
  .ticker-inner { display: flex; gap: 48px; animation: scroll-left 30s linear infinite; white-space: nowrap; }
  .ticker-item { font-size: 11px; font-family: 'Share Tech Mono', monospace;
    color: var(--text-mid); }
  @keyframes scroll-left { from { transform: translateX(0); } to { transform: translateX(-50%); } }

  /* Heatmap */
  .heatmap-grid { display: grid; grid-template-columns: repeat(24, 1fr); gap: 3px; }
  .heatmap-cell { height: 12px; border-radius: 2px; transition: transform 0.1s; }
  .heatmap-cell:hover { transform: scaleY(1.5); }
  .heatmap-labels { display: flex; justify-content: space-between; margin-top: 4px; }
  .heatmap-lbl { font-size: 9px; color: var(--text-dim);
    font-family: 'Share Tech Mono', monospace; }

  /* Protocol pills */
  .proto-pills { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
  .proto-pill { display: flex; flex-direction: column; align-items: center;
    gap: 4px; flex: 1; min-width: 60px; }
  .proto-pill-bar { width: 100%; height: 50px; background: var(--bg-panel);
    border-radius: 4px; overflow: hidden; display: flex; align-items: flex-end; }
  .proto-pill-fill { width: 100%; border-radius: 4px 4px 0 0; transition: height 0.6s ease; }
  .proto-pill-lbl { font-size: 9px; text-transform: uppercase; letter-spacing: 1px;
    color: var(--text-dim); font-family: 'Share Tech Mono', monospace; }
  .proto-pill-val { font-family: 'Share Tech Mono', monospace; font-size: 11px; }

  /* Animated threat indicator */
  @keyframes threat-pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(255,51,85,0.5); }
    50%      { box-shadow: 0 0 0 8px rgba(255,51,85,0); }
  }
  .threat-pulse { animation: threat-pulse 1.5s infinite; }
`;

// ═══════════════════════════════════════════════════════
// DATA SIMULATION
// ═══════════════════════════════════════════════════════

const ATTACK_TYPES = ["Port Scan", "SYN Flood", "Brute Force", "Malware C2", "Abnormal Behavior", "Suspicious Traffic"];
const SEVERITIES   = ["critical", "high", "medium", "low"];
const CAMPUS_IPS   = Array.from({length: 20}, (_,i) => `10.10.${Math.floor(i/5)+1}.${(i%5)+10}`);
const EXTERNAL_IPS = Array.from({length: 10}, (_,i) => `198.51.${100+i}.${Math.floor(Math.random()*200+1)}`);
const ALL_IPS      = [...CAMPUS_IPS, ...EXTERNAL_IPS];

const randOf   = (arr) => arr[Math.floor(Math.random() * arr.length)];
const randInt  = (a,b) => Math.floor(Math.random()*(b-a))+a;
const fmtTime  = (d)   => d.toTimeString().slice(0,8);
const fmtDate  = (d)   => d.toLocaleDateString("en-GB",{day:"2-digit",month:"short"});

function generateAlert() {
  const sev = randOf(SEVERITIES);
  const src  = randOf(ALL_IPS);
  return {
    id:          Date.now() + Math.random(),
    timestamp:   new Date(),
    src_ip:      src,
    dst_ip:      randOf(CAMPUS_IPS),
    dst_port:    randOf([22,80,443,3306,8080,4444,21]),
    attack_type: randOf(ATTACK_TYPES),
    severity:    sev,
    description: `Detected suspicious activity from ${src}`,
    threat_score:randInt(20,100),
    status:      "open",
  };
}

function generateTrafficPoint(base = 0) {
  const t = new Date();
  return {
    time:     fmtTime(t),
    packets:  base + randInt(50, 400),
    bytes_kb: base/10 + randInt(10, 200),
    anomalies:randInt(0, 8),
  };
}

function generateIncidents(n = 12) {
  return Array.from({length: n}, (_, i) => {
    const d = new Date(Date.now() - i * 4 * 60000);
    return {
      id:          i + 1,
      timestamp:   d,
      src_ip:      randOf(ALL_IPS),
      dst_ip:      randOf(CAMPUS_IPS),
      attack_type: randOf(ATTACK_TYPES),
      severity:    SEVERITIES[Math.min(i % 4, 3)],
      status:      i % 3 === 0 ? "resolved" : "open",
      threat_score:randInt(25, 98),
      description: "Automated detection rule triggered",
    };
  });
}

function generateHeatmap() {
  return Array.from({length: 24}, (_, h) => ({
    hour:  h,
    value: h >= 8 && h <= 18 ? randInt(20, 100) : randInt(0, 25),
  }));
}

function generateTopIPs() {
  return Array.from({length: 8}, (_, i) => ({
    ip:      randOf(ALL_IPS),
    score:   Math.max(10, 95 - i * 11 + randInt(-5, 5)),
    count:   randInt(1, 45),
    last:    fmtTime(new Date(Date.now() - randInt(0, 3600000))),
  })).sort((a,b) => b.score - a.score);
}

const COLORS = {
  "Port Scan":          "#00d4ff",
  "SYN Flood":          "#ff3355",
  "Brute Force":        "#ffaa00",
  "Malware C2":         "#aa44ff",
  "Abnormal Behavior":  "#00ff88",
  "Suspicious Traffic": "#ff7755",
};

const PIE_DATA = ATTACK_TYPES.map(t => ({ name: t, value: randInt(5, 35) }));

// ═══════════════════════════════════════════════════════
// CUSTOM TOOLTIP
// ═══════════════════════════════════════════════════════
const CTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="custom-tt">
      <div style={{color:"#7a9bb5",marginBottom:4}}>{label}</div>
      {payload.map((p,i) => (
        <div key={i} style={{color:p.color||"#e8f4ff"}}>
          {p.name}: <strong>{typeof p.value === "number" ? p.value.toLocaleString() : p.value}</strong>
        </div>
      ))}
    </div>
  );
};

// ═══════════════════════════════════════════════════════
// MAIN DASHBOARD COMPONENT
// ═══════════════════════════════════════════════════════
export default function CampusIDS() {
  const [activeTab,    setActiveTab]    = useState("overview");
  const [alerts,       setAlerts]       = useState(() => Array.from({length:8}, generateAlert));
  const [trafficData,  setTrafficData]  = useState(() => Array.from({length:20}, (_,i) => generateTrafficPoint(i*5)));
  const [incidents,    setIncidents]    = useState(generateIncidents);
  const [heatmap,      setHeatmap]      = useState(generateHeatmap);
  const [topIPs,       setTopIPs]       = useState(generateTopIPs);
  const [stats,        setStats]        = useState({
    totalPackets: 182_441, totalBytes: 847.3,
    activeThreats: 7, securityScore: 74,
    tcpPct: 61, udpPct: 28, icmpPct: 11,
    packetsPerSec: 142,
  });
  const [newAlert,     setNewAlert]     = useState(null);
  const [tickerItems,  setTickerItems]  = useState([]);
  const [elapsed,      setElapsed]      = useState(0);
  const alertFeedRef = useRef(null);

  // ── Simulate live data ──────────────────────────────
  useEffect(() => {
    const iv = setInterval(() => {
      setElapsed(e => e + 1);
      setStats(s => ({
        ...s,
        totalPackets:   s.totalPackets + randInt(80, 250),
        totalBytes:     +(s.totalBytes + randInt(0,50)/100).toFixed(1),
        packetsPerSec:  randInt(95, 210),
        activeThreats:  Math.max(0, s.activeThreats + (Math.random() > 0.6 ? 1 : -1)),
        securityScore:  Math.max(20, Math.min(99, s.securityScore + (Math.random() > 0.5 ? 1 : -1))),
      }));

      setTrafficData(prev => {
        const next = [...prev.slice(-29), generateTrafficPoint()];
        return next;
      });

      // Occasionally fire an alert
      if (Math.random() < 0.25) {
        const a = generateAlert();
        setNewAlert(a);
        setAlerts(prev => [a, ...prev].slice(0, 50));
        setTickerItems(prev => [`[${fmtTime(new Date())}] ${a.attack_type} from ${a.src_ip}`, ...prev].slice(0, 20));
        setTimeout(() => setNewAlert(null), 3000);
      }
    }, 2000);
    return () => clearInterval(iv);
  }, []);

  // ── Auto-scroll alert feed ──────────────────────────
  useEffect(() => {
    if (alertFeedRef.current) {
      alertFeedRef.current.scrollTop = 0;
    }
  }, [alerts.length]);

  const resolveIncident = useCallback((id) => {
    setIncidents(prev => prev.map(i => i.id === id ? {...i, status:"resolved"} : i));
  }, []);

  const scoreColor = stats.securityScore >= 80 ? "#00ff88"
                   : stats.securityScore >= 60 ? "#ffaa00" : "#ff3355";
  const scoreGrade = stats.securityScore >= 90 ? "A" : stats.securityScore >= 75 ? "B"
                   : stats.securityScore >= 60 ? "C" : stats.securityScore >= 40 ? "D" : "F";

  const tickerStr = tickerItems.join("   ///   ");
  const doubledTicker = tickerStr + "   ///   " + tickerStr;

  const attDist = ATTACK_TYPES.map((t,i) => ({
    name: t, value: randInt(5, 35),
    color: Object.values(COLORS)[i],
  }));

  return (
    <>
      <style>{STYLE}</style>
      <div className="ids-root">
        {/* ── Sidebar ── */}
        <aside className="sidebar">
          <div className="sidebar-logo">🛡️</div>
          {[
            { id:"overview",   icon:"📊", label:"Overview" },
            { id:"incidents",  icon:"🚨", label:"Incidents", badge:true },
            { id:"traffic",    icon:"📡", label:"Traffic" },
            { id:"analytics",  icon:"📈", label:"Analytics" },
            { id:"ips",        icon:"🌐", label:"IPs" },
            { id:"reports",    icon:"📋", label:"Reports" },
          ].map(n => (
            <div key={n.id} className={`nav-item ${activeTab===n.id?"active":""}`}
                 title={n.label} onClick={() => setActiveTab(n.id)}>
              {n.icon}
              {n.badge && alerts.filter(a=>a.status==="open").length > 0 && <span className="nav-badge"/>}
            </div>
          ))}
          <div style={{flex:1}}/>
          <div className="nav-item" title="Settings">⚙️</div>
          <div style={{marginTop:4}}>
            <div style={{width:36,height:36,borderRadius:"50%",
              background:"linear-gradient(135deg,#1a2d42,#0d1520)",
              border:"1px solid var(--border-hi)", display:"flex",
              alignItems:"center",justifyContent:"center",fontSize:14,cursor:"pointer"}}>
              👤
            </div>
          </div>
        </aside>

        {/* ── Main ── */}
        <div className="main">
          {/* ── Topbar ── */}
          <header className="topbar">
            <div>
              <div className="topbar-title">Campus IDS</div>
              <div className="topbar-sub">University Network Security Operations Center</div>
            </div>
            <div className="topbar-spacer"/>
            {newAlert && (
              <div style={{
                background:"rgba(255,51,85,0.1)",border:"1px solid var(--red)",
                borderRadius:5,padding:"5px 14px",display:"flex",alignItems:"center",
                gap:8,animation:"slide-in 0.3s ease-out",
              }}>
                <span style={{color:"var(--red)",fontSize:12}}>🚨</span>
                <span style={{fontSize:11,color:"var(--red)",fontFamily:"'Share Tech Mono',monospace"}}>
                  {newAlert.attack_type} — {newAlert.src_ip}
                </span>
              </div>
            )}
            <div className="topbar-stat">
              <span className="topbar-stat-val">{stats.packetsPerSec} <span style={{fontSize:10,color:"var(--text-dim)"}}>pkt/s</span></span>
              <span className="topbar-stat-lbl">Capture Rate</span>
            </div>
            <div className="topbar-stat">
              <span className="topbar-stat-val" style={{color:scoreColor}}>{stats.securityScore}</span>
              <span className="topbar-stat-lbl">Security Score</span>
            </div>
            <div className="topbar-stat">
              <span className="topbar-stat-val" style={{fontFamily:"'Share Tech Mono',monospace",fontSize:12}}>
                {new Date().toTimeString().slice(0,8)}
              </span>
              <span className="topbar-stat-lbl">System Time</span>
            </div>
            <div style={{display:"flex",alignItems:"center",gap:6,paddingLeft:16,borderLeft:"1px solid var(--border)"}}>
              <div className="status-dot"/>
              <span style={{fontSize:10,color:"var(--green)",letterSpacing:1,textTransform:"uppercase",
                fontFamily:"'Share Tech Mono',monospace"}}>LIVE</span>
            </div>
          </header>

          {/* ── Content ── */}
          <div className="content">
            {/* ── Tab bar ── */}
            <div className="tab-bar">
              {["Overview","Incidents","Traffic","Analytics","IPs","Reports"].map(t => (
                <button key={t} className={`tab-btn ${activeTab===t.toLowerCase()?"active":""}`}
                        onClick={() => setActiveTab(t.toLowerCase())}>{t}</button>
              ))}
            </div>

            {/* ═══════════════ OVERVIEW ═══════════════ */}
            {activeTab === "overview" && (
              <>
                <div className="section-lbl">Network Status</div>

                {/* KPI Row */}
                <div className="grid-5">
                  {[
                    { icon:"📦", val:stats.totalPackets.toLocaleString(), lbl:"Packets Captured",
                      delta:"▲ +2.3K last min", color:"cyan" },
                    { icon:"⚠️", val:stats.activeThreats, lbl:"Active Threats",
                      delta:`${alerts.filter(a=>a.severity==="critical").length} critical`, color:"red" },
                    { icon:"💾", val:`${stats.totalBytes} GB`, lbl:"Traffic Volume",
                      delta:"▲ +48 MB/min", color:"green" },
                    { icon:"🎯", val:incidents.filter(i=>i.status==="open").length, lbl:"Open Incidents",
                      delta:`${incidents.filter(i=>i.status==="resolved").length} resolved`, color:"amber" },
                    { icon:"🌐", val:topIPs.length, lbl:"Suspicious IPs",
                      delta:`Top: ${topIPs[0]?.ip||"—"}`, color:"purple" },
                  ].map((s,i) => (
                    <div key={i} className={`card stat-card stat-${s.color}`}>
                      <div className="s-icon">{s.icon}</div>
                      <div className="s-val" style={{color:`var(--${s.color})`}}>{s.val}</div>
                      <div className="s-label">{s.lbl}</div>
                      <div className="s-delta" style={{color:"var(--text-dim)"}}>{s.delta}</div>
                    </div>
                  ))}
                </div>

                {/* Main charts row */}
                <div className="grid-half">
                  {/* Live traffic chart */}
                  <div className="card">
                    <div className="card-hdr">
                      <span className="card-hdr-icon">📡</span>
                      <span className="card-hdr-title">Live Network Traffic</span>
                      <span className="card-hdr-badge badge-live">● LIVE</span>
                    </div>
                    <ResponsiveContainer width="100%" height={200}>
                      <AreaChart data={trafficData} margin={{top:5,right:5,bottom:0,left:-20}}>
                        <defs>
                          <linearGradient id="gPkt" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#00d4ff" stopOpacity={0.3}/>
                            <stop offset="95%" stopColor="#00d4ff" stopOpacity={0}/>
                          </linearGradient>
                          <linearGradient id="gAno" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#ff3355" stopOpacity={0.3}/>
                            <stop offset="95%" stopColor="#ff3355" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(26,45,66,0.8)" />
                        <XAxis dataKey="time" tick={{fill:"#3d5970",fontSize:9}} interval={4}/>
                        <YAxis tick={{fill:"#3d5970",fontSize:9}}/>
                        <Tooltip content={<CTooltip/>}/>
                        <Area type="monotone" dataKey="packets" stroke="#00d4ff" fill="url(#gPkt)" strokeWidth={2} name="Packets"/>
                        <Area type="monotone" dataKey="anomalies" stroke="#ff3355" fill="url(#gAno)" strokeWidth={1.5} name="Anomalies"/>
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Security score + attack dist */}
                  <div style={{display:"flex",flexDirection:"column",gap:16}}>
                    <div className="card">
                      <div className="card-hdr">
                        <span className="card-hdr-icon">🛡️</span>
                        <span className="card-hdr-title">Security Score</span>
                      </div>
                      <div className="score-wrap">
                        <div className="score-ring">
                          <RadialBarChart width={140} height={140}
                            cx={70} cy={70} innerRadius={50} outerRadius={65}
                            data={[{value: stats.securityScore, fill: scoreColor}]}
                            startAngle={220} endAngle={-40}>
                            <RadialBar background={{fill:"rgba(26,45,66,0.4)"}}
                              dataKey="value" cornerRadius={6}/>
                          </RadialBarChart>
                          <div className="score-num">
                            <span className="big" style={{color:scoreColor}}>{stats.securityScore}</span>
                            <span className="lbl">/ 100</span>
                          </div>
                        </div>
                        <div className="score-grade" style={{color:scoreColor}}>{scoreGrade} Grade</div>
                        <div className="score-status">
                          {stats.securityScore >= 80 ? "SECURE" : stats.securityScore >= 60 ? "MODERATE RISK" : "HIGH RISK"}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Second row: attack dist + alerts */}
                <div className="grid-2">
                  {/* Attack distribution pie */}
                  <div className="card">
                    <div className="card-hdr">
                      <span className="card-hdr-icon">🎯</span>
                      <span className="card-hdr-title">Attack Distribution</span>
                      <span className="card-hdr-badge badge-num">7 DAYS</span>
                    </div>
                    <div style={{display:"flex",alignItems:"center",gap:16}}>
                      <ResponsiveContainer width={160} height={160}>
                        <PieChart>
                          <Pie data={PIE_DATA} cx={75} cy={75} innerRadius={45} outerRadius={70}
                               dataKey="value" strokeWidth={0}>
                            {PIE_DATA.map((_, i) => (
                              <Cell key={i} fill={Object.values(COLORS)[i % Object.values(COLORS).length]}/>
                            ))}
                          </Pie>
                          <Tooltip content={<CTooltip/>}/>
                        </PieChart>
                      </ResponsiveContainer>
                      <div style={{flex:1,display:"flex",flexDirection:"column",gap:5}}>
                        {PIE_DATA.map((d,i) => (
                          <div key={i} style={{display:"flex",alignItems:"center",gap:6}}>
                            <div style={{width:8,height:8,borderRadius:2,flexShrink:0,
                              background:Object.values(COLORS)[i%Object.values(COLORS).length]}}/>
                            <span style={{fontSize:10,color:"var(--text-mid)",flex:1,
                              fontFamily:"'Exo 2',sans-serif"}}>{d.name}</span>
                            <span style={{fontSize:10,color:"var(--text-hi)",
                              fontFamily:"'Share Tech Mono',monospace"}}>{d.value}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Alert feed */}
                  <div className="card">
                    <div className="card-hdr">
                      <span className="card-hdr-icon">🚨</span>
                      <span className="card-hdr-title">Live Alert Feed</span>
                      <span className="card-hdr-badge badge-num">{alerts.length}</span>
                    </div>
                    <div className="alert-feed" ref={alertFeedRef}>
                      {alerts.slice(0,15).map(a => (
                        <div key={a.id} className={`alert-item ${a.severity}`}>
                          <span className={`alert-sev sev-${a.severity}`}>{a.severity.toUpperCase()}</span>
                          <div className="alert-body">
                            <div className="alert-type">{a.attack_type}</div>
                            <div className="alert-ip">{a.src_ip} → {a.dst_ip}:{a.dst_port}</div>
                          </div>
                          <span className="alert-time">{fmtTime(a.timestamp)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Heatmap */}
                <div className="card" style={{marginBottom:20}}>
                  <div className="card-hdr">
                    <span className="card-hdr-icon">🔥</span>
                    <span className="card-hdr-title">Attack Activity Heatmap (24h)</span>
                  </div>
                  <div className="heatmap-grid">
                    {heatmap.map((h,i) => {
                      const intensity = h.value / 100;
                      const r = Math.floor(255 * Math.min(intensity * 2, 1));
                      const g = Math.floor(255 * Math.max(1 - intensity * 2, 0) * 0.8);
                      const b = Math.floor(50 * (1 - intensity));
                      return (
                        <div key={i} className="heatmap-cell" title={`${h.hour}:00 — ${h.value} events`}
                             style={{background:`rgba(${r},${g},${b},${0.2 + intensity*0.8})`}}/>
                      );
                    })}
                  </div>
                  <div className="heatmap-labels">
                    {[0,3,6,9,12,15,18,21,23].map(h => (
                      <span key={h} className="heatmap-lbl">{String(h).padStart(2,"0")}:00</span>
                    ))}
                  </div>
                </div>

                {/* Top suspicious IPs */}
                <div className="card">
                  <div className="card-hdr">
                    <span className="card-hdr-icon">🌐</span>
                    <span className="card-hdr-title">Top Suspicious IPs</span>
                    <span className="card-hdr-badge badge-num">REAL-TIME</span>
                  </div>
                  <table className="ip-table">
                    <thead>
                      <tr>
                        <th>#</th><th>IP Address</th><th>Threat Score</th>
                        <th>Incidents</th><th>Last Seen</th><th>Risk</th>
                      </tr>
                    </thead>
                    <tbody>
                      {topIPs.map((ip,i) => {
                        const c = ip.score >= 80 ? "var(--red)" : ip.score >= 60 ? "var(--amber)" : "var(--cyan)";
                        return (
                          <tr key={i}>
                            <td style={{color:"var(--text-dim)",fontFamily:"'Share Tech Mono',monospace"}}>{i+1}</td>
                            <td className="ip-mono">{ip.ip}</td>
                            <td>
                              <div className="threat-bar-wrap">
                                <div className="threat-bar">
                                  <div className="threat-fill" style={{width:`${ip.score}%`,background:c}}/>
                                </div>
                                <span style={{fontSize:11,color:c,fontFamily:"'Share Tech Mono',monospace",minWidth:28}}>
                                  {ip.score}
                                </span>
                              </div>
                            </td>
                            <td style={{color:"var(--text-hi)",fontFamily:"'Share Tech Mono',monospace"}}>{ip.count}</td>
                            <td style={{color:"var(--text-dim)",fontFamily:"'Share Tech Mono',monospace",fontSize:11}}>{ip.last}</td>
                            <td>
                              <span style={{fontSize:10,padding:"2px 7px",borderRadius:3,
                                background:ip.score>=80?"rgba(255,51,85,0.1)":ip.score>=60?"rgba(255,170,0,0.1)":"rgba(0,212,255,0.1)",
                                color:c,fontFamily:"'Share Tech Mono',monospace",border:`1px solid ${c}33`}}>
                                {ip.score>=80?"CRITICAL":ip.score>=60?"HIGH":"MEDIUM"}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </>
            )}

            {/* ═══════════════ INCIDENTS ═══════════════ */}
            {activeTab === "incidents" && (
              <>
                <div className="section-lbl">Incident Management</div>
                <div className="grid-5" style={{marginBottom:20}}>
                  {[
                    {lbl:"Total",  val:incidents.length, c:"cyan"},
                    {lbl:"Open",   val:incidents.filter(i=>i.status==="open").length, c:"red"},
                    {lbl:"Resolved",val:incidents.filter(i=>i.status==="resolved").length, c:"green"},
                    {lbl:"Critical",val:incidents.filter(i=>i.severity==="critical").length, c:"red"},
                    {lbl:"High",    val:incidents.filter(i=>i.severity==="high").length, c:"amber"},
                  ].map((s,i) => (
                    <div key={i} className={`card stat-card stat-${s.c}`}>
                      <div className="s-val" style={{color:`var(--${s.c})`,fontSize:28}}>{s.val}</div>
                      <div className="s-label">{s.lbl} Incidents</div>
                    </div>
                  ))}
                </div>

                <div className="card">
                  <div className="card-hdr">
                    <span className="card-hdr-icon">🚨</span>
                    <span className="card-hdr-title">All Incidents</span>
                    <span className="card-hdr-badge badge-num">{incidents.length} TOTAL</span>
                  </div>
                  <table className="incident-table">
                    <thead>
                      <tr>
                        <th>ID</th><th>Time</th><th>Source IP</th><th>Attack Type</th>
                        <th>Severity</th><th>Score</th><th>Status</th><th>Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {incidents.map(inc => {
                        const sc = inc.severity;
                        const c  = sc==="critical"?"var(--red)":sc==="high"?"var(--amber)":sc==="medium"?"var(--cyan)":"var(--green)";
                        return (
                          <tr key={inc.id}>
                            <td style={{color:"var(--text-dim)",fontFamily:"'Share Tech Mono',monospace",fontSize:11}}>
                              #{String(inc.id).padStart(4,"0")}
                            </td>
                            <td style={{color:"var(--text-dim)",fontFamily:"'Share Tech Mono',monospace",fontSize:10}}>
                              {fmtTime(inc.timestamp)}<br/>
                              <span style={{fontSize:9}}>{fmtDate(inc.timestamp)}</span>
                            </td>
                            <td className="ip-mono">{inc.src_ip}</td>
                            <td style={{fontWeight:600,color:"var(--text-hi)",fontFamily:"'Rajdhani',sans-serif",fontSize:13}}>
                              {inc.attack_type}
                            </td>
                            <td>
                              <span className={`alert-sev sev-${sc}`}>{sc.toUpperCase()}</span>
                            </td>
                            <td>
                              <span style={{color:c,fontFamily:"'Share Tech Mono',monospace",fontSize:13,fontWeight:"bold"}}>
                                {inc.threat_score}
                              </span>
                            </td>
                            <td>
                              {inc.status === "open"
                                ? <span className="status-open">● OPEN</span>
                                : <span className="status-resolved">✓ RESOLVED</span>
                              }
                            </td>
                            <td>
                              {inc.status === "open" && (
                                <button className="resolve-btn" onClick={() => resolveIncident(inc.id)}>
                                  RESOLVE
                                </button>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </>
            )}

            {/* ═══════════════ TRAFFIC ═══════════════ */}
            {activeTab === "traffic" && (
              <>
                <div className="section-lbl">Network Traffic Analysis</div>
                <div className="card" style={{marginBottom:16}}>
                  <div className="card-hdr">
                    <span className="card-hdr-icon">📊</span>
                    <span className="card-hdr-title">Packet Volume — Last 60 Seconds</span>
                    <span className="card-hdr-badge badge-live">● LIVE</span>
                  </div>
                  <ResponsiveContainer width="100%" height={240}>
                    <AreaChart data={trafficData}>
                      <defs>
                        <linearGradient id="gPkt2" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#00d4ff" stopOpacity={0.25}/>
                          <stop offset="95%" stopColor="#00d4ff" stopOpacity={0}/>
                        </linearGradient>
                        <linearGradient id="gByt" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#00ff88" stopOpacity={0.2}/>
                          <stop offset="95%" stopColor="#00ff88" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(26,45,66,0.8)"/>
                      <XAxis dataKey="time" tick={{fill:"#3d5970",fontSize:9}} interval={4}/>
                      <YAxis yAxisId="left"  tick={{fill:"#3d5970",fontSize:9}}/>
                      <YAxis yAxisId="right" orientation="right" tick={{fill:"#3d5970",fontSize:9}}/>
                      <Tooltip content={<CTooltip/>}/>
                      <Legend wrapperStyle={{fontSize:11,color:"#7a9bb5"}}/>
                      <Area yAxisId="left" type="monotone" dataKey="packets" stroke="#00d4ff"
                            fill="url(#gPkt2)" strokeWidth={2} name="Packets"/>
                      <Area yAxisId="right" type="monotone" dataKey="bytes_kb" stroke="#00ff88"
                            fill="url(#gByt)" strokeWidth={1.5} name="KB/s"/>
                    </AreaChart>
                  </ResponsiveContainer>
                </div>

                <div className="grid-2">
                  <div className="card">
                    <div className="card-hdr">
                      <span className="card-hdr-icon">🔀</span>
                      <span className="card-hdr-title">Protocol Breakdown</span>
                    </div>
                    <div className="proto-pills">
                      {[
                        {proto:"TCP",  pct:stats.tcpPct,  c:"#00d4ff"},
                        {proto:"UDP",  pct:stats.udpPct,  c:"#00ff88"},
                        {proto:"ICMP", pct:stats.icmpPct, c:"#ffaa00"},
                      ].map(p => (
                        <div key={p.proto} className="proto-pill">
                          <div className="proto-pill-bar">
                            <div className="proto-pill-fill"
                                 style={{height:`${p.pct}%`,background:p.c,opacity:0.7}}/>
                          </div>
                          <div className="proto-pill-lbl">{p.proto}</div>
                          <div className="proto-pill-val" style={{color:p.c}}>{p.pct}%</div>
                        </div>
                      ))}
                    </div>
                    <ResponsiveContainer width="100%" height={160} style={{marginTop:16}}>
                      <BarChart data={[
                        {name:"TCP",value:stats.tcpPct, fill:"#00d4ff"},
                        {name:"UDP",value:stats.udpPct, fill:"#00ff88"},
                        {name:"ICMP",value:stats.icmpPct, fill:"#ffaa00"},
                      ]}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(26,45,66,0.8)"/>
                        <XAxis dataKey="name" tick={{fill:"#7a9bb5",fontSize:11}}/>
                        <YAxis tick={{fill:"#3d5970",fontSize:9}}/>
                        <Tooltip content={<CTooltip/>}/>
                        <Bar dataKey="value" name="%" radius={[3,3,0,0]}>
                          {[{fill:"#00d4ff"},{fill:"#00ff88"},{fill:"#ffaa00"}].map((c,i)=>(
                            <Cell key={i} fill={c.fill}/>
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="card">
                    <div className="card-hdr">
                      <span className="card-hdr-icon">📡</span>
                      <span className="card-hdr-title">Anomaly Count</span>
                    </div>
                    <ResponsiveContainer width="100%" height={240}>
                      <BarChart data={trafficData.slice(-15)}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(26,45,66,0.8)"/>
                        <XAxis dataKey="time" tick={{fill:"#3d5970",fontSize:9}} interval={2}/>
                        <YAxis tick={{fill:"#3d5970",fontSize:9}}/>
                        <Tooltip content={<CTooltip/>}/>
                        <Bar dataKey="anomalies" fill="#ff3355" opacity={0.8} name="Anomalies" radius={[3,3,0,0]}/>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </>
            )}

            {/* ═══════════════ ANALYTICS ═══════════════ */}
            {activeTab === "analytics" && (
              <>
                <div className="section-lbl">Security Analytics</div>
                <div className="grid-2" style={{marginBottom:16}}>
                  <div className="card">
                    <div className="card-hdr">
                      <span className="card-hdr-icon">📈</span>
                      <span className="card-hdr-title">Threat Severity Trend</span>
                    </div>
                    <ResponsiveContainer width="100%" height={220}>
                      <AreaChart data={Array.from({length:24},(_,i)=>({
                        hour:`${String(i).padStart(2,"0")}:00`,
                        critical:randInt(0,5), high:randInt(0,15),
                        medium:randInt(0,25),  low:randInt(0,30),
                      }))}>
                        <defs>
                          {["#ff3355","#ffaa00","#00d4ff","#00ff88"].map((c,i)=>(
                            <linearGradient key={i} id={`g${i}`} x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor={c} stopOpacity={0.3}/>
                              <stop offset="95%" stopColor={c} stopOpacity={0}/>
                            </linearGradient>
                          ))}
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(26,45,66,0.8)"/>
                        <XAxis dataKey="hour" tick={{fill:"#3d5970",fontSize:8}} interval={3}/>
                        <YAxis tick={{fill:"#3d5970",fontSize:9}}/>
                        <Tooltip content={<CTooltip/>}/>
                        <Legend wrapperStyle={{fontSize:10,color:"#7a9bb5"}}/>
                        <Area type="monotone" dataKey="critical" stroke="#ff3355" fill="url(#g0)" strokeWidth={2} stackId="1" name="Critical"/>
                        <Area type="monotone" dataKey="high"     stroke="#ffaa00" fill="url(#g1)" strokeWidth={2} stackId="1" name="High"/>
                        <Area type="monotone" dataKey="medium"   stroke="#00d4ff" fill="url(#g2)" strokeWidth={1.5} stackId="1" name="Medium"/>
                        <Area type="monotone" dataKey="low"      stroke="#00ff88" fill="url(#g3)" strokeWidth={1} stackId="1" name="Low"/>
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="card">
                    <div className="card-hdr">
                      <span className="card-hdr-icon">🎯</span>
                      <span className="card-hdr-title">Attack Type Distribution</span>
                    </div>
                    <ResponsiveContainer width="100%" height={220}>
                      <BarChart data={attDist} layout="vertical" margin={{left:10,right:20}}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(26,45,66,0.8)" horizontal={false}/>
                        <XAxis type="number" tick={{fill:"#3d5970",fontSize:9}}/>
                        <YAxis dataKey="name" type="category" tick={{fill:"#7a9bb5",fontSize:9}} width={120}/>
                        <Tooltip content={<CTooltip/>}/>
                        <Bar dataKey="value" name="Count" radius={[0,3,3,0]}>
                          {attDist.map((d,i)=><Cell key={i} fill={d.color}/>)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="card">
                  <div className="card-hdr">
                    <span className="card-hdr-icon">📅</span>
                    <span className="card-hdr-title">Weekly Incident Volume</span>
                  </div>
                  <ResponsiveContainer width="100%" height={180}>
                    <LineChart data={Array.from({length:7},(_,i)=>{
                      const d = new Date(Date.now()-i*86400000);
                      return {
                        day: d.toLocaleDateString("en-GB",{weekday:"short"}),
                        incidents: randInt(5,40), resolved: randInt(3,35)
                      };
                    }).reverse()}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(26,45,66,0.8)"/>
                      <XAxis dataKey="day" tick={{fill:"#7a9bb5",fontSize:11}}/>
                      <YAxis tick={{fill:"#3d5970",fontSize:9}}/>
                      <Tooltip content={<CTooltip/>}/>
                      <Legend wrapperStyle={{fontSize:11,color:"#7a9bb5"}}/>
                      <Line type="monotone" dataKey="incidents" stroke="#ff3355" strokeWidth={2} dot={{fill:"#ff3355",r:3}} name="Incidents"/>
                      <Line type="monotone" dataKey="resolved"  stroke="#00ff88" strokeWidth={2} dot={{fill:"#00ff88",r:3}} name="Resolved"/>
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </>
            )}

            {/* ═══════════════ IPs ═══════════════ */}
            {activeTab === "ips" && (
              <>
                <div className="section-lbl">IP Reputation Intelligence</div>
                <div className="card">
                  <div className="card-hdr">
                    <span className="card-hdr-icon">🌐</span>
                    <span className="card-hdr-title">IP Threat Registry</span>
                    <span className="card-hdr-badge badge-num">{topIPs.length} FLAGGED</span>
                  </div>
                  <table className="ip-table">
                    <thead>
                      <tr>
                        <th>Rank</th><th>IP Address</th><th>Threat Score</th>
                        <th>Total Incidents</th><th>Last Seen</th>
                        <th>Status</th><th>Risk Level</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...topIPs,...topIPs.slice(0,4)].map((ip,i) => {
                        const sc = ip.score >= 80 ? "CRITICAL" : ip.score >= 60 ? "HIGH" : "MEDIUM";
                        const c  = ip.score >= 80 ? "var(--red)" : ip.score >= 60 ? "var(--amber)" : "var(--cyan)";
                        return (
                          <tr key={i}>
                            <td style={{color:"var(--text-dim)",fontFamily:"'Share Tech Mono',monospace"}}>{i+1}</td>
                            <td className="ip-mono">{ip.ip}</td>
                            <td>
                              <div className="threat-bar-wrap">
                                <div className="threat-bar" style={{width:120}}>
                                  <div className="threat-fill" style={{width:`${ip.score}%`,background:c}}/>
                                </div>
                                <span style={{fontSize:12,color:c,fontFamily:"'Share Tech Mono',monospace"}}>
                                  {ip.score}
                                </span>
                              </div>
                            </td>
                            <td style={{fontFamily:"'Share Tech Mono',monospace",color:"var(--text-hi)"}}>{ip.count}</td>
                            <td style={{fontFamily:"'Share Tech Mono',monospace",color:"var(--text-dim)",fontSize:11}}>{ip.last}</td>
                            <td>
                              <span style={{fontSize:10,color:"var(--amber)",fontFamily:"'Share Tech Mono',monospace"}}>
                                {i < 3 ? "⛔ BLOCKED" : "⚠️ MONITOR"}
                              </span>
                            </td>
                            <td>
                              <span style={{fontSize:10,padding:"2px 8px",borderRadius:3,
                                background:`${c}22`,color:c,fontFamily:"'Share Tech Mono',monospace",
                                border:`1px solid ${c}44`}}>{sc}</span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </>
            )}

            {/* ═══════════════ REPORTS ═══════════════ */}
            {activeTab === "reports" && (
              <>
                <div className="section-lbl">Security Reports</div>
                <div className="grid-2">
                  <div className="card">
                    <div className="card-hdr">
                      <span className="card-hdr-icon">📋</span>
                      <span className="card-hdr-title">Weekly Summary</span>
                    </div>
                    {[
                      {lbl:"Total Incidents Detected", val:"147", c:"var(--red)"},
                      {lbl:"Critical Severity",        val:"12",  c:"var(--red)"},
                      {lbl:"High Severity",            val:"34",  c:"var(--amber)"},
                      {lbl:"Unique Attacker IPs",      val:"28",  c:"var(--cyan)"},
                      {lbl:"Packets Analysed",         val:"1.2M",c:"var(--green)"},
                      {lbl:"Avg Security Score",       val:"76",  c:"var(--green)"},
                      {lbl:"Incidents Resolved",       val:"109", c:"var(--green)"},
                      {lbl:"ML Anomalies Detected",    val:"53",  c:"var(--purple)"},
                    ].map((r,i) => (
                      <div key={i} style={{
                        display:"flex",justifyContent:"space-between",alignItems:"center",
                        padding:"9px 0",borderBottom:"1px solid var(--border)",
                      }}>
                        <span style={{fontSize:12,color:"var(--text-mid)"}}>{r.lbl}</span>
                        <span style={{fontFamily:"'Rajdhani',sans-serif",fontWeight:700,
                          fontSize:18,color:r.c}}>{r.val}</span>
                      </div>
                    ))}
                  </div>
                  <div className="card">
                    <div className="card-hdr">
                      <span className="card-hdr-icon">📊</span>
                      <span className="card-hdr-title">Top Attack Types This Week</span>
                    </div>
                    {ATTACK_TYPES.map((t,i) => {
                      const count = randInt(5,50);
                      const c = Object.values(COLORS)[i];
                      return (
                        <div key={i} style={{marginBottom:12}}>
                          <div style={{display:"flex",justifyContent:"space-between",marginBottom:4}}>
                            <span style={{fontSize:11,color:"var(--text-mid)"}}>{t}</span>
                            <span style={{fontSize:11,color:c,fontFamily:"'Share Tech Mono',monospace"}}>{count}</span>
                          </div>
                          <div style={{height:5,background:"var(--bg-panel)",borderRadius:3,overflow:"hidden"}}>
                            <div style={{height:"100%",width:`${count*2}%`,background:c,
                              borderRadius:3,transition:"width 0.6s ease"}}/>
                          </div>
                        </div>
                      );
                    })}
                    <div style={{marginTop:20,padding:"12px",background:"var(--bg-panel)",
                      borderRadius:5,border:"1px solid var(--border)"}}>
                      <div style={{fontSize:10,color:"var(--text-dim)",textTransform:"uppercase",
                        letterSpacing:2,marginBottom:6}}>Report Period</div>
                      <div style={{fontFamily:"'Share Tech Mono',monospace",fontSize:12,color:"var(--text-hi)"}}>
                        {new Date(Date.now()-7*86400000).toLocaleDateString()} — {new Date().toLocaleDateString()}
                      </div>
                      <div style={{marginTop:10,display:"flex",gap:8}}>
                        <button style={{flex:1,padding:"7px",background:"var(--cyan-glow)",
                          border:"1px solid var(--cyan-dim)",borderRadius:4,color:"var(--cyan)",
                          cursor:"pointer",fontFamily:"'Rajdhani',sans-serif",fontWeight:700,
                          letterSpacing:1,fontSize:12,textTransform:"uppercase"}}>
                          Export JSON
                        </button>
                        <button style={{flex:1,padding:"7px",background:"rgba(0,255,136,0.05)",
                          border:"1px solid var(--green-dim)",borderRadius:4,color:"var(--green)",
                          cursor:"pointer",fontFamily:"'Rajdhani',sans-serif",fontWeight:700,
                          letterSpacing:1,fontSize:12,textTransform:"uppercase"}}>
                          Export CSV
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>

          {/* ── Ticker ── */}
          <div className="ticker-wrap">
            <span className="ticker-lbl">⚡ LIVE</span>
            <div className="ticker-scroll">
              <div className="ticker-inner">
                {[doubledTicker, doubledTicker].map((t,i) => (
                  <span key={i} className="ticker-item">{t || "Monitoring network traffic... All systems operational"}</span>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
