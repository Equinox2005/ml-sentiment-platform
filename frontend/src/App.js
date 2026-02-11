import { useState, useCallback, useRef, useMemo } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  PieChart, Pie, Cell, ResponsiveContainer, Legend,
  LineChart, Line, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from "recharts";
import Papa from "papaparse";

/* ─── Theme ──────────────────────────────────────────────────────── */
const C = {
  positive: "#34d399", negative: "#f87171", neutral: "#fbbf24",
  bg: "#06060c", surface: "#0e0e18", surfaceAlt: "#111120",
  border: "#1e1e30", borderHover: "#2e2e48",
  text: "#e2e2ef", textDim: "#6b6b8a", textMuted: "#44445a",
  accent: "#a78bfa", accentAlt: "#7c3aed",
  compare1: "#38bdf8", compare2: "#fb923c",
  glow: "rgba(167,139,250,0.08)",
};

const ICONS = { positive: "▲", negative: "▼", neutral: "●" };

/* ─── API Config ─────────────────────────────────────────────────── */
const API_BASE = process.env.REACT_APP_API_URL || 'https://ml-sentiment-platform.onrender.com';

/* ─── Mock Classifier ────────────────────────────────────────────── */
const POS_WORDS = new Set(["love","great","amazing","fantastic","awesome","excellent","wonderful","happy","best","perfect","beautiful","incredible","brilliant","impressive","outstanding","superb","delightful","phenomenal","thrilled","recommend","enjoy","pleased","glad","fun","good","nice","cool","solid","smooth","fast","clean","helpful","friendly","reliable","quality","favorite","elegant","exciting","valuable","worth","premium","polished","intuitive","responsive","sleek","refined","powerful","efficient","comfortable","generous","charming"]);
const NEG_WORDS = new Set(["terrible","awful","horrible","worst","hate","bad","poor","disappointing","waste","garbage","useless","broken","scam","regret","disgusting","annoying","frustrating","defective","refund","pathetic","slow","ugly","rude","cheap","fail","crash","bug","error","mess","confusing","overpriced","mediocre","boring","painful","clunky","unreliable","flimsy","dreadful","junk","sucks","laggy","glitchy","unusable","uncomfortable","stiff","dated","tacky","noisy","fragile","bland"]);

function mockPredict(text) {
  const words = text.toLowerCase().replace(/[^a-z\s]/g, "").split(/\s+/).filter(Boolean);
  let pos = 0, neg = 0;
  const wordScores = [];
  words.forEach(w => {
    if (POS_WORDS.has(w)) { pos++; wordScores.push({ word: w, score: 0.8 + Math.random() * 0.2, type: "positive" }); }
    else if (NEG_WORDS.has(w)) { neg++; wordScores.push({ word: w, score: -(0.8 + Math.random() * 0.2), type: "negative" }); }
  });
  const total = Math.max(pos + neg, 1);
  let pP = 0.18 + (pos / total) * 0.68;
  let nP = 0.18 + (neg / total) * 0.68;
  if (pos === 0 && neg === 0) { pP = 0.28; nP = 0.28; }
  const nU = 0.14 + Math.random() * 0.04;
  const sum = pP + nP + nU;
  const probs = { positive: +(pP / sum).toFixed(4), negative: +(nP / sum).toFixed(4), neutral: +(nU / sum).toFixed(4) };
  const pred = Object.entries(probs).sort((a, b) => b[1] - a[1])[0][0];
  const topWords = wordScores.sort((a, b) => Math.abs(b.score) - Math.abs(a.score)).slice(0, 10);
  return {
    text, cleaned_text: text.toLowerCase().replace(/[^a-z\s]/g, ""),
    prediction: pred, confidence: probs[pred], probabilities: probs,
    inference_time_ms: +(Math.random() * 2 + 0.1).toFixed(2),
    timestamp: new Date().toISOString(),
    key_words: topWords,
  };
}

/* ─── Sample Prompts ─────────────────────────────────────────────── */
const SAMPLES = [
  { label: "5-Star Review", text: "Absolutely love this product! The quality is outstanding and it exceeded every expectation. Best purchase I've made all year.", icon: "⭐" },
  { label: "Angry Customer", text: "This is the worst experience I've ever had. The product arrived broken, support was rude, and I still haven't gotten a refund after 3 weeks.", icon: "😤" },
  { label: "Neutral Feedback", text: "The product is okay. It does what it says but nothing special. Delivery was on time and packaging was standard.", icon: "😐" },
  { label: "Mixed Feelings", text: "The design is beautiful and I love how it looks, but the performance is disappointing and the battery life is terrible.", icon: "🤔" },
  { label: "Tech Review", text: "Fast processor, smooth interface, and the camera quality is incredible. Some bugs in the software but overall a solid device.", icon: "💻" },
  { label: "Restaurant", text: "The food was disgusting and overpriced. Waited 45 minutes for cold pasta. The only good thing was our waiter who was very friendly.", icon: "🍽" },
];

const COMPARE_PRESETS = [
  { label: "Product vs Competitor", textA: "This phone has a stunning display, incredible camera, and the battery lasts all day. Premium build quality and smooth performance. Love it!", textB: "Disappointing screen quality, camera is mediocre at best. Battery barely lasts half a day and the build feels cheap and flimsy for the price.", icon: "📱" },
  { label: "Before vs After", textA: "The old version was slow, buggy, and frustrating to use. The interface was confusing and I regret buying it.", textB: "The new update is amazing! Everything is fast, smooth, and intuitive. They really listened to feedback. Loving the redesign.", icon: "🔄" },
  { label: "Positive vs Negative", textA: "Best customer service experience ever. The team was friendly, helpful, and resolved my issue in minutes. Highly recommend!", textB: "Worst customer service I've encountered. Rude staff, long wait times, and my problem is still unresolved after multiple calls. Terrible.", icon: "⚖️" },
];

/* ─── Helpers ────────────────────────────────────────────────────── */
function exportCSV(history) {
  const rows = [["text", "prediction", "confidence", "positive_prob", "negative_prob", "neutral_prob", "timestamp"]];
  history.forEach(h => {
    rows.push([`"${h.text.replace(/"/g, '""')}"`, h.prediction, h.confidence, h.probabilities.positive, h.probabilities.negative, h.probabilities.neutral, h.timestamp]);
  });
  const blob = new Blob([rows.map(r => r.join(",")).join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = `sentiment-results-${Date.now()}.csv`; a.click();
  URL.revokeObjectURL(url);
}

function exportJSON(history) {
  const blob = new Blob([JSON.stringify(history, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = `sentiment-results-${Date.now()}.json`; a.click();
  URL.revokeObjectURL(url);
}

/* ─── Shared Styles ──────────────────────────────────────────────── */
const card = { background: C.surface, borderRadius: 14, border: `1px solid ${C.border}`, padding: 24, transition: "border-color 0.2s" };
const labelStyle = { fontSize: 10, fontWeight: 600, color: C.textDim, textTransform: "uppercase", letterSpacing: "1.2px", marginBottom: 14, fontFamily: "'Courier New', monospace" };
const emptyBox = { display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", color: C.textMuted, fontSize: 13, gap: 6 };

/* ─── Tab Button ─────────────────────────────────────────────────── */
function Tab({ active, children, onClick }) {
  return (
    <button onClick={onClick} style={{
      padding: "8px 18px", borderRadius: 8, border: `1px solid ${active ? C.accent + "55" : C.border}`,
      background: active ? C.glow : "transparent", color: active ? C.accent : C.textDim,
      fontSize: 12, fontWeight: 600, cursor: "pointer", fontFamily: "'Courier New', monospace",
      transition: "all 0.2s",
    }}>{children}</button>
  );
}

/* ─── Word Cloud ─────────────────────────────────────────────────── */
function WordCloud({ history }) {
  const wordMap = useMemo(() => {
    const map = {};
    history.forEach(h => {
      (h.key_words || []).forEach(kw => {
        const key = kw.word.toLowerCase();
        if (!map[key]) map[key] = { word: kw.word, count: 0, posCount: 0, negCount: 0 };
        map[key].count++;
        if (kw.type === "positive") map[key].posCount++;
        else map[key].negCount++;
      });
    });
    return Object.values(map).sort((a, b) => b.count - a.count).slice(0, 40);
  }, [history]);

  if (wordMap.length === 0) return <div style={{ ...emptyBox, height: 180 }}>Word cloud builds from your predictions</div>;

  const maxCount = Math.max(...wordMap.map(w => w.count));

  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 8, justifyContent: "center", alignItems: "center", padding: "12px 0", minHeight: 140 }}>
      {wordMap.map((w, i) => {
        const ratio = w.count / maxCount;
        const size = 11 + ratio * 22;
        const opacity = 0.45 + ratio * 0.55;
        const isPos = w.posCount >= w.negCount;
        const color = isPos ? C.positive : C.negative;
        return (
          <span key={i} title={`"${w.word}" appeared ${w.count}x (${w.posCount} positive, ${w.negCount} negative)`}
            style={{
              fontSize: size, fontWeight: ratio > 0.5 ? 700 : 500,
              color, opacity, fontFamily: "'Courier New', monospace",
              padding: "3px 8px", borderRadius: 6,
              background: `${color}${ratio > 0.4 ? "12" : "08"}`,
              cursor: "default", transition: "transform 0.15s",
              lineHeight: 1.3,
            }}
            onMouseEnter={e => e.target.style.transform = "scale(1.15)"}
            onMouseLeave={e => e.target.style.transform = "scale(1)"}>
            {w.word}
          </span>
        );
      })}
    </div>
  );
}

/* ─── Comparison Panel ───────────────────────────────────────────── */
function ComparisonResult({ result, label, color }) {
  if (!result) return (
    <div style={{ ...emptyBox, height: 160 }}>
      <div style={{ fontSize: 28, opacity: 0.15 }}>◎</div>
      <span>Enter text above and compare</span>
    </div>
  );
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, padding: 14, borderRadius: 10, background: `${C[result.prediction]}0d`, border: `1px solid ${C[result.prediction]}28` }}>
        <div style={{ fontSize: 22, color: C[result.prediction], width: 40, height: 40, display: "flex", alignItems: "center", justifyContent: "center", borderRadius: 9, background: `${C[result.prediction]}15` }}>{ICONS[result.prediction]}</div>
        <div>
          <div style={{ fontSize: 18, fontWeight: 700, color: C[result.prediction], textTransform: "uppercase", letterSpacing: "1.5px" }}>{result.prediction}</div>
          <div style={{ fontSize: 11, color: C.textDim, marginTop: 1 }}>{(result.confidence * 100).toFixed(1)}% confidence</div>
        </div>
      </div>
      {Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]).map(([lbl, prob]) => (
        <div key={lbl} style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ width: 62, fontSize: 10, fontFamily: "monospace", color: C.textDim }}>{lbl}</span>
          <div style={{ flex: 1, height: 18, borderRadius: 4, background: C.bg, overflow: "hidden" }}>
            <div style={{ height: "100%", borderRadius: 4, background: C[lbl], width: `${prob * 100}%`, transition: "width 0.5s ease", display: "flex", alignItems: "center", justifyContent: "flex-end", paddingRight: prob > 0.1 ? 5 : 0 }}>
              {prob > 0.08 && <span style={{ fontSize: 9, fontWeight: 700, fontFamily: "monospace", color: "#fff" }}>{(prob * 100).toFixed(1)}%</span>}
            </div>
          </div>
        </div>
      ))}
      {result.key_words?.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 5, marginTop: 2 }}>
          {result.key_words.slice(0, 6).map((kw, i) => (
            <span key={i} style={{
              padding: "3px 9px", borderRadius: 14, fontSize: 10, fontWeight: 600,
              fontFamily: "monospace",
              background: kw.type === "positive" ? `${C.positive}15` : `${C.negative}15`,
              color: kw.type === "positive" ? C.positive : C.negative,
              border: `1px solid ${kw.type === "positive" ? C.positive : C.negative}25`,
            }}>
              {kw.type === "positive" ? "+" : "−"} {kw.word}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════ */
export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [activeTab, setActiveTab] = useState("single");
  const [csvResults, setCsvResults] = useState([]);
  const [csvProcessing, setCsvProcessing] = useState(false);
  const [showExport, setShowExport] = useState(false);
  const fileRef = useRef();

  // Comparison state
  const [compareA, setCompareA] = useState("");
  const [compareB, setCompareB] = useState("");
  const [compareResultA, setCompareResultA] = useState(null);
  const [compareResultB, setCompareResultB] = useState(null);

  /* Single prediction */
  const analyze = useCallback(async () => {
    if (!text.trim()) return;
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error('API error');
      const r = await res.json();
      setResult(r);
      setHistory(prev => [r, ...prev].slice(0, 200));
    } catch {
      // Fallback to mock if API is unavailable
      const r = mockPredict(text);
      setResult(r);
      setHistory(prev => [r, ...prev].slice(0, 200));
    }
  }, [text]);

  /* Compare */
  const runCompare = useCallback(async () => {
    if (!compareA.trim() || !compareB.trim()) return;
    try {
      const res = await fetch(`${API_BASE}/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text_a: compareA, text_b: compareB }),
      });
      if (!res.ok) throw new Error('API error');
      const data = await res.json();
      setCompareResultA(data.result_a);
      setCompareResultB(data.result_b);
      setHistory(prev => [data.result_b, data.result_a, ...prev].slice(0, 200));
    } catch {
      const rA = mockPredict(compareA);
      const rB = mockPredict(compareB);
      setCompareResultA(rA);
      setCompareResultB(rB);
      setHistory(prev => [rB, rA, ...prev].slice(0, 200));
    }
  }, [compareA, compareB]);

  /* CSV upload */
  const handleCSV = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setCsvProcessing(true);
    Papa.parse(file, {
      header: true, skipEmptyLines: true,
      complete: async (res) => {
        const textCol = Object.keys(res.data[0] || {}).find(k => /text|review|comment|content|message|body/i.test(k));
        if (!textCol) { alert("Could not find a text column. Please ensure your CSV has a column named 'text', 'review', 'comment', 'content', or 'message'."); setCsvProcessing(false); return; }
        const texts = res.data.slice(0, 500).filter(row => row[textCol]?.trim()).map(row => row[textCol]);
        try {
          // Process in batches of 50
          let allResults = [];
          for (let i = 0; i < texts.length; i += 50) {
            const batch = texts.slice(i, i + 50);
            const batchRes = await fetch(`${API_BASE}/predict/batch`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ texts: batch }),
            });
            if (!batchRes.ok) throw new Error('API error');
            const data = await batchRes.json();
            allResults = [...allResults, ...data.results];
          }
          setCsvResults(allResults);
          setHistory(prev => [...allResults, ...prev].slice(0, 500));
        } catch {
          // Fallback to mock
          const results = texts.map(t => mockPredict(t));
          setCsvResults(results);
          setHistory(prev => [...results, ...prev].slice(0, 500));
        }
        setCsvProcessing(false);
      },
      error: () => { setCsvProcessing(false); alert("Error parsing CSV"); },
    });
    e.target.value = "";
  }, []);

  /* Derived stats */
  const source = activeTab === "csv" && csvResults.length ? csvResults : history;
  const labelCounts = source.reduce((a, h) => { a[h.prediction] = (a[h.prediction] || 0) + 1; return a; }, {});
  const pieData = Object.entries(labelCounts).map(([name, value]) => ({ name, value }));
  const barData = result ? Object.entries(result.probabilities).map(([name, value]) => ({ name: name[0].toUpperCase() + name.slice(1), value: +(value * 100).toFixed(1), fill: C[name] })) : [];
  const avgConf = source.length ? (source.reduce((s, h) => s + h.confidence, 0) / source.length * 100).toFixed(1) : "—";

  const timeData = [...source].reverse().slice(-30).map((h, i) => ({
    idx: i + 1,
    positive: +(h.probabilities.positive * 100).toFixed(1),
    negative: +(h.probabilities.negative * 100).toFixed(1),
    neutral: +(h.probabilities.neutral * 100).toFixed(1),
  }));

  const csvSummary = csvResults.length ? {
    total: csvResults.length,
    pos: csvResults.filter(r => r.prediction === "positive").length,
    neg: csvResults.filter(r => r.prediction === "negative").length,
    neu: csvResults.filter(r => r.prediction === "neutral").length,
    avgConf: (csvResults.reduce((s, r) => s + r.confidence, 0) / csvResults.length * 100).toFixed(1),
  } : null;

  /* Compare radar data */
  const radarData = (compareResultA && compareResultB) ? [
    { metric: "Positive", A: +(compareResultA.probabilities.positive * 100).toFixed(1), B: +(compareResultB.probabilities.positive * 100).toFixed(1) },
    { metric: "Negative", A: +(compareResultA.probabilities.negative * 100).toFixed(1), B: +(compareResultB.probabilities.negative * 100).toFixed(1) },
    { metric: "Neutral", A: +(compareResultA.probabilities.neutral * 100).toFixed(1), B: +(compareResultB.probabilities.neutral * 100).toFixed(1) },
    { metric: "Confidence", A: +(compareResultA.confidence * 100).toFixed(1), B: +(compareResultB.confidence * 100).toFixed(1) },
  ] : [];

  return (
    <div style={{ minHeight: "100vh", background: `radial-gradient(ellipse at 20% 0%, #12102a 0%, ${C.bg} 55%)`, color: C.text, fontFamily: "'Segoe UI', system-ui, -apple-system, sans-serif" }}>

      {/* ─── Header ──────────────────────────────────────────────── */}
      <header style={{ borderBottom: `1px solid ${C.border}`, padding: "18px 40px", display: "flex", alignItems: "center", justifyContent: "space-between", background: "rgba(6,6,12,0.85)", backdropFilter: "blur(16px)", position: "sticky", top: 0, zIndex: 20 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 32, height: 32, borderRadius: 8, background: `linear-gradient(135deg, ${C.accent}, ${C.accentAlt})`, display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 800, fontSize: 14, color: "#fff" }}>S</div>
          <span style={{ fontSize: 16, fontWeight: 600, letterSpacing: "-0.4px" }}>Sentiment Platform</span>
          <span style={{ fontSize: 10, padding: "3px 8px", borderRadius: 6, background: C.glow, color: C.accent, fontFamily: "monospace", marginLeft: 4 }}>v3.0</span>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {history.length > 0 && (
            <div style={{ position: "relative" }}>
              <button onClick={() => setShowExport(!showExport)} style={{ padding: "7px 14px", borderRadius: 8, border: `1px solid ${C.border}`, background: "transparent", color: C.textDim, fontSize: 11, cursor: "pointer", fontFamily: "monospace", fontWeight: 600 }}>↓ Export</button>
              {showExport && (
                <div style={{ position: "absolute", top: "110%", right: 0, background: C.surfaceAlt, border: `1px solid ${C.border}`, borderRadius: 10, padding: 6, zIndex: 30, minWidth: 140 }}>
                  <button onClick={() => { exportCSV(history); setShowExport(false); }} style={{ display: "block", width: "100%", padding: "9px 14px", background: "transparent", border: "none", color: C.text, fontSize: 12, cursor: "pointer", textAlign: "left", borderRadius: 6 }} onMouseEnter={e => e.target.style.background = C.border} onMouseLeave={e => e.target.style.background = "transparent"}>Export as CSV</button>
                  <button onClick={() => { exportJSON(history); setShowExport(false); }} style={{ display: "block", width: "100%", padding: "9px 14px", background: "transparent", border: "none", color: C.text, fontSize: 12, cursor: "pointer", textAlign: "left", borderRadius: 6 }} onMouseEnter={e => e.target.style.background = C.border} onMouseLeave={e => e.target.style.background = "transparent"}>Export as JSON</button>
                </div>
              )}
            </div>
          )}
          <span style={{ fontSize: 10, padding: "4px 10px", borderRadius: 20, border: `1px solid ${C.border}`, color: C.textMuted, fontFamily: "monospace" }}>scikit-learn + FastAPI + React</span>
        </div>
      </header>

      <main style={{ maxWidth: 1200, margin: "0 auto", padding: "28px 40px" }}>

        {/* ─── Stats Row ─────────────────────────────────────────── */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 16, marginBottom: 20 }}>
          {[
            { v: source.length, l: "Predictions", c: C.accent },
            { v: labelCounts.positive || 0, l: "Positive", c: C.positive },
            { v: labelCounts.negative || 0, l: "Negative", c: C.negative },
            { v: avgConf + "%", l: "Avg Confidence", c: C.neutral },
          ].map((s, i) => (
            <div key={i} style={{ ...card, padding: "18px 16px", textAlign: "center" }}>
              <div style={{ fontSize: 28, fontWeight: 700, fontFamily: "'Courier New', monospace", color: s.c }}>{s.v}</div>
              <div style={{ fontSize: 10, color: C.textDim, marginTop: 3, textTransform: "uppercase", letterSpacing: "0.6px" }}>{s.l}</div>
            </div>
          ))}
        </div>

        {/* ─── Mode Tabs ─────────────────────────────────────────── */}
        <div style={{ display: "flex", gap: 8, marginBottom: 20 }}>
          <Tab active={activeTab === "single"} onClick={() => setActiveTab("single")}>Single Text</Tab>
          <Tab active={activeTab === "compare"} onClick={() => setActiveTab("compare")}>Compare</Tab>
          <Tab active={activeTab === "csv"} onClick={() => setActiveTab("csv")}>CSV Upload</Tab>
        </div>

        {/* ══════ SINGLE MODE ════════════════════════════════════════ */}
        {activeTab === "single" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 20 }}>
            <div style={card}>
              <div style={labelStyle}>Input Text</div>
              <textarea value={text} onChange={e => setText(e.target.value)}
                onKeyDown={e => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) analyze(); }}
                placeholder="Type or paste text to analyze..."
                style={{ width: "100%", minHeight: 120, background: C.bg, border: `1px solid ${C.border}`, borderRadius: 10, padding: 14, color: C.text, fontSize: 14, fontFamily: "inherit", lineHeight: 1.6, resize: "vertical", outline: "none", boxSizing: "border-box" }}
                onFocus={e => e.target.style.borderColor = C.accent + "66"} onBlur={e => e.target.style.borderColor = C.border}
              />
              <button onClick={analyze} disabled={!text.trim()}
                style={{ width: "100%", padding: 13, borderRadius: 10, border: "none", background: !text.trim() ? C.border : `linear-gradient(135deg,${C.accent},${C.accentAlt})`, color: "#fff", fontSize: 13, fontWeight: 600, cursor: text.trim() ? "pointer" : "not-allowed", marginTop: 12, opacity: text.trim() ? 1 : 0.45 }}>
                ⌘ Analyze Sentiment
              </button>
              <div style={{ marginTop: 18 }}>
                <div style={{ ...labelStyle, marginBottom: 10 }}>Quick Samples</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                  {SAMPLES.map((s, i) => (
                    <button key={i} onClick={() => setText(s.text)}
                      style={{ padding: "10px 12px", borderRadius: 9, border: `1px solid ${C.border}`, background: C.bg, color: C.textDim, fontSize: 11, cursor: "pointer", textAlign: "left", display: "flex", alignItems: "center", gap: 8, transition: "all 0.15s" }}
                      onMouseEnter={e => { e.currentTarget.style.borderColor = C.accent + "55"; e.currentTarget.style.color = C.text; }}
                      onMouseLeave={e => { e.currentTarget.style.borderColor = C.border; e.currentTarget.style.color = C.textDim; }}>
                      <span style={{ fontSize: 16, flexShrink: 0 }}>{s.icon}</span><span>{s.label}</span>
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div style={card}>
              <div style={labelStyle}>Prediction Result</div>
              {result ? (
                <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 14, padding: 16, borderRadius: 11, background: `${C[result.prediction]}0d`, border: `1px solid ${C[result.prediction]}28` }}>
                    <div style={{ fontSize: 24, color: C[result.prediction], width: 44, height: 44, display: "flex", alignItems: "center", justifyContent: "center", borderRadius: 10, background: `${C[result.prediction]}15` }}>{ICONS[result.prediction]}</div>
                    <div>
                      <div style={{ fontSize: 20, fontWeight: 700, color: C[result.prediction], textTransform: "uppercase", letterSpacing: "1.5px" }}>{result.prediction}</div>
                      <div style={{ fontSize: 12, color: C.textDim, marginTop: 2 }}>{(result.confidence * 100).toFixed(1)}% confidence</div>
                    </div>
                  </div>
                  {Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]).map(([lbl, prob]) => (
                    <div key={lbl} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <span style={{ width: 68, fontSize: 11, fontFamily: "monospace", color: C.textDim }}>{lbl}</span>
                      <div style={{ flex: 1, height: 20, borderRadius: 5, background: C.bg, overflow: "hidden" }}>
                        <div style={{ height: "100%", borderRadius: 5, background: C[lbl], width: `${prob * 100}%`, transition: "width 0.5s ease", display: "flex", alignItems: "center", justifyContent: "flex-end", paddingRight: prob > 0.1 ? 6 : 0 }}>
                          {prob > 0.08 && <span style={{ fontSize: 9, fontWeight: 700, fontFamily: "monospace", color: "#fff" }}>{(prob * 100).toFixed(1)}%</span>}
                        </div>
                      </div>
                    </div>
                  ))}
                  {result.key_words?.length > 0 && (
                    <div style={{ marginTop: 4 }}>
                      <div style={{ ...labelStyle, marginBottom: 10 }}>Key Words Detected</div>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                        {result.key_words.map((kw, i) => (
                          <span key={i} style={{
                            padding: "5px 12px", borderRadius: 20, fontSize: 11, fontWeight: 600, fontFamily: "monospace",
                            background: kw.type === "positive" ? `${C.positive}18` : `${C.negative}18`,
                            color: kw.type === "positive" ? C.positive : C.negative,
                            border: `1px solid ${kw.type === "positive" ? C.positive : C.negative}30`,
                          }}>{kw.type === "positive" ? "+" : "−"} {kw.word}</span>
                        ))}
                      </div>
                    </div>
                  )}
                  <div style={{ fontSize: 10, fontFamily: "monospace", color: C.textMuted, textAlign: "right" }}>Inference: {result.inference_time_ms}ms</div>
                </div>
              ) : (
                <div style={{ ...emptyBox, height: 260 }}><div style={{ fontSize: 36, opacity: 0.15 }}>◎</div><span>Enter text or select a sample</span></div>
              )}
            </div>
          </div>
        )}

        {/* ══════ COMPARE MODE ═══════════════════════════════════════ */}
        {activeTab === "compare" && (
          <div style={{ marginBottom: 20 }}>
            {/* Preset buttons */}
            <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
              <span style={{ fontSize: 10, color: C.textMuted, fontFamily: "monospace", alignSelf: "center", marginRight: 4 }}>PRESETS:</span>
              {COMPARE_PRESETS.map((p, i) => (
                <button key={i} onClick={() => { setCompareA(p.textA); setCompareB(p.textB); }}
                  style={{ padding: "7px 14px", borderRadius: 8, border: `1px solid ${C.border}`, background: C.bg, color: C.textDim, fontSize: 11, cursor: "pointer", display: "flex", alignItems: "center", gap: 6, transition: "all 0.15s" }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor = C.accent + "55"; e.currentTarget.style.color = C.text; }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor = C.border; e.currentTarget.style.color = C.textDim; }}>
                  <span>{p.icon}</span> {p.label}
                </button>
              ))}
            </div>

            {/* Input side by side */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 20 }}>
              <div style={card}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
                  <div style={{ width: 10, height: 10, borderRadius: 3, background: C.compare1 }} />
                  <span style={{ ...labelStyle, marginBottom: 0 }}>Text A</span>
                </div>
                <textarea value={compareA} onChange={e => setCompareA(e.target.value)}
                  placeholder="Paste first text to compare..."
                  style={{ width: "100%", minHeight: 100, background: C.bg, border: `1px solid ${C.border}`, borderRadius: 10, padding: 14, color: C.text, fontSize: 13, fontFamily: "inherit", lineHeight: 1.5, resize: "vertical", outline: "none", boxSizing: "border-box" }}
                  onFocus={e => e.target.style.borderColor = C.compare1 + "66"} onBlur={e => e.target.style.borderColor = C.border}
                />
              </div>
              <div style={card}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
                  <div style={{ width: 10, height: 10, borderRadius: 3, background: C.compare2 }} />
                  <span style={{ ...labelStyle, marginBottom: 0 }}>Text B</span>
                </div>
                <textarea value={compareB} onChange={e => setCompareB(e.target.value)}
                  placeholder="Paste second text to compare..."
                  style={{ width: "100%", minHeight: 100, background: C.bg, border: `1px solid ${C.border}`, borderRadius: 10, padding: 14, color: C.text, fontSize: 13, fontFamily: "inherit", lineHeight: 1.5, resize: "vertical", outline: "none", boxSizing: "border-box" }}
                  onFocus={e => e.target.style.borderColor = C.compare2 + "66"} onBlur={e => e.target.style.borderColor = C.border}
                />
              </div>
            </div>

            <button onClick={runCompare} disabled={!compareA.trim() || !compareB.trim()}
              style={{ width: "100%", padding: 14, borderRadius: 10, border: "none", background: (!compareA.trim() || !compareB.trim()) ? C.border : `linear-gradient(135deg, ${C.compare1}, ${C.compare2})`, color: "#fff", fontSize: 14, fontWeight: 600, cursor: (compareA.trim() && compareB.trim()) ? "pointer" : "not-allowed", marginBottom: 20, opacity: (compareA.trim() && compareB.trim()) ? 1 : 0.45, letterSpacing: "0.3px" }}>
              ⚖ Compare Sentiment
            </button>

            {/* Results side by side */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 20 }}>
              <div style={{ ...card, borderColor: compareResultA ? C.compare1 + "33" : C.border }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
                  <div style={{ width: 10, height: 10, borderRadius: 3, background: C.compare1 }} />
                  <span style={{ ...labelStyle, marginBottom: 0 }}>Result A</span>
                </div>
                <ComparisonResult result={compareResultA} label="A" color={C.compare1} />
              </div>
              <div style={{ ...card, borderColor: compareResultB ? C.compare2 + "33" : C.border }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
                  <div style={{ width: 10, height: 10, borderRadius: 3, background: C.compare2 }} />
                  <span style={{ ...labelStyle, marginBottom: 0 }}>Result B</span>
                </div>
                <ComparisonResult result={compareResultB} label="B" color={C.compare2} />
              </div>
            </div>

            {/* Radar overlay chart */}
            {radarData.length > 0 && (
              <div style={card}>
                <div style={labelStyle}>Side-by-Side Radar</div>
                <ResponsiveContainer width="100%" height={260}>
                  <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="75%">
                    <PolarGrid stroke={C.border} />
                    <PolarAngleAxis dataKey="metric" tick={{ fill: C.textDim, fontSize: 11 }} />
                    <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fill: C.textMuted, fontSize: 9 }} />
                    <Radar name="Text A" dataKey="A" stroke={C.compare1} fill={C.compare1} fillOpacity={0.15} strokeWidth={2} />
                    <Radar name="Text B" dataKey="B" stroke={C.compare2} fill={C.compare2} fillOpacity={0.15} strokeWidth={2} />
                    <Legend formatter={v => <span style={{ color: C.textDim, fontSize: 11 }}>{v}</span>} />
                    <Tooltip contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, color: C.text, fontSize: 11 }} formatter={v => `${v}%`} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        )}

        {/* ══════ CSV MODE ═══════════════════════════════════════════ */}
        {activeTab === "csv" && (
          <div style={{ ...card, marginBottom: 20 }}>
            <div style={labelStyle}>CSV Bulk Analysis</div>
            <div onClick={() => fileRef.current?.click()}
              style={{ border: `2px dashed ${C.border}`, borderRadius: 12, padding: "36px 24px", textAlign: "center", cursor: "pointer", background: C.bg, transition: "border-color 0.2s" }}
              onMouseEnter={e => e.currentTarget.style.borderColor = C.accent + "55"} onMouseLeave={e => e.currentTarget.style.borderColor = C.border}
              onDragOver={e => { e.preventDefault(); e.currentTarget.style.borderColor = C.accent; }}
              onDragLeave={e => e.currentTarget.style.borderColor = C.border}
              onDrop={e => { e.preventDefault(); e.currentTarget.style.borderColor = C.border; const f = e.dataTransfer.files[0]; if (f) { const dt = new DataTransfer(); dt.items.add(f); fileRef.current.files = dt.files; fileRef.current.dispatchEvent(new Event("change", { bubbles: true })); } }}>
              <input ref={fileRef} type="file" accept=".csv,.tsv" onChange={handleCSV} style={{ display: "none" }} />
              <div style={{ fontSize: 32, marginBottom: 8, opacity: 0.25 }}>📄</div>
              <div style={{ color: C.textDim, fontSize: 14, fontWeight: 500 }}>{csvProcessing ? "Processing..." : "Drop a CSV file here or click to browse"}</div>
              <div style={{ color: C.textMuted, fontSize: 11, marginTop: 6 }}>Column named "text", "review", "comment", "content", or "message" • Max 500 rows</div>
            </div>
            {csvSummary && (
              <div style={{ marginTop: 20 }}>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 12, marginBottom: 16 }}>
                  {[
                    { v: csvSummary.total, l: "Total", c: C.accent },
                    { v: csvSummary.pos, l: "Positive", c: C.positive },
                    { v: csvSummary.neg, l: "Negative", c: C.negative },
                    { v: csvSummary.neu, l: "Neutral", c: C.neutral },
                    { v: csvSummary.avgConf + "%", l: "Avg Conf", c: C.textDim },
                  ].map((s, i) => (
                    <div key={i} style={{ textAlign: "center", padding: "14px 8px", borderRadius: 10, background: C.bg, border: `1px solid ${C.border}` }}>
                      <div style={{ fontSize: 22, fontWeight: 700, fontFamily: "monospace", color: s.c }}>{s.v}</div>
                      <div style={{ fontSize: 9, color: C.textDim, marginTop: 3, textTransform: "uppercase" }}>{s.l}</div>
                    </div>
                  ))}
                </div>
                <div style={{ maxHeight: 260, overflowY: "auto", borderRadius: 10, border: `1px solid ${C.border}` }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                    <thead>
                      <tr style={{ background: C.bg, position: "sticky", top: 0 }}>
                        <th style={{ padding: "10px 14px", textAlign: "left", color: C.textDim, fontFamily: "monospace", fontWeight: 600, fontSize: 10, textTransform: "uppercase" }}>Text</th>
                        <th style={{ padding: "10px 14px", textAlign: "center", color: C.textDim, fontFamily: "monospace", fontWeight: 600, fontSize: 10, width: 90 }}>Sentiment</th>
                        <th style={{ padding: "10px 14px", textAlign: "right", color: C.textDim, fontFamily: "monospace", fontWeight: 600, fontSize: 10, width: 70 }}>Conf</th>
                      </tr>
                    </thead>
                    <tbody>
                      {csvResults.slice(0, 100).map((r, i) => (
                        <tr key={i} style={{ borderTop: `1px solid ${C.border}` }}>
                          <td style={{ padding: "9px 14px", color: C.textDim, maxWidth: 400, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.text}</td>
                          <td style={{ padding: "9px 14px", textAlign: "center" }}>
                            <span style={{ fontSize: 10, fontWeight: 600, padding: "3px 8px", borderRadius: 5, background: `${C[r.prediction]}18`, color: C[r.prediction], fontFamily: "monospace" }}>{r.prediction}</span>
                          </td>
                          <td style={{ padding: "9px 14px", textAlign: "right", fontFamily: "monospace", color: C.textDim, fontSize: 11 }}>{(r.confidence * 100).toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ─── Charts Row ────────────────────────────────────────── */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 20 }}>
          <div style={card}>
            <div style={labelStyle}>{activeTab === "single" ? "Probability Distribution" : "Overall Distribution"}</div>
            {(activeTab === "single" ? barData.length > 0 : pieData.length > 0) ? (
              activeTab === "single" ? (
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={barData} margin={{ top: 8, right: 8, bottom: 0, left: -14 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                    <XAxis dataKey="name" tick={{ fill: C.textDim, fontSize: 10 }} />
                    <YAxis tick={{ fill: C.textDim, fontSize: 10 }} domain={[0, 100]} tickFormatter={v => `${v}%`} />
                    <Tooltip contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, color: C.text, fontSize: 11 }} formatter={v => `${v}%`} />
                    <Bar dataKey="value" radius={[5, 5, 0, 0]}>{barData.map((e, i) => <Cell key={i} fill={e.fill} />)}</Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={pieData.map(d => ({ name: d.name[0].toUpperCase() + d.name.slice(1), value: d.value, fill: C[d.name] }))} margin={{ top: 8, right: 8, bottom: 0, left: -14 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                    <XAxis dataKey="name" tick={{ fill: C.textDim, fontSize: 10 }} />
                    <YAxis tick={{ fill: C.textDim, fontSize: 10 }} />
                    <Tooltip contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, color: C.text, fontSize: 11 }} />
                    <Bar dataKey="value" radius={[5, 5, 0, 0]}>{pieData.map((e, i) => <Cell key={i} fill={C[e.name]} />)}</Bar>
                  </BarChart>
                </ResponsiveContainer>
              )
            ) : <div style={{ ...emptyBox, height: 200 }}>Chart appears after predictions</div>}
          </div>
          <div style={card}>
            <div style={labelStyle}>Sentiment Breakdown</div>
            {pieData.length > 0 ? (
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie data={pieData} cx="50%" cy="50%" innerRadius={48} outerRadius={78} paddingAngle={3} dataKey="value">
                    {pieData.map((e, i) => <Cell key={i} fill={C[e.name]} />)}
                  </Pie>
                  <Tooltip contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, color: C.text, fontSize: 11 }} />
                  <Legend formatter={v => <span style={{ color: C.textDim, fontSize: 11 }}>{v}</span>} />
                </PieChart>
              </ResponsiveContainer>
            ) : <div style={{ ...emptyBox, height: 200 }}>Distribution builds as you analyze</div>}
          </div>
        </div>

        {/* ─── Word Cloud ────────────────────────────────────────── */}
        <div style={{ ...card, marginBottom: 20 }}>
          <div style={labelStyle}>Word Cloud — Key Phrases Across All Predictions</div>
          <WordCloud history={history} />
        </div>

        {/* ─── Sentiment Trend ───────────────────────────────────── */}
        {timeData.length > 1 && (
          <div style={{ ...card, marginBottom: 20 }}>
            <div style={labelStyle}>Sentiment Trend</div>
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={timeData} margin={{ top: 8, right: 16, bottom: 0, left: -14 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                <XAxis dataKey="idx" tick={{ fill: C.textDim, fontSize: 10 }} />
                <YAxis tick={{ fill: C.textDim, fontSize: 10 }} domain={[0, 100]} tickFormatter={v => `${v}%`} />
                <Tooltip contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, color: C.text, fontSize: 11 }} formatter={v => `${v}%`} />
                <Line type="monotone" dataKey="positive" stroke={C.positive} strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="negative" stroke={C.negative} strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="neutral" stroke={C.neutral} strokeWidth={1.5} dot={false} strokeDasharray="4 4" />
                <Legend formatter={v => <span style={{ color: C.textDim, fontSize: 10 }}>{v}</span>} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* ─── History ───────────────────────────────────────────── */}
        <div style={card}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
            <div style={{ ...labelStyle, marginBottom: 0 }}>Prediction History</div>
            {history.length > 0 && (
              <button onClick={() => { setHistory([]); setCsvResults([]); setResult(null); setCompareResultA(null); setCompareResultB(null); }}
                style={{ fontSize: 10, padding: "4px 10px", borderRadius: 6, border: `1px solid ${C.border}`, background: "transparent", color: C.textMuted, cursor: "pointer", fontFamily: "monospace" }}
                onMouseEnter={e => e.target.style.color = C.negative} onMouseLeave={e => e.target.style.color = C.textMuted}>
                Clear All
              </button>
            )}
          </div>
          {history.length > 0 ? (
            <div style={{ maxHeight: 280, overflowY: "auto" }}>
              {history.slice(0, 50).map((h, i) => (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, padding: "11px 14px", borderRadius: 9, background: C.bg, border: `1px solid ${C.border}`, marginBottom: 6 }}>
                  <div style={{ width: 8, height: 8, borderRadius: "50%", background: C[h.prediction], flexShrink: 0 }} />
                  <div style={{ flex: 1, fontSize: 12, color: C.textDim, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{h.text}</div>
                  <span style={{ fontSize: 10, fontWeight: 600, padding: "3px 8px", borderRadius: 5, background: `${C[h.prediction]}18`, color: C[h.prediction], fontFamily: "monospace", flexShrink: 0 }}>
                    {h.prediction} {(h.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ ...emptyBox, height: 120 }}>Your prediction history will appear here</div>
          )}
        </div>
      </main>
    </div>
  );
}
