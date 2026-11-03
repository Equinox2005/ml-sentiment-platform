import { Suspense, lazy, useMemo } from "react";
import { downloadFile, resultsToCsv } from "../lib/csv";
import { capitalize, labelsOf, percent } from "../lib/format";
import Summary from "./Summary";

// Recharts is most of the bundle, so load it once there is something to plot
const TrendChart = lazy(() => import("./TrendChart"));

const LIST_LIMIT = 100;
const CLOUD_LIMIT = 30;

export default function Session({ history, onClear }) {
  const date = new Date().toISOString().slice(0, 10);

  function exportCsv() {
    downloadFile(`sentiment-history-${date}.csv`, resultsToCsv(history, labelsOf(history)), "text/csv");
  }

  function exportJson() {
    const results = history.map(({ id, ...result }) => result);
    downloadFile(`sentiment-history-${date}.json`, JSON.stringify(results, null, 2), "application/json");
  }

  return (
    <section className="session" aria-labelledby="history-title">
      <div className="section-head">
        <h2 id="history-title" className="section-title">
          History
        </h2>
        <div className="section-actions">
          <button type="button" className="button button-small" onClick={exportCsv}>
            Export CSV
          </button>
          <button type="button" className="button button-small" onClick={exportJson}>
            Export JSON
          </button>
          <button type="button" className="button button-small" onClick={onClear}>
            Clear
          </button>
        </div>
      </div>

      <Summary results={history} totalLabel="Predictions" />

      <div className="grid-2 session-charts">
        <div>
          <h3 className="label">Positive probability per prediction</h3>
          {history.length > 1 ? (
            <Suspense fallback={<div className="chart-placeholder" />}>
              <TrendChart history={history} />
            </Suspense>
          ) : (
            <p className="empty">Needs at least two predictions.</p>
          )}
        </div>
        <div>
          <h3 className="label">Frequent key words</h3>
          <WordCloud history={history} />
        </div>
      </div>

      <h3 className="label">Recent</h3>
      <ul className="history">
        {history.slice(0, LIST_LIMIT).map((h) => (
          <li key={h.id} className={`history-item tone-${h.prediction}`}>
            <span className="dot" />
            <span className="history-text" title={h.text}>
              {h.text}
            </span>
            <span className="sentiment">{capitalize(h.prediction)}</span>
            <span className="num">{percent(h.confidence, 0)}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

function WordCloud({ history }) {
  const words = useMemo(() => {
    const byWord = new Map();
    for (const h of history) {
      for (const kw of h.key_words) {
        const entry = byWord.get(kw.word) ?? { word: kw.word, count: 0, positive: 0 };
        entry.count += 1;
        if (kw.type === "positive") entry.positive += 1;
        byWord.set(kw.word, entry);
      }
    }
    return [...byWord.values()].sort((a, b) => b.count - a.count).slice(0, CLOUD_LIMIT);
  }, [history]);

  if (words.length === 0) return <p className="empty">No key words yet.</p>;
  const maxCount = words[0].count;

  return (
    <div className="cloud">
      {words.map((w) => (
        <span
          key={w.word}
          className={`cloud-word tone-${w.positive * 2 >= w.count ? "positive" : "negative"}`}
          style={{ fontSize: 13 + ((w.count - 1) / Math.max(maxCount - 1, 1)) * 9 }}
          title={`${w.word}: in ${w.count} prediction${w.count === 1 ? "" : "s"}`}
        >
          {w.word}
        </span>
      ))}
    </div>
  );
}
