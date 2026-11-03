import { capitalize, labelsOf, percent } from "../lib/format";

/** Counts per label, average confidence, and a bar with the label split. */
export default function Summary({ results, totalLabel }) {
  const labels = labelsOf(results);
  const counts = {};
  for (const r of results) counts[r.prediction] = (counts[r.prediction] ?? 0) + 1;
  const avgConfidence = results.reduce((sum, r) => sum + r.confidence, 0) / results.length;

  return (
    <div>
      <dl className="stats">
        <Stat label={totalLabel} value={results.length} />
        {labels.map((label) => (
          <Stat key={label} label={capitalize(label)} value={counts[label] ?? 0} tone={label} />
        ))}
        <Stat label="Avg. confidence" value={percent(avgConfidence)} />
      </dl>
      <div className="split" aria-hidden="true">
        {labels.map((label) =>
          counts[label] ? (
            <span key={label} className={`tone-${label}`} style={{ flex: `${counts[label]} 1 0` }} />
          ) : null
        )}
      </div>
    </div>
  );
}

function Stat({ label, value, tone }) {
  return (
    <div className={tone ? `tone-${tone}` : undefined}>
      <dt className="stat-label">
        {tone && <span className="dot" />}
        {label}
      </dt>
      <dd className="stat-value">{value}</dd>
    </div>
  );
}
