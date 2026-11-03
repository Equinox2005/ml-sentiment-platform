import { capitalize, orderLabels, percent } from "../lib/format";

export default function Result({ result, keyWordLimit = 8 }) {
  const labels = orderLabels(Object.keys(result.probabilities));
  const keyWords = result.key_words.slice(0, keyWordLimit);
  const maxScore = Math.max(0, ...keyWords.map((kw) => kw.score));

  return (
    <div>
      <div className={`result-head tone-${result.prediction}`}>
        <span className="result-label">{capitalize(result.prediction)}</span>
        <span className="result-confidence">{percent(result.confidence)} confidence</span>
      </div>

      <div className="probs">
        {labels.map((label) => (
          <div key={label} className={`prob-row tone-${label}`}>
            <span className="prob-name">{capitalize(label)}</span>
            <span className="bar">
              <span className="bar-fill" style={{ width: percent(result.probabilities[label]) }} />
            </span>
            <span className="num">{percent(result.probabilities[label])}</span>
          </div>
        ))}
      </div>

      {keyWords.length > 0 && (
        <>
          <h3 className="label" title="TF-IDF value times model weight: how far each term pushed the score">
            Key words
          </h3>
          <ul className="key-words">
            {keyWords.map((kw) => (
              <li key={kw.word} className={`key-word tone-${kw.type}`}>
                <span className="key-word-text" title={kw.word}>
                  {kw.word}
                </span>
                <span className="bar">
                  <span className="bar-fill" style={{ width: `${(kw.score / maxScore) * 100}%` }} />
                </span>
                <span className="num">
                  {kw.type === "positive" ? "+" : "−"}
                  {kw.score.toFixed(2)}
                </span>
              </li>
            ))}
          </ul>
        </>
      )}

      <p className="meta">Inference {result.inference_time_ms} ms</p>
    </div>
  );
}
