import { useRef, useState } from "react";
import { predict } from "../api";
import { SUBMIT_SHORTCUT } from "../lib/format";
import Result from "./Result";
import TextArea from "./TextArea";

const EXAMPLES = [
  {
    label: "5-star review",
    text: "Absolutely love this product! The quality is outstanding and it exceeded every expectation. Best purchase I've made all year.",
  },
  {
    label: "Angry customer",
    text: "This is the worst experience I've ever had. The product arrived broken, support was rude, and I still haven't gotten a refund after 3 weeks.",
  },
  {
    label: "Neutral feedback",
    text: "The product is okay. It does what it says but nothing special. Delivery was on time and packaging was standard.",
  },
  {
    label: "Mixed feelings",
    text: "The design is beautiful and I love how it looks, but the performance is disappointing and the battery life is terrible.",
  },
  {
    label: "Tech review",
    text: "Fast processor, smooth interface, and the camera quality is incredible. Some bugs in the software but overall a solid device.",
  },
  {
    label: "Restaurant",
    text: "The food was disgusting and overpriced. Waited 45 minutes for cold pasta. The only good thing was our waiter who was very friendly.",
  },
];

export default function SingleText({ onResults }) {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const latestRequest = useRef(0);

  async function analyze(value = text) {
    if (!value.trim()) return;
    // Only the most recent request updates the page, e.g. after clicking two examples quickly
    const request = ++latestRequest.current;
    setLoading(true);
    setError(null);
    try {
      const r = await predict(value);
      if (request !== latestRequest.current) return;
      setResult(r);
      onResults([r]);
    } catch (err) {
      if (request === latestRequest.current) setError(err.message);
    } finally {
      if (request === latestRequest.current) setLoading(false);
    }
  }

  function tryExample(example) {
    setText(example.text);
    analyze(example.text);
  }

  return (
    <div className="grid-2">
      <div>
        <label className="label" htmlFor="single-text">
          Text
        </label>
        <TextArea
          id="single-text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onSubmit={() => analyze()}
          placeholder="Type or paste text to analyze..."
        />
        <div className="actions">
          <button
            type="button"
            className="button button-primary"
            onClick={() => analyze()}
            disabled={!text.trim() || loading}
          >
            {loading ? "Analyzing..." : "Analyze"}
          </button>
          <span className="kbd">{SUBMIT_SHORTCUT}</span>
        </div>
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}

        <div className="examples">
          <span className="label">Examples</span>
          <div className="chips">
            {EXAMPLES.map((example) => (
              <button
                key={example.label}
                type="button"
                className="chip"
                onClick={() => tryExample(example)}
              >
                {example.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div aria-live="polite">
        <h2 className="label">Result</h2>
        {result ? <Result result={result} /> : <p className="empty">No result yet.</p>}
      </div>
    </div>
  );
}
