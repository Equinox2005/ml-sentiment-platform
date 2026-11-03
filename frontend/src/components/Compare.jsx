import { useRef, useState } from "react";
import { compare } from "../api";
import { SUBMIT_SHORTCUT } from "../lib/format";
import Result from "./Result";
import TextArea from "./TextArea";

const EXAMPLES = [
  {
    label: "Product vs competitor",
    a: "This phone has a stunning display, incredible camera, and the battery lasts all day. Premium build quality and smooth performance. Love it!",
    b: "Disappointing screen quality, camera is mediocre at best. Battery barely lasts half a day and the build feels cheap and flimsy for the price.",
  },
  {
    label: "Before vs after",
    a: "The old version was slow, buggy, and frustrating to use. The interface was confusing and I regret buying it.",
    b: "The new update is amazing! Everything is fast, smooth, and intuitive. They really listened to feedback. Loving the redesign.",
  },
  {
    label: "Positive vs negative",
    a: "Best customer service experience ever. The team was friendly, helpful, and resolved my issue in minutes. Highly recommend!",
    b: "Worst customer service I've encountered. Rude staff, long wait times, and my problem is still unresolved after multiple calls. Terrible.",
  },
];

function summarize({ delta }) {
  const points = delta.positive * 100;
  if (Math.abs(points) < 0.5) return "A and B score about the same.";
  const [more, less] = points > 0 ? ["A", "B"] : ["B", "A"];
  return `${more} is ${Math.abs(points).toFixed(1)} points more positive than ${less}.`;
}

export default function Compare({ onResults }) {
  const [textA, setTextA] = useState("");
  const [textB, setTextB] = useState("");
  const [comparison, setComparison] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const latestRequest = useRef(0);
  const ready = textA.trim() && textB.trim();

  async function run(a = textA, b = textB) {
    if (!a.trim() || !b.trim()) return;
    const request = ++latestRequest.current;
    setLoading(true);
    setError(null);
    try {
      const data = await compare(a, b);
      if (request !== latestRequest.current) return;
      setComparison(data);
      onResults([data.result_a, data.result_b]);
    } catch (err) {
      if (request === latestRequest.current) setError(err.message);
    } finally {
      if (request === latestRequest.current) setLoading(false);
    }
  }

  function tryExample(example) {
    setTextA(example.a);
    setTextB(example.b);
    run(example.a, example.b);
  }

  return (
    <>
      <div className="grid-2">
        <div>
          <label className="label" htmlFor="compare-a">
            Text A
          </label>
          <TextArea
            id="compare-a"
            value={textA}
            onChange={(e) => setTextA(e.target.value)}
            onSubmit={() => run()}
            placeholder="First text"
          />
        </div>
        <div>
          <label className="label" htmlFor="compare-b">
            Text B
          </label>
          <TextArea
            id="compare-b"
            value={textB}
            onChange={(e) => setTextB(e.target.value)}
            onSubmit={() => run()}
            placeholder="Second text"
          />
        </div>
      </div>

      <div className="actions">
        <button
          type="button"
          className="button button-primary"
          onClick={() => run()}
          disabled={!ready || loading}
        >
          {loading ? "Comparing..." : "Compare"}
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

      {comparison && (
        <div aria-live="polite">
          <p className="compare-summary">{summarize(comparison)}</p>
          <div className="grid-2">
            <div>
              <h2 className="label">Result A</h2>
              <Result result={comparison.result_a} keyWordLimit={5} />
            </div>
            <div>
              <h2 className="label">Result B</h2>
              <Result result={comparison.result_b} keyWordLimit={5} />
            </div>
          </div>
        </div>
      )}
    </>
  );
}
