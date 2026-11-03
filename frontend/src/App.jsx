import { useCallback, useEffect, useRef, useState } from "react";
import { API_IS_REMOTE, API_URL, getHealth, getModelInfo } from "./api";
import Compare from "./components/Compare";
import CsvUpload from "./components/CsvUpload";
import Session from "./components/Session";
import SingleText from "./components/SingleText";
import { percent } from "./lib/format";

const REPO_URL = "https://github.com/Equinox2005/ml-sentiment-platform";
const MAX_HISTORY = 500;
const SLOW_API_MS = 2500;

const TABS = [
  { id: "single", label: "Single text", Panel: SingleText },
  { id: "compare", label: "Compare", Panel: Compare },
  { id: "csv", label: "CSV upload", Panel: CsvUpload },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("single");
  const [history, setHistory] = useState([]);
  const [apiStatus, setApiStatus] = useState("checking");
  const [modelInfo, setModelInfo] = useState(null);
  const nextId = useRef(0);

  const checkApi = useCallback(async () => {
    setApiStatus("checking");
    const slow = setTimeout(
      () => setApiStatus((status) => (status === "checking" ? "waking" : status)),
      SLOW_API_MS
    );
    try {
      await getHealth();
      setApiStatus("ready");
      setModelInfo(await getModelInfo().catch(() => null));
    } catch {
      setApiStatus("down");
    } finally {
      clearTimeout(slow);
    }
  }, []);

  useEffect(() => {
    checkApi();
  }, [checkApi]);

  // History is newest first; results arrive in the order they were sent
  const addResults = useCallback((results) => {
    const entries = results.map((result) => ({ ...result, id: nextId.current++ })).reverse();
    setHistory((prev) => [...entries, ...prev].slice(0, MAX_HISTORY));
  }, []);

  function handleTabKeys(event) {
    const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
    if (!step) return;
    const index = TABS.findIndex((tab) => tab.id === activeTab);
    const next = TABS[(index + step + TABS.length) % TABS.length];
    setActiveTab(next.id);
    document.getElementById(`tab-${next.id}`)?.focus();
  }

  return (
    <>
      <header className="header-bar">
        <div className="container header">
          <h1 className="brand">
            <span className="brand-mark" aria-hidden="true">
              S
            </span>
            Sentiment Platform
          </h1>
          <nav className="header-links">
            {API_IS_REMOTE && <a href={`${API_URL}/docs`}>API docs</a>}
            <a href={REPO_URL}>GitHub</a>
          </nav>
        </div>
      </header>

      <ApiNotice status={apiStatus} onRetry={checkApi} />

      <main className="container">
        <div className="tabs" role="tablist" aria-label="Mode" onKeyDown={handleTabKeys}>
          {TABS.map((tab) => (
            <button
              key={tab.id}
              id={`tab-${tab.id}`}
              type="button"
              role="tab"
              className="tab"
              aria-selected={tab.id === activeTab}
              aria-controls={`panel-${tab.id}`}
              tabIndex={tab.id === activeTab ? 0 : -1}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {TABS.map(({ id, Panel }) => (
          <div key={id} id={`panel-${id}`} role="tabpanel" aria-labelledby={`tab-${id}`} hidden={id !== activeTab}>
            <Panel onResults={addResults} />
          </div>
        ))}

        {history.length > 0 && <Session history={history} onClear={() => setHistory([])} />}
      </main>

      {modelInfo && <ModelFooter info={modelInfo} />}
    </>
  );
}

function ApiNotice({ status, onRetry }) {
  if (status === "waking") {
    return (
      <div className="notice" role="status">
        <div className="container">
          {API_IS_REMOTE
            ? "Waking up the API. It runs on a free instance that sleeps when idle, so this can take up to a minute."
            : `Waiting for the API at ${API_URL}...`}
        </div>
      </div>
    );
  }
  if (status === "down") {
    return (
      <div className="notice" role="alert">
        <div className="container">
          <span>
            Can't reach the API at <code>{API_URL}</code>.
          </span>
          <button type="button" className="link-button" onClick={onRetry}>
            Try again
          </button>
        </div>
      </div>
    );
  }
  return null;
}

function ModelFooter({ info }) {
  const count = (n) => n.toLocaleString("en-US");
  const examples = info.dataset === "Sentiment140" ? "tweets" : "examples";
  const parts = [
    info.model_type === "LogisticRegression" ? "Logistic regression on TF-IDF features" : info.model_type,
  ];
  if (info.train_samples) {
    const source = info.dataset ? ` from ${info.dataset}` : "";
    parts.push(`trained on ${count(info.train_samples)} ${examples}${source}`);
  }
  if (info.accuracy != null && info.test_samples) {
    parts.push(`${percent(info.accuracy)} accuracy on ${count(info.test_samples)} held-out ${examples}`);
  }
  return <footer className="container footer">{parts.join(" · ")}</footer>;
}
