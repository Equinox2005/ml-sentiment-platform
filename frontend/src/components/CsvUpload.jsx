import { useRef, useState } from "react";
import { MAX_BATCH_SIZE, MAX_TEXT_LENGTH, predictBatch } from "../api";
import { TEXT_COLUMNS, downloadFile, findTextColumn, parseCsv, resultsToCsv } from "../lib/csv";
import { capitalize, labelsOf, percent } from "../lib/format";
import Summary from "./Summary";

const MAX_ROWS = 500;
const SAMPLE_FILE = `${import.meta.env.BASE_URL}sample-reviews.csv`;
const COLUMN_NAMES = `${TEXT_COLUMNS.slice(0, -1).join(", ")} or ${TEXT_COLUMNS.at(-1)}`;

export default function CsvUpload({ onResults }) {
  const inputRef = useRef(null);
  const runRef = useRef(0);
  const [dragging, setDragging] = useState(false);
  const [job, setJob] = useState(null);

  async function analyzeFile(file) {
    const run = ++runRef.current;
    const isCurrent = () => run === runRef.current;
    setJob({ status: "running", file: file.name, done: 0, total: 0 });

    try {
      const { data, meta } = await parseCsv(file);
      const column = findTextColumn(meta.fields ?? []);
      if (!column) throw new Error(`No text column found. Name one of the columns ${COLUMN_NAMES}.`);

      const texts = data.map((row) => (row[column] ?? "").trim()).filter(Boolean);
      if (texts.length === 0) throw new Error(`The "${column}" column is empty.`);
      const rows = texts.slice(0, MAX_ROWS).map((text) => text.slice(0, MAX_TEXT_LENGTH));

      const results = [];
      let skipped = 0;
      for (let i = 0; i < rows.length; i += MAX_BATCH_SIZE) {
        if (!isCurrent()) return;
        setJob({ status: "running", file: file.name, done: i, total: rows.length });
        const batch = await predictBatch(rows.slice(i, i + MAX_BATCH_SIZE));
        results.push(...batch.results);
        skipped += batch.skipped?.length ?? 0;
      }
      if (!isCurrent()) return;

      setJob({
        status: "done",
        file: file.name,
        results,
        skipped,
        notAnalyzed: texts.length - rows.length,
      });
      onResults(results);
    } catch (err) {
      if (isCurrent()) setJob({ status: "error", file: file.name, message: err.message });
    }
  }

  async function loadSample() {
    try {
      const res = await fetch(SAMPLE_FILE);
      if (!res.ok) throw new Error("Couldn't load the sample file.");
      analyzeFile(new File([await res.blob()], "sample-reviews.csv", { type: "text/csv" }));
    } catch (err) {
      setJob({ status: "error", file: "sample-reviews.csv", message: err.message });
    }
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files[0];
    if (file) analyzeFile(file);
  }

  function handleFileInput(event) {
    const file = event.target.files?.[0];
    if (file) analyzeFile(file);
    event.target.value = "";
  }

  function downloadResults() {
    const name = job.file.replace(/\.[^.]+$/, "");
    downloadFile(`${name}-sentiment.csv`, resultsToCsv(job.results, labelsOf(job.results)), "text/csv");
  }

  return (
    <>
      <div
        className={`dropzone${dragging ? " is-dragging" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        <input ref={inputRef} type="file" accept=".csv,.tsv,text/csv" hidden onChange={handleFileInput} />
        <p>
          Drop a CSV file here or{" "}
          <button
            type="button"
            className="link-button"
            onClick={(e) => {
              e.stopPropagation();
              inputRef.current?.click();
            }}
          >
            choose one
          </button>
        </p>
        <p className="hint">
          Uses the column named {COLUMN_NAMES}. Up to {MAX_ROWS} rows.
        </p>
      </div>
      <p className="meta">
        <button type="button" className="link-button" onClick={loadSample}>
          Use a sample file
        </button>
      </p>

      {job?.status === "running" && (
        <p className="meta" role="status">
          Analyzing {job.file}
          {job.total > 0 && ` (${job.done} of ${job.total} rows done)`}...
        </p>
      )}
      {job?.status === "error" && (
        <p className="error" role="alert">
          {job.file}: {job.message}
        </p>
      )}
      {job?.status === "done" && (
        <BatchResults job={job} onDownload={downloadResults} />
      )}
    </>
  );
}

function rows(count) {
  return count === 1 ? "1 row" : `${count} rows`;
}

function BatchResults({ job, onDownload }) {
  const notes = [];
  if (job.skipped) {
    notes.push(`Skipped ${rows(job.skipped)} with nothing left to analyze after removing links, numbers and symbols.`);
  }
  if (job.notAnalyzed) {
    notes.push(`Only the first ${MAX_ROWS} rows were analyzed; ${rows(job.notAnalyzed)} left out.`);
  }

  return (
    <div aria-live="polite">
      <div className="batch-head">
        <h2 className="section-title">{job.file}</h2>
        <button type="button" className="button button-small" onClick={onDownload}>
          Download results
        </button>
      </div>
      {job.results.length > 0 && <Summary results={job.results} totalLabel="Rows" />}
      {notes.length > 0 && <p className="hint batch-notes">{notes.join(" ")}</p>}

      {job.results.length > 0 && (
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>Text</th>
                <th>Sentiment</th>
                <th className="align-right">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {job.results.map((r, i) => (
                <tr key={i}>
                  <td className="cell-text" title={r.text}>
                    {r.text}
                  </td>
                  <td className={`sentiment tone-${r.prediction}`}>{capitalize(r.prediction)}</td>
                  <td className="num">{percent(r.confidence)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
