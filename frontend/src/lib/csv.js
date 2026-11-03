import Papa from "papaparse";

export const TEXT_COLUMNS = ["text", "review", "comment", "content", "message", "body"];

/** Pick the column to analyze: an exact name first, then a header that contains one (but not an id). */
export function findTextColumn(fields) {
  const names = fields.map((field) => field.trim().toLowerCase());
  for (const wanted of TEXT_COLUMNS) {
    const i = names.indexOf(wanted);
    if (i !== -1) return fields[i];
  }
  const i = names.findIndex(
    (name) => TEXT_COLUMNS.some((wanted) => name.includes(wanted)) && !/(^|[\s_-])id$/.test(name)
  );
  return i === -1 ? null : fields[i];
}

export function parseCsv(file) {
  return new Promise((resolve, reject) => {
    Papa.parse(file, { header: true, skipEmptyLines: "greedy", complete: resolve, error: reject });
  });
}

/** One row per prediction with a probability column for each label. */
export function resultsToCsv(results, labels) {
  const rows = results.map((r) => ({
    text: r.text,
    prediction: r.prediction,
    confidence: r.confidence,
    ...Object.fromEntries(labels.map((label) => [`${label}_probability`, r.probabilities[label] ?? ""])),
    timestamp: r.timestamp,
  }));
  // escapeFormulae stops spreadsheet apps from running text like "=HYPERLINK(...)"
  return Papa.unparse(rows, { escapeFormulae: true });
}

export function downloadFile(filename, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
