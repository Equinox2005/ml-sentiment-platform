const LABEL_ORDER = ["positive", "negative", "neutral"];

export function percent(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

export function capitalize(text) {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

export function orderLabels(labels) {
  return [...labels].sort((a, b) => LABEL_ORDER.indexOf(a) - LABEL_ORDER.indexOf(b));
}

/** Every label that appears in the results' probabilities, in display order. */
export function labelsOf(results) {
  return orderLabels(new Set(results.flatMap((r) => Object.keys(r.probabilities))));
}

export const SUBMIT_SHORTCUT = /Mac|iPhone|iPad/.test(navigator.userAgent) ? "⌘ Enter" : "Ctrl Enter";
