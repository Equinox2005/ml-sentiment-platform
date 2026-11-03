// The dev server and the Docker image proxy /api to the backend.
// Production builds without VITE_API_URL talk to the hosted API.
const HOSTED_API = "https://sentiment-api-ql9i.onrender.com";

export const API_URL = (
  import.meta.env.VITE_API_URL ?? (import.meta.env.DEV ? "/api" : HOSTED_API)
).replace(/\/+$/, "");

export const API_IS_REMOTE = /^https?:\/\//.test(API_URL);

// Same limits as the API
export const MAX_TEXT_LENGTH = 5000;
export const MAX_BATCH_SIZE = 50;

async function request(path, body) {
  const options =
    body === undefined
      ? undefined
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        };

  let res;
  try {
    res = await fetch(API_URL + path, options);
  } catch {
    throw new Error("Couldn't reach the API.");
  }

  const data = await res.json().catch(() => null);
  if (!res.ok) throw new Error(errorMessage(data, res.status));
  return data;
}

function errorMessage(data, status) {
  const detail = data?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail[0]?.msg) return detail[0].msg;
  return `The API returned an error (${status}).`;
}

export const getHealth = () => request("/health");
export const getModelInfo = () => request("/model");
export const predict = (text) => request("/predict", { text });
export const compare = (textA, textB) => request("/compare", { text_a: textA, text_b: textB });
export const predictBatch = (texts) => request("/predict/batch", { texts });
