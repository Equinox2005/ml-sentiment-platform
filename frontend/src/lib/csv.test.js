import { describe, expect, it } from "vitest";
import { findTextColumn, resultsToCsv } from "./csv";

describe("findTextColumn", () => {
  it("prefers an exact column name", () => {
    expect(findTextColumn(["id", "review_id", "Review"])).toBe("Review");
  });

  it("falls back to a header that contains a known name", () => {
    expect(findTextColumn(["id", "review_text", "stars"])).toBe("review_text");
  });

  it("skips id columns", () => {
    expect(findTextColumn(["review_id", "stars"])).toBeNull();
  });
});

describe("resultsToCsv", () => {
  const result = {
    text: 'He said "fine", =1+1',
    prediction: "positive",
    confidence: 0.9,
    probabilities: { negative: 0.1, positive: 0.9 },
    timestamp: "2026-01-01T00:00:00Z",
  };

  it("writes a probability column per label", () => {
    const [header, row] = resultsToCsv([result], ["positive", "negative"]).split("\r\n");
    expect(header).toBe("text,prediction,confidence,positive_probability,negative_probability,timestamp");
    expect(row).toBe('"He said ""fine"", =1+1",positive,0.9,0.9,0.1,2026-01-01T00:00:00Z');
  });

  it("neutralizes cells that start like a formula", () => {
    const csv = resultsToCsv([{ ...result, text: "=HYPERLINK(1)" }], ["positive"]);
    expect(csv.split("\r\n")[1].startsWith("\"'=HYPERLINK(1)\"")).toBe(true);
  });
});
