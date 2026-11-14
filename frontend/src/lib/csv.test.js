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

  it("prefers a header that ends with a known name", () => {
    expect(findTextColumn(["reviewer", "rating", "review_text"])).toBe("review_text");
    expect(findTextColumn(["review_id", "reviewer_name", "review_date", "review_body"])).toBe("review_body");
    expect(findTextColumn(["message_type", "message_text"])).toBe("message_text");
    expect(findTextColumn(["comment_count", "comments"])).toBe("comments");
    expect(findTextColumn(["reviewerID", "reviewerName", "reviewText", "overall"])).toBe("reviewText");
  });

  it("does not take a count column for a plural name", () => {
    expect(findTextColumn(["body_clean", "score", "num_comments"])).toBe("body_clean");
    expect(findTextColumn(["review_headline", "star_rating", "total_reviews"])).toBe("review_headline");
  });

  it("still accepts a header that only contains a known name", () => {
    expect(findTextColumn(["stars", "review_headline"])).toBe("review_headline");
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
