import { expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import type { PredictionResponse, UncertaintyReason } from "../lib/api";
import { PredictionResult } from "./PredictionResult";

const result: PredictionResponse = {
  predicted_category: "HR",
  confidence: 0.4,
  top_predictions: [
    { category: "HR", probability: 0.4 },
    { category: "SALES", probability: 0.3 },
    { category: "FINANCE", probability: 0.1 },
  ],
  top_terms: ["employee"],
  text_stats: { word_count: 70, short_input: false },
  source: "text",
  text_preview: null,
  calibrated_predicted_category: "SALES",
  calibrated_confidence: 0.6089,
  calibrated_top_predictions: [
    { category: "SALES", probability: 0.6089 },
    { category: "HR", probability: 0.2 },
    { category: "FINANCE", probability: 0.1 },
  ],
  is_uncertain: false,
  uncertainty_reason: null,
};
it("shows the calibrated winner and truncates displayed percentages", () => {
  const html = renderToStaticMarkup(
    <PredictionResult result={result} loading={false} />,
  );
  expect(html).toContain(">Sales</h3>");
  expect(html).toContain("Calibrated confidence");
  expect(html).toContain("60.8%");
  expect(html).not.toContain("60.9%");
  expect(html).not.toContain("40.0%");
  expect(html).toContain('aria-describedby="confidence-explanation"');
  expect(html).toContain('role="tooltip"');
  expect(html).toContain("not a guarantee of correctness");
});
it.each<[Exclude<UncertaintyReason, null>, string]>([
  ["short_input", "fewer than 50 words"],
  ["low_confidence", "No category has enough calibrated probability"],
  ["small_margin", "two leading categories are too close"],
])("shows neutral uncertainty and candidates for %s", (reason, explanation) => {
  const html = renderToStaticMarkup(
    <PredictionResult
      result={{ ...result, is_uncertain: true, uncertainty_reason: reason }}
      loading={false}
    />,
  );
  expect(html).toContain("The model isn&#x27;t confident about this CV");
  expect(html).toContain("Possible categories");
  expect(html).toContain(explanation);
  expect(html).not.toContain(">Sales</h3>");
  expect(html).not.toContain("Top predicted job category");
  expect((html.match(/role="progressbar"/g) ?? []).length).toBe(3);
  expect(html).toContain("bg-slate-400");
});
it("labels an old backend as uncalibrated without invented uncertainty", () => {
  const html = renderToStaticMarkup(
    <PredictionResult
      result={{ predicted_category: "HR", confidence: 0.4 }}
      loading={false}
    />,
  );
  expect(html).toContain("Model confidence (uncalibrated)");
  expect(html).toContain("Restart the updated backend");
  expect(html).not.toContain("Possible categories");
});
it("preserves empty and loading states", () => {
  expect(
    renderToStaticMarkup(<PredictionResult result={null} loading={false} />),
  ).toContain("Waiting for your first resume");
  const loading = renderToStaticMarkup(
    <PredictionResult result={result} loading={true} />,
  );
  expect(loading).toContain("Reading the signals");
  expect(loading).not.toContain(">Sales</h3>");
});
