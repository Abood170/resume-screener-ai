import { afterEach, expect, it, vi } from "vitest";
import { predictResume } from "./api";

afterEach(() => vi.unstubAllGlobals());

const response = {
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
  calibrated_confidence: 0.6,
  calibrated_top_predictions: [
    { category: "SALES", probability: 0.6 },
    { category: "HR", probability: 0.2 },
    { category: "FINANCE", probability: 0.1 },
  ],
  is_uncertain: false,
  uncertainty_reason: null,
};
function respond(body: unknown) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(new Response(JSON.stringify(body))),
  );
}
it("keeps calibrated and legacy categories distinct", async () => {
  respond(response);
  expect(await predictResume({ text: "skills" })).toEqual(response);
});
it.each(["low_confidence", "small_margin", "short_input"])(
  "preserves uncertainty reason %s",
  async (reason) => {
    const body = {
      ...response,
      is_uncertain: true,
      uncertainty_reason: reason,
      text_stats: {
        word_count: reason === "short_input" ? 6 : 70,
        short_input: reason === "short_input",
      },
    };
    respond(body);
    expect(await predictResume({ text: "skills" })).toEqual(body);
  },
);
it.each([
  { calibrated_confidence: 1.2 },
  { calibrated_predicted_category: "HR" },
  { is_uncertain: true, uncertainty_reason: null },
  { is_uncertain: false, uncertainty_reason: "low_confidence" },
  { is_uncertain: true, uncertainty_reason: "unknown" },
  { text_stats: { word_count: 6, short_input: true } },
  { is_uncertain: true, uncertainty_reason: "short_input" },
  {
    calibrated_top_predictions: [
      { category: "SALES", probability: 0.6 },
      { category: "HR", probability: 0.6 },
      { category: "FINANCE", probability: 0.5 },
    ],
  },
])("rejects inconsistent calibration fields %#", async (changes) => {
  respond({ ...response, ...changes });
  await expect(predictResume({ text: "skills" })).rejects.toMatchObject({
    kind: "response",
  });
});
it("rejects partial calibrated responses", async () => {
  const incomplete: Record<string, unknown> = { ...response };
  delete incomplete.calibrated_top_predictions;
  respond(incomplete);
  await expect(predictResume({ text: "skills" })).rejects.toMatchObject({
    kind: "response",
  });
});
it("rejects calibration fields on a response missing file/text details", async () => {
  respond({ predicted_category: "HR", confidence: 0.4, is_uncertain: false });
  await expect(predictResume({ text: "skills" })).rejects.toMatchObject({
    kind: "response",
  });
});
