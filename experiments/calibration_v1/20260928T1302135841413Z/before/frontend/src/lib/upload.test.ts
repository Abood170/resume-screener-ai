import { afterEach, expect, it, vi } from "vitest";
import {
  predictResumeFile,
  predictResume,
  validateFile,
  MAX_FILE_BYTES,
} from "./api";

afterEach(() => vi.unstubAllGlobals());
const detailed = {
  predicted_category: "HR",
  confidence: 0.4,
  top_predictions: [
    { category: "HR", probability: 0.4 },
    { category: "SALES", probability: 0.2 },
    { category: "FINANCE", probability: 0.1 },
  ],
  top_terms: ["employee"],
  text_stats: { word_count: 8, short_input: true },
  source: "file",
  text_preview: "Employee training",
};

it("sends multipart without overriding the browser boundary", async () => {
  const fetch = vi
    .fn()
    .mockResolvedValue(new Response(JSON.stringify(detailed)));
  vi.stubGlobal("fetch", fetch);
  expect(
    await predictResumeFile(new File(["Employee training"], "cv.txt")),
  ).toEqual(detailed);
  const [url, options] = fetch.mock.calls[0];
  expect(url).toMatch(/\/predict\/file$/);
  expect(options.body).toBeInstanceOf(FormData);
  expect(options.body.get("file").name).toBe("cv.txt");
  expect(options.headers["Content-Type"]).toBeUndefined();
});

it.each([400, 413, 415, 422])(
  "displays a file error for HTTP %i",
  async (status) => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(JSON.stringify({ detail: "Unreadable CV" }), { status }),
        ),
    );
    await expect(
      predictResumeFile(new File(["contents"], "cv.pdf")),
    ).rejects.toMatchObject({ message: "Unreadable CV", status });
  },
);

it("rejects invalid files locally before fetching", async () => {
  const fetch = vi.fn();
  vi.stubGlobal("fetch", fetch);
  await expect(
    predictResumeFile(new File(["text"], "cv.exe")),
  ).rejects.toMatchObject({ kind: "validation" });
  expect(fetch).not.toHaveBeenCalled();
  expect(validateFile(new File([], "cv.txt"))).toContain("empty");
  expect(
    validateFile(new File([new Uint8Array(MAX_FILE_BYTES + 1)], "cv.txt")),
  ).toContain("too large");
  expect(validateFile(new File(["text"], "CV.TXT"))).toBeNull();
});

it("preserves detailed pasted-text results", async () => {
  const response = { ...detailed, source: "text", text_preview: null };
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(new Response(JSON.stringify(response))),
  );
  expect(await predictResume({ text: "Employee training" })).toEqual(response);
});

it("does not invent file details from a legacy response", async () => {
  vi.stubGlobal(
    "fetch",
    vi
      .fn()
      .mockResolvedValue(
        new Response(
          JSON.stringify({ predicted_category: "HR", confidence: 0.4 }),
        ),
      ),
  );
  await expect(
    predictResumeFile(new File(["text"], "cv.txt")),
  ).rejects.toMatchObject({ kind: "response" });
});

it("rejects invalid probabilities in expanded responses", async () => {
  vi.stubGlobal(
    "fetch",
    vi
      .fn()
      .mockResolvedValue(
        new Response(
          JSON.stringify({
            ...detailed,
            top_predictions: [
              ...detailed.top_predictions.slice(0, 2),
              { category: "OTHER", probability: 4 },
            ],
          }),
        ),
      ),
  );
  await expect(
    predictResumeFile(new File(["text"], "cv.txt")),
  ).rejects.toMatchObject({ kind: "response" });
});
