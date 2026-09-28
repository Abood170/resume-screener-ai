import { afterEach, describe, expect, it, vi } from "vitest";
import { getHealth, predictResume } from "./api";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});
function respond(body: unknown, status = 200) {
  const fetch = vi
    .fn()
    .mockResolvedValue(
      new Response(JSON.stringify(body), {
        status,
        headers: { "Content-Type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetch);
  return fetch;
}
describe("FastAPI network boundary", () => {
  it("sends the request schema and returns a validated prediction", async () => {
    const fetch = respond({ predicted_category: "HR", confidence: 0.35 });
    expect(await predictResume({ text: "Employee training" })).toEqual({
      predicted_category: "HR",
      confidence: 0.35,
    });
    expect(fetch.mock.calls[0][0]).toMatch(/\/predict$/);
    expect(fetch.mock.calls[0][1]).toMatchObject({
      method: "POST",
      body: '{"text":"Employee training"}',
      credentials: "omit",
    });
  });
  it("checks the real health contract", async () => {
    respond({ status: "ok", model: "random_forest" });
    expect(await getHealth()).toEqual({ status: "ok", model: "random_forest" });
  });
  it("rejects a malformed health response", async () => {
    respond({ status: "ok" });
    await expect(getHealth()).rejects.toMatchObject({ kind: "response" });
  });
  it("extracts Pydantic validation messages without echoing input", async () => {
    respond(
      {
        detail: [
          {
            loc: ["body", "text"],
            msg: "Input should be a valid string",
            type: "string_type",
            input: "private text",
          },
        ],
      },
      422,
    );
    await expect(predictResume({ text: "" })).rejects.toMatchObject({
      kind: "validation",
      message: "Input should be a valid string",
      status: 422,
    });
  });
  it("handles the backend’s string validation detail", async () => {
    respond({ detail: "text contains no recognized resume vocabulary" }, 422);
    await expect(predictResume({ text: "!!!" })).rejects.toMatchObject({
      message: "text contains no recognized resume vocabulary",
    });
  });
  it("explains unavailable model artifacts", async () => {
    respond({ detail: "Model unavailable" }, 503);
    await expect(getHealth()).rejects.toMatchObject({
      kind: "server",
      status: 503,
    });
  });
  it("does not expose a server stack trace", async () => {
    respond({ detail: "Traceback: private internal path" }, 500);
    await expect(predictResume({ text: "Python" })).rejects.toMatchObject({
      message:
        "The server could not complete this request. Please try again shortly.",
    });
  });
  it("explains network / CORS failures", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new TypeError("Failed to fetch")),
    );
    await expect(getHealth()).rejects.toMatchObject({
      kind: "network",
      message: expect.stringContaining("CORS"),
    });
  });
  it.each([
    null,
    {},
    { predicted_category: "HR", confidence: -1 },
    { predicted_category: "HR", confidence: 1.1 },
    { predicted_category: "HR", confidence: "0.8" },
    { predicted_category: "", confidence: 0.4 },
  ])("rejects malformed prediction %#", async (body) => {
    respond(body);
    await expect(predictResume({ text: "Training" })).rejects.toMatchObject({
      kind: "response",
    });
  });
  it("rejects HTML returned as a successful response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("<html>Wrong service</html>")),
    );
    await expect(getHealth()).rejects.toMatchObject({ kind: "response" });
  });
  it("times out a hung request", async () => {
    vi.useFakeTimers();
    vi.stubGlobal(
      "fetch",
      vi.fn(
        (_url: string, init: RequestInit) =>
          new Promise((_resolve, reject) => {
            init.signal?.addEventListener("abort", () =>
              reject(new DOMException("Aborted", "AbortError")),
            );
          }),
      ),
    );
    const assertion = expect(getHealth()).rejects.toMatchObject({
      kind: "timeout",
    });
    await vi.advanceTimersByTimeAsync(8_000);
    await assertion;
  });
  it("preserves caller cancellation rather than displaying a network error", async () => {
    const controller = new AbortController();
    vi.stubGlobal(
      "fetch",
      vi.fn(
        (_url: string, init: RequestInit) =>
          new Promise((_resolve, reject) => {
            init.signal?.addEventListener("abort", () =>
              reject(new DOMException("Aborted", "AbortError")),
            );
          }),
      ),
    );
    const assertion = expect(
      getHealth(controller.signal),
    ).rejects.toMatchObject({ name: "AbortError" });
    controller.abort();
    await assertion;
  });
});
