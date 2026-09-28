/** The public FastAPI contract. Responses are validated at the network boundary. */
export interface ResumeRequest {
  text: string;
}
export interface PredictionResponse {
  predicted_category: string;
  confidence: number;
  // Optional only for compatibility with an older running backend. No details
  // are fabricated when an old server returns just its original two fields.
  top_predictions?: CategoryProbability[];
  top_terms?: string[];
  text_stats?: { word_count: number; short_input: boolean };
  source?: "text" | "file";
  text_preview?: string | null;
}
export interface CategoryProbability {
  category: string;
  probability: number;
}
export interface DetailedPredictionResponse extends PredictionResponse {
  top_predictions: CategoryProbability[];
  top_terms: string[];
  text_stats: { word_count: number; short_input: boolean };
  source: "text" | "file";
  text_preview: string | null;
}
export interface HealthResponse {
  status: "ok";
  model: string;
}
export interface ValidationError {
  loc: (string | number)[];
  msg: string;
  type: string;
  input?: unknown;
  ctx?: Record<string, unknown>;
}
export interface HTTPValidationError {
  detail: ValidationError[];
}
export interface HTTPError {
  detail: string;
}

export const MAX_CHARACTERS = 50_000;
export const MAX_FILE_BYTES = 5 * 1024 * 1024;
export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL?.trim() || "http://127.0.0.1:8000"
).replace(/\/+$/, "");
export type ApiErrorKind =
  | "network"
  | "timeout"
  | "validation"
  | "server"
  | "response";

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly kind: ApiErrorKind,
    public readonly status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function validationMessage(body: unknown): string {
  if (isRecord(body) && typeof body.detail === "string")
    return body.detail.slice(0, 400);
  if (isRecord(body) && Array.isArray(body.detail)) {
    const issues = body.detail.filter(
      (issue): issue is ValidationError =>
        isRecord(issue) &&
        typeof issue.msg === "string" &&
        Array.isArray(issue.loc) &&
        issue.loc.every(
          (part: unknown) =>
            typeof part === "string" || typeof part === "number",
        ) &&
        typeof issue.type === "string",
    );
    if (issues.length)
      return issues
        .slice(0, 3)
        .map((issue) => issue.msg.slice(0, 160))
        .join(" ");
  }
  return "Please check your resume text and try again.";
}

async function request(
  path: string,
  options: RequestInit,
  signal?: AbortSignal,
  timeoutMs = 30_000,
): Promise<unknown> {
  const controller = new AbortController();
  let timedOut = false;
  const cancel = () => controller.abort();
  signal?.addEventListener("abort", cancel, { once: true });
  if (signal?.aborted) cancel();
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      signal: controller.signal,
      credentials: "omit",
      cache: "no-store",
      headers: { Accept: "application/json", ...options.headers },
    });
    const body: unknown = await response.json().catch(() => null);
    if (!response.ok) {
      if ([400, 413, 415, 422].includes(response.status))
        throw new ApiError(
          validationMessage(body),
          "validation",
          response.status,
        );
      if (response.status === 503)
        throw new ApiError(
          "The API is reachable, but the model is unavailable. Check the backend startup logs and model files.",
          "server",
          503,
        );
      throw new ApiError(
        "The server could not complete this request. Please try again shortly.",
        "server",
        response.status,
      );
    }
    return body;
  } catch (error) {
    if (signal?.aborted)
      throw new DOMException("Request cancelled", "AbortError");
    if (timedOut)
      throw new ApiError(
        "The API took too long to respond. Please try again.",
        "timeout",
      );
    if (error instanceof ApiError) throw error;
    // Browsers intentionally do not distinguish CORS failures from network failures.
    throw new ApiError(
      "Cannot reach the API. Check that FastAPI is running and that CORS allows this frontend’s origin.",
      "network",
    );
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", cancel);
  }
}

export async function predictResume(
  payload: ResumeRequest,
  signal?: AbortSignal,
): Promise<PredictionResponse> {
  const body = await request(
    "/predict",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
    signal,
  );
  return parsePrediction(body);
}

function parsePrediction(body: unknown): PredictionResponse {
  if (
    !isRecord(body) ||
    typeof body.predicted_category !== "string" ||
    !body.predicted_category.trim() ||
    typeof body.confidence !== "number" ||
    !Number.isFinite(body.confidence) ||
    body.confidence < 0 ||
    body.confidence > 1
  ) {
    throw new ApiError(
      "The API returned an unexpected prediction format. Check that the expected backend is running.",
      "response",
    );
  }
  const base: PredictionResponse = {
    predicted_category: body.predicted_category,
    confidence: body.confidence,
  };
  const detailKeys = [
    "top_predictions",
    "top_terms",
    "text_stats",
    "source",
    "text_preview",
  ];
  if (detailKeys.every((key) => !(key in body))) return base;
  const predictions = body.top_predictions;
  const stats = body.text_stats;
  if (
    !Array.isArray(predictions) ||
    predictions.length !== 3 ||
    !predictions.every(
      (p) =>
        isRecord(p) &&
        typeof p.category === "string" &&
        p.category.trim() &&
        typeof p.probability === "number" &&
        Number.isFinite(p.probability) &&
        p.probability >= 0 &&
        p.probability <= 1,
    ) ||
    !Array.isArray(body.top_terms) ||
    body.top_terms.length > 8 ||
    !body.top_terms.every((term) => typeof term === "string") ||
    !isRecord(stats) ||
    !Number.isInteger(stats.word_count) ||
    (stats.word_count as number) < 0 ||
    typeof stats.short_input !== "boolean" ||
    !["text", "file"].includes(body.source as string) ||
    !(body.text_preview === null || typeof body.text_preview === "string") ||
    (typeof body.text_preview === "string" &&
      Array.from(body.text_preview).length > 500) ||
    (body.source === "file" && typeof body.text_preview !== "string")
  ) {
    throw new ApiError(
      "The API returned incomplete prediction details. Please check the backend version.",
      "response",
    );
  }
  const top = predictions as unknown as CategoryProbability[];
  if (
    top[0].category !== base.predicted_category ||
    top[0].probability !== base.confidence ||
    top.some((p, i) => i > 0 && p.probability > top[i - 1].probability) ||
    new Set(top.map((p) => p.category)).size !== 3
  ) {
    throw new ApiError(
      "The API returned inconsistent category probabilities.",
      "response",
    );
  }
  return {
    ...base,
    top_predictions: top,
    top_terms: body.top_terms as string[],
    text_stats: stats as DetailedPredictionResponse["text_stats"],
    source: body.source as "text" | "file",
    text_preview: body.text_preview as string | null,
  };
}

export function validateFile(file: File): string | null {
  if (!/\.(pdf|docx|txt)$/i.test(file.name))
    return "Choose a PDF, DOCX, or TXT file.";
  if (file.size > MAX_FILE_BYTES)
    return "Your file is too large. The maximum is 5 MB (5,242,880 bytes).";
  if (!file.size)
    return "This file is empty. Please choose a CV with readable text.";
  return null;
}

export async function predictResumeFile(
  file: File,
  signal?: AbortSignal,
): Promise<DetailedPredictionResponse> {
  const error = validateFile(file);
  if (error) throw new ApiError(error, "validation");
  const form = new FormData();
  form.append("file", file);
  // Let the browser set Content-Type and the multipart boundary.
  const body = await request(
    "/predict/file",
    { method: "POST", body: form },
    signal,
  );
  const result = parsePrediction(body);
  if (
    result.source !== "file" ||
    !result.top_predictions ||
    !result.top_terms ||
    !result.text_stats ||
    typeof result.text_preview !== "string"
  ) {
    throw new ApiError(
      "The API did not return extracted file details. Please restart the updated backend.",
      "response",
    );
  }
  return result as DetailedPredictionResponse;
}

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const body = await request("/health", { method: "GET" }, signal, 8_000);
  if (
    !isRecord(body) ||
    body.status !== "ok" ||
    typeof body.model !== "string" ||
    !body.model.trim()
  ) {
    throw new ApiError(
      "The API returned an unexpected health response.",
      "response",
    );
  }
  return { status: "ok", model: body.model };
}
