import type { PredictionResponse } from "../lib/api";
import { Icon, Spinner } from "./Icon";

function categoryLabel(category: string) {
  return category
    .split(/[-_\s]+/)
    .map((word) =>
      ["HR", "IT", "BPO"].includes(word)
        ? word
        : word.charAt(0).toUpperCase() + word.slice(1).toLowerCase(),
    )
    .join(" ");
}

export function PredictionResult({
  result,
  loading,
}: {
  result: PredictionResponse | null;
  loading: boolean;
}) {
  // Legacy servers may return one genuine prediction. Never invent extra ranks.
  const predictions =
    result?.top_predictions ??
    (result
      ? [
          {
            category: result.predicted_category,
            probability: result.confidence,
          },
        ]
      : []);
  return (
    <section
      aria-labelledby="result-heading"
      className="flex h-full flex-col rounded-2xl border border-line bg-white shadow-card"
    >
      <div className="flex items-center justify-between border-b border-line px-5 py-5 sm:px-6">
        <div className="flex items-center gap-2.5">
          <span className="step-label">02</span>
          <h2
            id="result-heading"
            className="text-base font-semibold tracking-tight"
          >
            Model prediction
          </h2>
        </div>
        <Icon name="spark" className="h-4 w-4 text-muted" />
      </div>
      <div
        aria-live="polite"
        aria-busy={loading}
        className="flex flex-1 flex-col p-6"
      >
        {loading ? (
          <div
            role="status"
            className="flex min-h-64 flex-1 flex-col items-center justify-center text-center"
          >
            <span className="mb-5 flex h-16 w-16 items-center justify-center rounded-2xl bg-emerald-50 text-accent">
              <Spinner className="h-6 w-6" />
            </span>
            <h3 className="text-lg font-semibold">Reading the signals</h3>
            <p className="mt-2 max-w-64 text-sm leading-6 text-muted">
              Extracting features and comparing job categories with the trained
              model.
            </p>
          </div>
        ) : result ? (
          <div className="result-enter flex-1">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="inline-flex items-center gap-1.5 rounded-md bg-emerald-50 px-2 py-1 text-[11px] font-medium text-accent">
                <Icon name="check" className="h-3 w-3" />
                Prediction complete
              </span>
              {result.text_stats && (
                <span className="text-[11px] text-muted">
                  {result.text_stats.word_count.toLocaleString("en-US")} words ·{" "}
                  {result.source === "file" ? "CV file" : "Pasted text"}
                </span>
              )}
            </div>
            <p className="mt-5 text-xs text-muted">
              Top predicted job category
            </p>
            <h3 className="mt-2 break-words text-3xl font-semibold leading-tight tracking-tight">
              {categoryLabel(result.predicted_category)}
            </h3>
            {result.text_stats?.short_input && (
              <div
                role="note"
                className="mt-4 flex gap-2 rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs leading-5 text-amber-950"
              >
                <Icon name="info" className="mt-0.5 h-4 w-4 shrink-0" />
                <p>
                  <strong>Limited context.</strong> This input has fewer than 50
                  words. Predictions on short text can be unreliable; include
                  more experience and skills.
                </p>
              </div>
            )}
            <p className="mb-4 mt-6 text-xs font-medium text-muted">
              Model confidence (uncalibrated)
            </p>
            <div className="space-y-4">
              {predictions.map((prediction, index) => {
                const percent = prediction.probability * 100;
                return (
                  <div key={prediction.category}>
                    <div className="mb-2 flex items-start justify-between gap-4 text-sm">
                      <span className="min-w-0 break-words">
                        <span className="mr-2 text-[11px] text-muted">
                          0{index + 1}
                        </span>
                        {categoryLabel(prediction.category)}
                      </span>
                      <span className="shrink-0 font-semibold tabular-nums">
                        {percent.toFixed(1)}%
                      </span>
                    </div>
                    <div
                      role="progressbar"
                      aria-label={`${categoryLabel(prediction.category)} model confidence (uncalibrated)`}
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-valuenow={percent}
                      className="h-2 overflow-hidden rounded-full bg-[#eaf0eb]"
                    >
                      <div
                        style={{ width: `${percent}%` }}
                        className={`h-full rounded-full transition-[width] duration-500 motion-reduce:transition-none ${index === 0 ? "bg-accent" : index === 1 ? "bg-[#73a38c]" : "bg-[#a3bfac]"}`}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
            <p className="mt-3 text-[11px] leading-5 text-muted">
              Probabilities across all job categories; the top three are not
              rescaled to total 100%.
            </p>
            {result.top_terms ? (
              <div className="mt-6 border-t border-line pt-5">
                <h4 className="text-sm font-semibold">Influential terms</h4>
                <p className="mt-1 text-xs leading-5 text-muted">
                  Terms in your CV with high TF-IDF × global model importance.
                </p>
                <div className="mt-3 flex flex-wrap gap-2">
                  {result.top_terms.map((term) => (
                    <span
                      key={term}
                      className="rounded-md border border-[#dce8d9] bg-[#f1f5ee] px-2.5 py-1 text-xs text-accent"
                    >
                      {term}
                    </span>
                  ))}
                </div>
                {!result.top_terms.length && (
                  <p className="mt-2 text-xs text-muted">
                    No positively weighted terms were found.
                  </p>
                )}
                <p className="mt-3 text-[11px] leading-5 text-muted">
                  Global feature relevance, not class-specific reasons or a
                  causal explanation. Terms may be normalized words or two-word
                  phrases.
                </p>
              </div>
            ) : (
              <p className="mt-5 rounded-lg bg-amber-50 p-3 text-xs text-amber-950">
                This server returned the original response format. Restart the
                updated backend to see all prediction details.
              </p>
            )}
            {result.source === "file" &&
              typeof result.text_preview === "string" && (
                <details className="mt-5 rounded-lg border border-line bg-canvas p-3">
                  <summary className="cursor-pointer rounded text-xs font-semibold focus-visible:outline-accent">
                    Extracted text preview
                  </summary>
                  <p className="mt-2 text-[11px] text-muted">
                    First 500 characters, before preprocessing.
                  </p>
                  <pre className="mt-3 max-h-48 overflow-auto whitespace-pre-wrap break-words font-sans text-xs leading-6">
                    {result.text_preview}
                  </pre>
                </details>
              )}
          </div>
        ) : (
          <div className="flex min-h-64 flex-1 flex-col items-center justify-center py-6 text-center">
            <span className="mb-5 flex h-20 w-20 items-center justify-center rounded-2xl border border-[#dfe8de] bg-[#f1f5ee]">
              <Icon name="document" className="h-9 w-9 text-[#678565]" />
            </span>
            <h3 className="text-lg font-semibold tracking-tight">
              Your CV, a little more context
            </h3>
            <p className="mt-2 max-w-64 text-sm leading-6 text-muted">
              Upload a CV or paste text to explore its top categories and
              influential terms.
            </p>
            <span className="mt-5 rounded-full border border-dashed border-line px-3 py-1 text-[11px] text-muted">
              Waiting for your first resume
            </span>
          </div>
        )}
        <div className="mt-7 flex items-start gap-2.5 rounded-xl bg-canvas p-4">
          <Icon name="info" className="mt-0.5 h-4 w-4 shrink-0 text-muted" />
          <div>
            <h4 className="text-xs font-semibold">
              A prediction, with perspective
            </h4>
            <p className="mt-1.5 text-xs leading-5 text-muted">
              Confidence reflects the model's output, not a guarantee of
              correctness or a measure of candidate suitability.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
