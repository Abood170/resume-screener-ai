import { useEffect, useRef, useState } from "react";
import { ApiError, predictResume, predictResumeFile } from "./lib/api";
import type { PredictionResponse } from "./lib/api";
import { HealthBadge } from "./components/HealthBadge";
import { ResumeInput } from "./components/ResumeInput";
import { PredictionResult } from "./components/PredictionResult";
import { ErrorBanner } from "./components/ErrorBanner";
import { Icon } from "./components/Icon";
import { FileUpload } from "./components/FileUpload";

export default function App() {
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<"file" | "text">("file");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const pending = useRef<AbortController | null>(null);
  useEffect(() => () => pending.current?.abort(), []);

  function resetResult() {
    // A prediction always belongs to the submitted text, never an edited draft.
    pending.current?.abort();
    pending.current = null;
    setLoading(false);
    setResult(null);
    setError(null);
  }
  function changeText(value: string) {
    resetResult();
    setText(value);
  }
  function changeFile(value: File | null) {
    resetResult();
    setFile(value);
  }
  function startOver() {
    resetResult();
    setText("");
    setFile(null);
  }
  function changeMode(value: "file" | "text") {
    resetResult();
    setMode(value);
  }
  async function submit() {
    if (pending.current || (mode === "text" ? !text.trim() : !file)) return;
    const controller = new AbortController();
    pending.current = controller;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const prediction =
        mode === "file" && file
          ? await predictResumeFile(file, controller.signal)
          : await predictResume({ text }, controller.signal);
      if (!controller.signal.aborted) setResult(prediction);
    } catch (cause) {
      if (!controller.signal.aborted)
        setError(
          cause instanceof ApiError
            ? cause.message
            : "Something went wrong. Please try again.",
        );
    } finally {
      if (pending.current === controller) {
        pending.current = null;
        setLoading(false);
      }
    }
  }
  return (
    <div className="min-h-screen bg-canvas text-ink">
      <a
        href="#resume-input"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-white focus:p-3"
      >
        Skip to resume input
      </a>
      <header className="border-b border-line bg-white/70">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4 px-5 py-5 sm:px-8">
          <div className="flex items-center gap-3">
            <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-ink text-white">
              <Icon name="document" className="h-5 w-5" />
            </span>
            <div>
              <p className="text-sm font-semibold tracking-tight">
                Resume Screener <span className="text-accent">AI</span>
              </p>
              <p className="mt-0.5 text-[11px] text-muted">
                Resume text. A predicted job category.
              </p>
            </div>
          </div>
          <HealthBadge />
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-5 pb-10 pt-10 sm:px-8 sm:pt-12">
        <div className="mb-8 flex flex-wrap items-end justify-between gap-6">
          <div>
            <p className="mb-3 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-accent">
              <span className="h-px w-6 bg-accent" />
              Machine learning, made tangible
            </p>
            <h1 className="text-3xl font-semibold leading-[1.16] tracking-[-0.04em] sm:text-[42px]">
              Find the category.
              <br />
              <span className="text-[#7a897c]">See the model at work.</span>
            </h1>
            <p className="mt-4 max-w-xl text-sm leading-6 text-muted">
              Explore how a trained model translates resume text into a job
              category.
              <br className="hidden sm:block" /> Upload a CV, paste your text,
              or start with an example.
            </p>
          </div>
          <div className="mb-1 flex items-center gap-2 rounded-full border border-line bg-white px-3 py-2 text-[11px] text-muted">
            <Icon name="spark" className="h-3.5 w-3.5 text-accent" />
            Trained locally. Served with FastAPI.
          </div>
        </div>
        {error && (
          <div className="mb-5">
            <ErrorBanner message={error} onDismiss={() => setError(null)} />
          </div>
        )}
        <div className="grid items-stretch gap-5 lg:grid-cols-[1.35fr_1fr]">
          <section
            id="resume-input"
            aria-labelledby="input-heading"
            className="flex flex-col rounded-2xl border border-line bg-white shadow-card"
          >
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-5 py-5 sm:px-7">
              <div className="flex items-center gap-2.5">
                <span className="step-label">01</span>
                <h2 id="input-heading" className="text-base font-semibold">
                  Your resume
                </h2>
              </div>
              <button
                type="button"
                onClick={startOver}
                className="rounded-md px-1 py-2 text-xs font-medium text-muted hover:text-accent"
              >
                Clear / start over
              </button>
            </div>
            <div
              role="tablist"
              aria-label="Resume input method"
              className="mx-5 mb-5 mt-5 flex rounded-lg bg-canvas p-1 sm:mx-7"
            >
              {(["file", "text"] as const).map((tab, index) => (
                <button
                  key={tab}
                  id={`tab-${tab}`}
                  role="tab"
                  type="button"
                  aria-selected={mode === tab}
                  aria-controls={`panel-${tab}`}
                  tabIndex={mode === tab ? 0 : -1}
                  disabled={loading}
                  onClick={() => changeMode(tab)}
                  onKeyDown={(event) => {
                    if (
                      ["ArrowLeft", "ArrowRight", "Home", "End"].includes(
                        event.key,
                      )
                    ) {
                      event.preventDefault();
                      const next =
                        event.key === "Home"
                          ? "file"
                          : event.key === "End"
                            ? "text"
                            : index === 0
                              ? "text"
                              : "file";
                      changeMode(next);
                      document.getElementById(`tab-${next}`)?.focus();
                    }
                  }}
                  className={`flex-1 rounded-md px-3 py-2 text-sm font-medium transition ${mode === tab ? "bg-white text-accent shadow-sm" : "text-muted hover:text-ink"} disabled:opacity-50`}
                >
                  {tab === "file" ? "Upload CV" : "Paste text"}
                </button>
              ))}
            </div>
            <div
              id={`panel-${mode}`}
              role="tabpanel"
              aria-labelledby={`tab-${mode}`}
              className="flex flex-1 flex-col"
            >
              {mode === "file" ? (
                <FileUpload
                  file={file}
                  onFile={changeFile}
                  onError={setError}
                  onSubmit={() => void submit()}
                  loading={loading}
                />
              ) : (
                <ResumeInput
                  embedded
                  text={text}
                  onChange={changeText}
                  onSubmit={() => void submit()}
                  loading={loading}
                />
              )}
            </div>
          </section>
          <PredictionResult result={result} loading={loading} />
        </div>
        <div className="mt-6 flex flex-wrap items-center justify-between gap-3 text-[11px] text-muted">
          <p>Text → TF-IDF features → trained classifier → category</p>
          <p>No generative AI or external model API.</p>
        </div>
      </main>
      <footer className="mx-auto max-w-6xl px-5 pb-7 sm:px-8">
        <p className="border-t border-line pt-5 text-center text-[11px] leading-5 text-muted">
          Built with scikit-learn + FastAPI + React. Demonstrates an ML
          classification workflow, not a hiring decision tool.
        </p>
      </footer>
    </div>
  );
}
