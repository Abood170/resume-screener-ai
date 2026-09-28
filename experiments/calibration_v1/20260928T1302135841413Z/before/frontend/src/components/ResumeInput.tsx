import { useRef } from "react";
import type { FormEvent } from "react";
import { MAX_CHARACTERS } from "../lib/api";
import { Icon, Spinner } from "./Icon";

const examples = [
  {
    label: "Accounting",
    text: "Accountant with experience preparing financial statements, reconciling general ledger accounts, and supporting internal and external audits. Managed month-end close, accounts payable, payroll, budgeting, and tax reporting. Used Excel and accounting software to improve reporting accuracy and document financial controls.",
  },
  {
    label: "Software & IT",
    text: "Software engineer developing Python applications and SQL databases. Built REST APIs, wrote automated tests, and maintained Linux servers. Collaborated with product teams to troubleshoot production issues, review code, and deploy reliable services. Experience with network support, database administration, Git, and cloud infrastructure.",
  },
  {
    label: "Human resources",
    text: "Human resources specialist supporting employee onboarding, recruitment, benefits administration, and employee relations. Coordinated interviews, maintained HR records, and delivered staff training. Partnered with managers on performance reviews, payroll questions, workplace policies, and improving the employee experience.",
  },
];

interface Props {
  text: string;
  loading: boolean;
  onChange: (text: string) => void;
  onSubmit: () => void;
  embedded?: boolean;
}
export function ResumeInput({
  text,
  loading,
  onChange,
  onSubmit,
  embedded = false,
}: Props) {
  const input = useRef<HTMLTextAreaElement>(null);
  // Python counts Unicode code points; JS string.length counts UTF-16 code units.
  const count = Array.from(text).length;
  const tooLong = count > MAX_CHARACTERS;
  function submit(event: FormEvent) {
    event.preventDefault();
    if (text.trim() && !tooLong && !loading) onSubmit();
  }
  return (
    <form
      onSubmit={submit}
      className={
        embedded
          ? "flex flex-1 flex-col"
          : "flex h-full flex-col rounded-2xl border border-line bg-white shadow-card"
      }
      aria-busy={loading}
    >
      {!embedded && (
        <div className="border-b border-line px-5 py-5 sm:px-7">
          <div className="flex items-center gap-2.5">
            <span className="step-label">01</span>
            <h2 className="text-base font-semibold tracking-tight">
              Your resume
            </h2>
          </div>
          <p className="mt-2 text-sm text-muted">
            Paste your experience, skills, and education below.
          </p>
        </div>
      )}
      <div className="flex flex-1 flex-col px-5 pb-6 pt-5 sm:px-7">
        <div className="mb-4 flex flex-wrap items-center gap-2">
          <span className="mr-1 text-xs text-muted">Try an example</span>
          {examples.map((example) => (
            <button
              key={example.label}
              type="button"
              disabled={loading}
              onClick={() => {
                onChange(example.text);
                input.current?.focus();
              }}
              className="rounded-md border border-line bg-canvas px-2.5 py-1.5 text-xs font-medium text-ink transition hover:border-accent/40 hover:bg-emerald-50 disabled:cursor-wait disabled:opacity-50"
            >
              {example.label}
              <span aria-hidden="true" className="ml-1.5 text-muted">
                ↗
              </span>
            </button>
          ))}
        </div>
        <label htmlFor="resume-text" className="sr-only">
          Resume text
        </label>
        <textarea
          ref={input}
          id="resume-text"
          value={text}
          readOnly={loading}
          onChange={(event) => onChange(event.target.value)}
          aria-describedby="resume-counter resume-tip"
          aria-invalid={tooLong}
          placeholder={
            "Paste resume text here…\n\nInclude work experience, technical skills, projects, and education for more context."
          }
          className="min-h-64 w-full flex-1 resize-y rounded-xl border border-line bg-[#fcfdfb] p-4 text-sm leading-7 text-ink placeholder:text-[#89938e] focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/15 read-only:opacity-70"
        />
        <div
          id="resume-counter"
          className={`mt-2 flex flex-wrap items-center justify-between gap-2 text-[11px] ${tooLong ? "text-red-700" : "text-muted"}`}
        >
          <span>
            {tooLong
              ? "Please shorten your text to continue."
              : "Plain text only"}
          </span>
          <span className="tabular-nums">
            {count.toLocaleString("en-US")} /{" "}
            {MAX_CHARACTERS.toLocaleString("en-US")} characters
          </span>
        </div>
        <div className="mt-5 flex flex-wrap items-center justify-between gap-4">
          <button
            type="button"
            onClick={() => {
              onChange("");
              input.current?.focus();
            }}
            disabled={!text || loading}
            className="rounded-md py-2 text-xs font-medium text-muted hover:text-ink disabled:cursor-not-allowed disabled:opacity-40"
          >
            Clear text
          </button>
          <button
            type="submit"
            disabled={!text.trim() || tooLong || loading}
            className="flex min-w-44 items-center justify-center gap-3 rounded-lg bg-accent px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-[#1b5541] disabled:cursor-not-allowed disabled:bg-[#dae3de] disabled:text-[#69786e] disabled:shadow-none"
          >
            {loading ? (
              <>
                <Spinner />
                Classifying…
              </>
            ) : (
              <>
                Classify resume
                <Icon name="arrow" className="h-4 w-4" />
              </>
            )}
          </button>
        </div>
        <p
          id="resume-tip"
          className="mt-4 text-[11px] leading-relaxed text-muted"
        >
          Text is sent to the configured API. This interface doesn’t save your
          resume.
        </p>
      </div>
    </form>
  );
}
