import { useRef, useState } from "react";
import type { DragEvent, FormEvent } from "react";
import { validateFile } from "../lib/api";
import { Icon, Spinner } from "./Icon";

interface Props {
  file: File | null;
  loading: boolean;
  onFile: (file: File | null) => void;
  onError: (message: string) => void;
  onSubmit: () => void;
}

export function FileUpload({
  file,
  loading,
  onFile,
  onError,
  onSubmit,
}: Props) {
  const input = useRef<HTMLInputElement>(null);
  const dragDepth = useRef(0);
  const [dragging, setDragging] = useState(false);
  function select(files: FileList | null) {
    if (loading || !files?.length) return;
    onFile(null);
    if (files.length !== 1) {
      onError("Please upload one CV at a time.");
      return;
    }
    const candidate = files[0];
    const error = validateFile(candidate);
    if (error) {
      onError(error);
      return;
    }
    onFile(candidate);
  }
  function drop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    dragDepth.current = 0;
    setDragging(false);
    select(event.dataTransfer.files);
  }
  function submit(event: FormEvent) {
    event.preventDefault();
    if (file && !loading) onSubmit();
  }
  return (
    <form
      onSubmit={submit}
      aria-busy={loading}
      className="flex flex-1 flex-col px-5 pb-6 sm:px-7"
    >
      <div
        onDragEnter={(e) => {
          e.preventDefault();
          dragDepth.current += 1;
          if (!loading) setDragging(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          dragDepth.current -= 1;
          if (dragDepth.current <= 0) setDragging(false);
        }}
        onDragOver={(e) => e.preventDefault()}
        onDrop={drop}
        className={`flex min-h-64 flex-col items-center justify-center rounded-xl border-2 border-dashed px-5 py-8 text-center transition ${dragging ? "border-accent bg-emerald-50" : "border-line bg-[#fcfdfb]"}`}
      >
        <span className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-[#edf3e9] text-accent">
          <Icon name="document" className="h-7 w-7" />
        </span>
        <h3 className="font-semibold">
          {dragging ? "Drop your CV here" : "Your experience starts here"}
        </h3>
        <p className="mt-2 text-sm text-muted">
          Drag and drop a CV, or browse your files.
        </p>
        <p id="file-requirements" className="mt-2 text-xs text-muted">
          PDF, DOCX, or TXT · Up to 5 MB · No scanned PDFs
        </p>
        <input
          ref={input}
          type="file"
          accept=".pdf,.docx,.txt"
          aria-label="Choose CV file"
          aria-describedby="file-requirements"
          className="sr-only"
          tabIndex={-1}
          disabled={loading}
          onChange={(e) => {
            select(e.target.files);
            e.target.value = "";
          }}
        />
        <button
          type="button"
          disabled={loading}
          onClick={() => input.current?.click()}
          className="mt-5 rounded-lg border border-line bg-white px-4 py-2.5 text-sm font-medium shadow-sm hover:border-accent disabled:cursor-wait disabled:opacity-50"
        >
          Browse file
        </button>
      </div>
      {file && (
        <div
          className="mt-4 flex items-center gap-3 rounded-xl border border-line bg-canvas p-3"
          aria-live="polite"
        >
          <Icon name="document" className="h-5 w-5 shrink-0 text-accent" />
          <div className="min-w-0 flex-1">
            <p className="break-all text-sm font-medium">{file.name}</p>
            <p className="mt-1 text-xs text-muted">
              {(file.size / 1024).toFixed(1)} KB · Ready to classify
            </p>
          </div>
          <button
            type="button"
            disabled={loading}
            onClick={() => onFile(null)}
            aria-label="Remove selected file"
            className="rounded-md p-2 text-muted hover:bg-white disabled:opacity-50"
          >
            <Icon name="close" className="h-4 w-4" />
          </button>
        </div>
      )}
      <p className="mt-4 text-xs leading-5 text-muted">
        Your file is processed in memory and not stored.
      </p>
      <p className="mt-1 text-[11px] leading-5 text-muted">
        After classification, you can preview the text the model read.
      </p>
      <button
        type="submit"
        disabled={!file || loading}
        className="mt-6 flex items-center justify-center gap-3 rounded-lg bg-accent px-5 py-3 text-sm font-semibold text-white transition hover:bg-[#1b5541] disabled:cursor-not-allowed disabled:bg-[#dae3de] disabled:text-[#69786e]"
      >
        {loading ? (
          <>
            <Spinner />
            Reading and classifying…
          </>
        ) : (
          <>
            Classify CV
            <Icon name="arrow" className="h-4 w-4" />
          </>
        )}
      </button>
    </form>
  );
}
