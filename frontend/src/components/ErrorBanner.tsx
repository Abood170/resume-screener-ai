import { Icon } from "./Icon";

export function ErrorBanner({
  message,
  onDismiss,
}: {
  message: string;
  onDismiss: () => void;
}) {
  return (
    <div
      role="alert"
      className="flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-900"
    >
      <Icon name="info" className="mt-0.5 h-4 w-4 shrink-0" />
      <div className="min-w-0 flex-1">
        <p className="font-semibold">We couldn’t classify this resume</p>
        <p className="mt-1 break-words leading-relaxed">{message}</p>
      </div>
      <button
        type="button"
        aria-label="Dismiss error"
        onClick={onDismiss}
        className="rounded p-1 hover:bg-red-100"
      >
        <Icon name="close" className="h-4 w-4" />
      </button>
    </div>
  );
}
