# Resume Screener AI — frontend

A single-page React + TypeScript interface for the existing trained resume classifier. Upload a PDF/DOCX/TXT CV, paste resume text, or choose an illustrative example, call the real FastAPI backend, and inspect the returned category and **calibrated confidence** when supported by the backend. No generated predictions, offline fallback, external model API, analytics, or browser storage are used.

## Backend connection and CORS

The existing backend enables CORSMiddleware with `allow_origins=["*"]`; this extension leaves it unchanged. Restrict origins to the deployed frontend domain before production use. The frontend calls `http://127.0.0.1:8000` directly, without a Vite proxy. A CORS block and a network failure can both appear as API unreachable.

## Run locally

Install Node.js **22.12 or newer** (a supported LTS version is recommended). Python and the saved artifacts are supplied by the parent backend project. See the [backend README](../README.md) for its setup and NLTK resources.

Terminal one, from `resume-screener-ml/`, after installing the updated backend requirements:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Terminal two:

```bash
cd frontend
npm install
```

Copy `.env.example` to `.env` (PowerShell: `Copy-Item .env.example .env`; macOS/Linux: `cp .env.example .env`). It contains:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Then:

```bash
npm run dev
```

Open **http://127.0.0.1:5173**. If `npm` is blocked by PowerShell’s script policy, use `npm.cmd` instead. The default API URL also works without an `.env` file. Restart Vite after changing environment variables; these variables are bundled client-side, so never place secrets in them. The dev port is strict to avoid silently changing the required CORS origin.

This workspace was verified using a portable Node installation at `%LOCALAPPDATA%\resume-screener-tools\node-v22.23.3-win-x64`; no global PATH changes were made. If Node is not otherwise installed, this session-only PowerShell command makes `npm.cmd` available on this machine:

```powershell
$env:Path = "$env:LOCALAPPDATA\resume-screener-tools\node-v22.23.3-win-x64;" + $env:Path
```

For repeat installs use `npm ci`, which follows the included lockfile exactly.

## Demo flow

1. Confirm **API online** in the header.
2. Drag a PDF, DOCX or TXT onto **Upload CV**, or select **Browse file**. Files must be nonempty and at most 5 MiB. The backend independently checks their contents. Your file is processed in memory and not stored.
3. Alternatively select **Paste text** and enter a CV or choose an example chip.
4. Submit to see three real categories and **calibrated confidence** when supported by the backend. Display percentages are truncated to one decimal place without renormalization.
5. Inspect **Influential terms** and the short-input warning. Terms rank input TF-IDF multiplied by global Random Forest feature importance: class-agnostic salience, not causal explanations or hiring recommendations.
6. Expand the upload text preview to inspect the first 500 characters. Scanned PDFs are unsupported. Non-English CVs and short snippets may be unreliable.
7. **Clear / start over** resets file, text and results and cancels an in-flight request.

## API and state handling

- `GET /health` expects `{ "status": "ok", "model": "..." }`.
- `POST /predict/file` sends FormData with field `file`; the browser sets its multipart boundary. Both prediction endpoints return `top_predictions`, `top_terms`, `text_stats`, `source` and `text_preview` in addition to legacy fields. See the [actual response](../README.md#api-contract-and-real-example). Legacy text-only responses display an upgrade notice instead of invented details.
- `POST /predict` sends `{ "text": "..." }` and expects `{ "predicted_category": "...", "confidence": 0.0 }`, where the numeric value must be finite and between zero and one. This is a schema illustration, not a stored prediction.
- Types mirror `ResumeRequest`, `PredictionResponse`, Pydantic validation issues, and FastAPI string error details. Responses are also checked at runtime; TypeScript alone cannot validate network data.
- Empty/whitespace inputs are disabled. The 50,000-character limit counts Unicode code points like Python; oversized pasted text stays visible so the user can shorten it rather than being silently truncated.
- Validation errors explain the backend’s reason. Service failures and malformed responses get friendly messages. Server internals are never shown. Non-string inputs cannot be produced by a textarea, but Pydantic non-string errors are handled by the wrapper.
- Prediction requests time out after 30 seconds; health checks after 8 seconds. Polls do not overlap. Requests and timers are cleaned up on unmount, and cancelled requests cannot replace a newer result.
- A failed health poll does not disable submitting: users can retry immediately without waiting for the next poll. There is no mock prediction fallback.
- Accessible labels, live result/error announcements, keyboard focus styles, a skip link, reduced-motion support, and a labeled progress bar are included. Layout stacks at smaller viewports.

## Structure

```text
src/
  components/
    FileUpload.tsx        # drop/browse, validation, filename and remove
    ResumeInput.tsx       # input, examples, limit, submit
    PredictionResult.tsx  # empty/loading/result and confidence
    HealthBadge.tsx       # polling and retry
    ErrorBanner.tsx       # friendly dismissible failures
    Icon.tsx              # small inline SVGs, no icon library
  lib/
    api.ts                # typed fetch boundary, cancellation and errors
    api.test.ts           # request and failure-contract tests
  App.tsx                 # local state and request ownership
  main.tsx
  index.css
```

React and React DOM are the only runtime package dependencies. Vite, TypeScript, Tailwind and Vitest are development tools. Tailwind’s Vite plugin compiles utility classes; `index.css` explicitly loads `tailwind.config.js` using Tailwind’s `@config` directive. Fonts use the local system stack, with no external font requests. [Vite setup](https://vite.dev/guide/) · [Tailwind directives](https://tailwindcss.com/docs/functions-and-directives).

## Checks and production build

```bash
npm run typecheck
npm test
npm run build
npm run preview
```

The build goes into `dist/`. Preview uses **http://127.0.0.1:4173**, which must also be allowed by backend CORS. For a deployed build, set `VITE_API_BASE_URL` before building; localhost refers to each visitor’s own machine. Host `dist/` on a static web server and keep the existing FastAPI service separate.

The network tests use isolated test-only fetch fixtures for successful responses, Pydantic errors, backend failures, malformed responses, timeouts and cancellation. These fixtures are never imported by the app. See `VERIFICATION.md` for the completed verification record and its browser-upload automation limitation.

## Calibrated results and uncertainty

New responses retain all legacy fields and additionally provide `calibrated_predicted_category`, `calibrated_confidence`, `calibrated_top_predictions`, `is_uncertain` and `uncertainty_reason`. The UI uses the calibrated category and bars together; the legacy winner can differ. Unknown or inconsistent fields produce a friendly error, never invented calibration. Older backends are clearly labeled uncalibrated.

Uncertain predictions show a neutral banner, a plain-language reason and **Possible categories**. Certain predictions keep the category headline with **Calibrated confidence** and a keyboard-focusable explanatory tooltip. This means no uncertainty rule was triggered, not guaranteed correctness or candidate quality. Display percentages are truncated to one decimal place. The ethical footer is unchanged. See the [generated methodology and exact results](../README.md#calibration-and-abstention).
