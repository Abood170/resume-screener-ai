# Deploy the backend to Hugging Face Spaces

Deployment files only; the frontend, model artifacts, training/inference code, routes, and existing tests are unchanged. The image includes local `models/` and `data/resumes.csv`; there is no model download at startup.

## Configuration

- Docker base: `python:3.12-slim`.
- Startup: Uvicorn binds `0.0.0.0`, using `PORT` with fallback `7860`.
- Spaces reads `sdk: docker` and `app_port: 7860` from **root `README.md`**. `README_HF.md` alone does not configure a Space. The existing README content has been preserved below the deployment additions.
- Keep `PORT=7860` on Spaces, or change both `PORT` and the README `app_port` together.
- NLTK stopwords, WordNet and OMW are downloaded during image build to a directory readable by UID 1000. No startup downloads or training commands run.
- The image runs as non-root user `user`, UID 1000. A build-time check loads the actual model and vectorizer and validates their checksums and corpus availability.
- Existing CORS allows all origins, credentials, methods and headers. The existing TODO to restrict origins is retained. The frontend currently omits credentials. Before production use, restrict allowed origins to your actual frontend domain; credentialed browser requests require explicit origins.
- `.dockerignore` excludes the frontend, virtual environment, notebook/test folders and caches. It retains the model files and CSV. Docker exclusions do not control Git uploads; use the explicit staging list below.

## Optional local container check

Install Docker Desktop with Linux containers, then from the backend root:

```powershell
docker build -t resume-screener-api .
docker run --rm -p 7860:7860 -e PORT=7860 resume-screener-api
```

In another terminal:

```powershell
curl.exe --fail http://127.0.0.1:7860/health
```

To verify startup and prediction do not require network access, run the image with `--network none` and query loopback from inside it:

```powershell
docker run --rm --name resume-screener-offline --network none resume-screener-api
```

In a second terminal, after startup completes:

```powershell
docker exec resume-screener-offline python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:7860/health').read().decode())"
```

## Create the account and Space manually

1. Sign up at <https://huggingface.co/join> if needed, and verify your email.
2. Open <https://huggingface.co/new-space>.
3. Choose your account as owner; name the Space `resume-screener-ai`.
4. Select **Docker**, the **Blank** template if prompted, **Public**, and CPU Basic hardware. Create the Space.
5. Under account **Settings → Access Tokens**, create a write-capable token for this Space. Use it as the Git password when prompted; never embed it in a remote URL or commit it.
6. Install [Git for Windows](https://git-scm.com/downloads/win) and [Git LFS](https://git-lfs.com/), then reopen PowerShell. Confirm `git --version` and `git lfs version` work. Docker is only needed locally if you want to run the optional container checks; Spaces builds the Docker image remotely.

## Connect this folder and push

The following commands are for this current folder, which was not a Git repository at preparation time, and a **new Space containing only its initial scaffold**. Replace `YOUR_USERNAME` with your Hugging Face username. The remote is named `hf`.

```powershell
cd "C:\Users\Abood\Desktop\Models 2\resume-screener-ml"
git init -b main
git lfs install
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/resume-screener-ai
git add .gitattributes .gitignore .dockerignore Dockerfile requirements.txt README.md README_HF.md DEPLOY_HF.md api src models data/resumes.csv data/provenance.json results.json reports/figures reports/eda.json reports/split.json
git status --short
git lfs ls-files
git commit -m "Prepare Resume Screener AI backend for Docker Spaces"
git fetch hf main
git merge --strategy=ours --allow-unrelated-histories hf/main -m "Join initial Space history while keeping the prepared backend"
git push -u hf main
```

Check `git status` before committing: the staged files must include `models/model.joblib`, `models/vectorizer.joblib`, `models/manifest.json` and the CSV, and must exclude `frontend/`, `.venv/` and `data/source.zip`. `git lfs ls-files` should list both joblib files and the CSV. The `.gitattributes` rules are present before the first add so large files enter Git as LFS objects from the start. Hugging Face materializes them before Docker build; the running container reads local files only.

If Git asks for identity, set `git config user.name "YOUR_NAME"` and `git config user.email "YOUR_EMAIL"`, then repeat the commit. The `ours` merge above preserves the prepared local tree and joins the new Space's initial history without force-pushing. **Do not use it on a Space containing existing work you want to keep**; merge that repository normally instead. For an already-initialized local repository, skip `git init` and inspect existing remotes/history before applying these first-push steps.

## Verify the public deployment

1. Open the Space page and inspect **Build logs**. Dependency installation, NLTK downloads and `Local model artifacts and NLTK corpora verified` must complete successfully.
2. In runtime logs, confirm Uvicorn listens on `0.0.0.0:7860`, and the Space becomes **Running**. A 503 `/health` response means startup did not load the model successfully; inspect its startup traceback.
3. Use the Space's **Open in new tab** / direct app URL to copy its exact `https://...hf.space` hostname. Do not assume punctuation in a username maps directly to the hostname.
4. In PowerShell:

```powershell
$spaceUrl = "https://YOUR-EXACT-SPACE-HOST.hf.space"
Invoke-RestMethod "$spaceUrl/health"
```

Expected fields: `status` is `ok` and `model` is `random_forest`. Open `$spaceUrl/docs` in a browser for the existing interactive API. There is no root `/` route, so the Space's embedded root page may show FastAPI's `Not Found`; this is expected for this API-only deployment.

An optional real prediction request:

```powershell
Invoke-RestMethod -Method Post -Uri "$spaceUrl/predict" -ContentType "application/json" -Body '{"text":"Accountant managing audits, financial statements and tax reporting"}'
```

Finally replace the Live Demo placeholder in `README.md`, commit it, and push to `hf main`. No account, token, remote, or Space was created automatically.

## Preparation validation and limits

Docker and Git were not available in the preparation environment, so no local Docker build or git push is claimed. The Dockerfile performs its own real artifact/corpus check during the eventual build. The existing backend test suite passed: 35 tests, with one dependency deprecation warning. The unchanged requirements resolved for Python 3.12 on Linux x86-64 using `uv pip compile`. Model and dataset checksums matched the existing manifest, README metadata matched across both files, and wildcard CORS was confirmed. Dependency resolution is not a substitute for installing and running the image; the first actual container build remains to be verified. Installation depends on package registries and corpus hosting during image build.

## Official references

- [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker)
- [Spaces README configuration](https://huggingface.co/docs/hub/spaces-config-reference)
- [Spaces Git and large-file handling](https://huggingface.co/docs/hub/spaces-github-actions)
- [Access tokens](https://huggingface.co/docs/hub/security-tokens)
