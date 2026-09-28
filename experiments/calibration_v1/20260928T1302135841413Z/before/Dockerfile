FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/usr/local/share/nltk_data \
    HOME=/home/user \
    PORT=7860

WORKDIR /app

# Cache dependency installation separately from application/model changes.
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

# Fail the build if a corpus cannot be downloaded. Inference stays offline.
RUN python -c "import nltk; [nltk.download(name, download_dir='/usr/local/share/nltk_data', raise_on_error=True) for name in ('stopwords', 'wordnet', 'omw-1.4')]" \
    && chmod -R a+rX /usr/local/share/nltk_data

# Spaces uses UID 1000. Give the application a writable home and workdir.
RUN useradd --create-home --uid 1000 user \
    && chown user:user /app
COPY --chown=user:user . .
USER user

# Verify the actual included artifacts and NLTK resources as the serving user.
# This also catches unhydrated Git LFS pointers before deployment reaches runtime.
RUN python -c "from src.predict import Predictor; Predictor(); print('Local model artifacts and NLTK corpora verified')"

EXPOSE 7860

# Shell expansion reads PORT; exec forwards shutdown signals to Uvicorn.
CMD ["sh", "-c", "exec python -m uvicorn api.main:app --host 0.0.0.0 --port \"${PORT:-7860}\""]
