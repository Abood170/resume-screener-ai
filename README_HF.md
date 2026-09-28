---
title: Resume Screener AI
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

Resume Screener AI classifies resume text into job categories.
It serves the existing locally trained scikit-learn model through FastAPI.
The trained model and vectorizer are included in this repository.
NLTK corpora are installed during the Docker build, not at runtime.
Use `/health` to check readiness and `/docs` to try the API.
Send resume text to `POST /predict` to receive a category and uncalibrated confidence.
The API demonstrates classification, not candidate ranking or hiring decisions.
See [README.md](README.md) for methodology, measured results, and limitations.
See [DEPLOY_HF.md](DEPLOY_HF.md) for manual deployment steps.
