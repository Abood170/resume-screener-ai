"""Load trusted local artifacts once; no API-based model is used."""

import json
from pathlib import Path
import joblib
import numpy as np
from src.data import ROOT, sha256
from src.preprocess import resources


class InvalidResumeError(ValueError):
    """Input has no usable model features."""


class Predictor:
    def __init__(self, model_dir: Path | None = None) -> None:
        folder = Path(model_dir) if model_dir is not None else ROOT / "models"
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        for name in ("model.joblib", "vectorizer.joblib"):
            if sha256(folder / name) != manifest["artifacts"][name]:
                raise ValueError(f"Artifact checksum mismatch: {name}")
        # joblib uses pickle: only load artifacts from a trusted training run.
        self.model = joblib.load(folder / "model.joblib")
        self.vectorizer = joblib.load(folder / "vectorizer.joblib")
        if self.model.classes_.tolist() != manifest["classes"]:
            raise ValueError("Model class manifest mismatch")
        if self.model.n_features_in_ != len(self.vectorizer.vocabulary_):
            raise ValueError("Model and vectorizer feature counts differ")
        resources()
        self.model_name: str = manifest["selected_model"]

    def predict(self, text: str) -> dict:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            raise InvalidResumeError("text must not be empty or whitespace")
        if len(text) > 50000:
            raise InvalidResumeError("text must contain at most 50000 characters")
        features = self.vectorizer.transform([text])
        if features.nnz == 0:
            raise InvalidResumeError("text contains no recognized resume vocabulary")
        probabilities = self.model.predict_proba(features)[0]
        index = int(np.argmax(probabilities))
        ranked_classes = np.argsort(-probabilities, kind="stable")[:3]
        # TF-IDF(input feature) * Random Forest global impurity importance.
        # Only features present in this CV are considered. This is a global,
        # class-agnostic salience heuristic, NOT a signed per-class contribution
        # or a causal explanation. Bigrams and noun lemmas match the vectorizer.
        importance = self.model.feature_importances_
        weighted = features.data * importance[features.indices]
        ranked_terms = np.argsort(-weighted, kind="stable")
        names = self.vectorizer.get_feature_names_out()
        top_terms = [
            str(names[features.indices[i]]) for i in ranked_terms if weighted[i] > 0
        ][:8]
        word_count = len(text.split())
        return {
            "predicted_category": str(self.model.classes_[index]),
            "confidence": float(probabilities[index]),
            "top_predictions": [
                {
                    "category": str(self.model.classes_[i]),
                    "probability": float(probabilities[i]),
                }
                for i in ranked_classes
            ],
            "top_terms": top_terms,
            "text_stats": {"word_count": word_count, "short_input": word_count < 50},
            "source": "text",
            "text_preview": None,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify resume text with local artifacts"
    )
    parser.add_argument("text")
    args = parser.parse_args()
    print(json.dumps(Predictor().predict(args.text), indent=2))
