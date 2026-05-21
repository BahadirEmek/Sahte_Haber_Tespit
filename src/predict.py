"""Prediction helpers for the fake news detection model."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from preprocessing import TextPreprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "fake_news_model.joblib"


class FakeNewsPredictor:
    """Load trained artifacts and predict whether a news text is fake or real."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        """Initialize predictor state without loading the model immediately."""
        self.model_path = model_path
        self.preprocessor = TextPreprocessor()
        self.model = None
        self.vectorizer = None

    def load_model(self) -> None:
        """Load the trained model and vectorizer from the joblib artifact."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                "Model dosyasi bulunamadi. Lutfen once `python src/train_model.py` komutu ile modeli egitin."
            )

        artifacts = joblib.load(self.model_path)
        if "model" not in artifacts or "vectorizer" not in artifacts:
            raise ValueError("Model dosyasi beklenen model/vectorizer bilgilerini icermiyor.")

        self.model = artifacts["model"]
        self.vectorizer = artifacts["vectorizer"]

    def clean_input(self, text: str) -> str:
        """Clean and validate user-provided news text."""
        cleaned_text = self.preprocessor.clean_text(text)
        if len(cleaned_text.split()) < 5:
            raise ValueError("Tahmin icin daha uzun ve anlamli bir haber metni girilmelidir.")

        return cleaned_text

    def predict(self, text: str) -> dict:
        """Return model prediction, confidence, cleaned text, and important terms."""
        if self.model is None or self.vectorizer is None:
            self.load_model()

        cleaned_text = self.clean_input(text)
        text_vectorized = self.vectorizer.transform([cleaned_text])
        prediction = int(self.model.predict(text_vectorized)[0])
        confidence = self._calculate_confidence(text_vectorized, prediction)

        return {
            "label": "Real News" if prediction == 1 else "Fake News",
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "cleaned_text": cleaned_text,
            "important_terms": self.get_important_terms(text_vectorized, prediction),
        }

    def get_important_terms(self, text_vectorized, prediction: int, top_n: int = 10) -> list[str]:
        """Estimate influential terms using TF-IDF values and model coefficients."""
        if not hasattr(self.model, "coef_") or not hasattr(self.vectorizer, "get_feature_names_out"):
            return []

        feature_names = self.vectorizer.get_feature_names_out()
        row = text_vectorized.toarray()[0]
        non_zero_indexes = np.flatnonzero(row)
        if non_zero_indexes.size == 0:
            return []

        coefficients = self.model.coef_[0]
        contributions = row[non_zero_indexes] * coefficients[non_zero_indexes]
        order = np.argsort(contributions)
        selected_indexes = order[-top_n:][::-1] if prediction == 1 else order[:top_n]

        return [str(feature_names[non_zero_indexes[index]]) for index in selected_indexes]

    def _calculate_confidence(self, text_vectorized, prediction: int) -> float:
        """Calculate prediction confidence as a percentage."""
        if not hasattr(self.model, "predict_proba"):
            return 0.0

        probabilities = self.model.predict_proba(text_vectorized)[0]
        class_index = list(self.model.classes_).index(prediction)
        return float(probabilities[class_index] * 100)


def predict_news(text: str) -> dict:
    """Predict whether a news text is fake or real using ``FakeNewsPredictor``."""
    predictor = FakeNewsPredictor()
    return predictor.predict(text)
