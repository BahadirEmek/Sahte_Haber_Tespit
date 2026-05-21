"""Train and evaluate a TF-IDF + Logistic Regression fake news classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from preprocessing import TextPreprocessor, clean_text


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


_LABEL_FAKE = "Fake News"
_LABEL_REAL = "Real News"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
MODEL_PATH = MODEL_DIR / "fake_news_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = SCREENSHOTS_DIR / "confusion_matrix.png"


class FakeNewsModelTrainer:
    """Coordinate data loading, preprocessing, training, evaluation, and saving."""

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        model_path: Path = MODEL_PATH,
        metrics_path: Path = METRICS_PATH,
        confusion_matrix_path: Path = CONFUSION_MATRIX_PATH,
    ) -> None:
        """Initialize paths and reusable model components."""
        self.data_dir = data_dir
        self.model_path = model_path
        self.metrics_path = metrics_path
        self.confusion_matrix_path = confusion_matrix_path
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.data: pd.DataFrame | None = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.metrics: dict[str, Any] = {}

    def load_data(self) -> pd.DataFrame:
        """Load fake and real news CSV files with friendly validation errors."""
        fake_path = self.data_dir / "Fake.csv"
        true_path = self.data_dir / "True.csv"
        missing_files = [str(path) for path in (fake_path, true_path) if not path.exists()]

        if missing_files:
            raise FileNotFoundError(
                "Veri dosyasi eksik. Lutfen su dosyalari data klasorune ekleyin: "
                + ", ".join(missing_files)
            )

        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)
        self._validate_columns(fake_df, "Fake.csv")
        self._validate_columns(true_df, "True.csv")

        fake_df = fake_df.copy()
        true_df = true_df.copy()
        fake_df["label"] = 0
        true_df["label"] = 1

        self.data = pd.concat([fake_df, true_df], ignore_index=True)
        return self.data

    def prepare_dataset(self) -> pd.DataFrame:
        """Merge title and text columns, clean content, and keep model columns."""
        if self.data is None:
            self.load_data()

        assert self.data is not None
        prepared_data = self.data.copy()
        prepared_data["title"] = prepared_data["title"].fillna("")
        prepared_data["text"] = prepared_data["text"].fillna("")
        prepared_data["content"] = prepared_data["title"] + " " + prepared_data["text"]
        prepared_data["content"] = prepared_data["content"].apply(self.preprocessor.clean_text)
        prepared_data = prepared_data[prepared_data["content"].str.len() > 0].reset_index(drop=True)

        if prepared_data.empty:
            raise ValueError("Temizleme sonrasi egitim icin kullanilabilecek haber metni kalmadi.")

        self.data = prepared_data[["content", "label"]]
        return self.data

    def split_data(self) -> tuple[Any, Any, Any, Any]:
        """Split prepared data into training and test groups."""
        if self.data is None or not {"content", "label"}.issubset(self.data.columns):
            self.prepare_dataset()

        assert self.data is not None
        stratify = self.data["label"] if self.data["label"].value_counts().min() >= 2 else None
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.data["content"],
            self.data["label"],
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )
        return self.x_train, self.x_test, self.y_train, self.y_test

    def train(self) -> tuple[LogisticRegression, TfidfVectorizer]:
        """Fit the TF-IDF vectorizer and Logistic Regression classifier."""
        if self.x_train is None or self.y_train is None:
            self.split_data()

        x_train_vectorized = self.vectorizer.fit_transform(self.x_train)
        self.model.fit(x_train_vectorized, self.y_train)
        return self.model, self.vectorizer

    def evaluate(self) -> dict[str, Any]:
        """Evaluate the trained model and save metrics plus confusion matrix."""
        if self.x_test is None or self.y_test is None:
            self.split_data()

        x_test_vectorized = self.vectorizer.transform(self.x_test)
        y_pred = self.model.predict(x_test_vectorized)

        precision, recall, f1_score, _ = precision_recall_fscore_support(
            self.y_test,
            y_pred,
            average="binary",
            zero_division=0,
        )
        report_dict = classification_report(
            self.y_test,
            y_pred,
            target_names=[_LABEL_FAKE, _LABEL_REAL],
            output_dict=True,
            zero_division=0,
        )
        report_text = classification_report(
            self.y_test,
            y_pred,
            target_names=[_LABEL_FAKE, _LABEL_REAL],
            zero_division=0,
        )

        assert self.data is not None
        self.metrics = {
            "accuracy": round(float(accuracy_score(self.y_test, y_pred)), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1_score), 4),
            "classification_report": report_dict,
            "total_samples": int(len(self.data)),
            "train_samples": int(len(self.x_train)),
            "test_samples": int(len(self.x_test)),
            "model_name": "Logistic Regression",
            "vectorizer_name": "TF-IDF Vectorizer",
        }

        self._save_metrics()
        self._save_confusion_matrix(y_pred)
        self._print_evaluation_summary(report_text)
        return self.metrics

    def save_artifacts(self) -> None:
        """Save model and vectorizer into a single joblib artifact."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "vectorizer": self.vectorizer}, self.model_path)
        print(f"Model dosyasi kaydedildi: {self.model_path}")

    def run(self) -> None:
        """Run the complete training workflow."""
        print("Fake News model egitimi baslatildi.")
        self.load_data()
        self.prepare_dataset()
        self.split_data()
        self.train()
        self.evaluate()
        self.save_artifacts()
        print("Egitim tamamlandi.")

    def _validate_columns(self, dataframe: pd.DataFrame, file_name: str) -> None:
        """Validate required CSV columns before training."""
        required_columns = {"title", "text"}
        missing_columns = required_columns.difference(dataframe.columns)
        if missing_columns:
            raise ValueError(
                f"{file_name} dosyasinda eksik kolon var: {', '.join(sorted(missing_columns))}"
            )

    def _save_metrics(self) -> None:
        """Write evaluation metrics to models/metrics.json."""
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.write_text(
            json.dumps(self.metrics, indent=4, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Metrik dosyasi kaydedildi: {self.metrics_path}")

    def _save_confusion_matrix(self, y_pred) -> None:
        """Save the confusion matrix chart for reporting and the Streamlit UI."""
        self.confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
        matrix = confusion_matrix(self.y_test, y_pred, labels=[0, 1])
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=[_LABEL_FAKE, _LABEL_REAL],
        )
        display.plot(cmap="Blues", values_format="d")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(self.confusion_matrix_path, dpi=150)
        plt.close()
        print(f"Confusion matrix kaydedildi: {self.confusion_matrix_path}")

    def _print_evaluation_summary(self, report_text: str) -> None:
        """Print a readable terminal summary for presentations."""
        print("\nModel Performansi")
        print("-" * 40)
        print(f"Accuracy : {self.metrics['accuracy']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall   : {self.metrics['recall']:.4f}")
        print(f"F1-score : {self.metrics['f1_score']:.4f}")
        print(f"Ornek sayisi: {self.metrics['total_samples']}")
        print(f"Egitim/Test: {self.metrics['train_samples']} / {self.metrics['test_samples']}")
        print("\nClassification Report")
        print("-" * 40)
        print(report_text)


def main() -> None:
    """Run the class-based training pipeline."""
    trainer = FakeNewsModelTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
