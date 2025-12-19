from email.mime import text
import unicodedata
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
import logging

# Configure basic logging for demonstration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TextClassifier:
    """
    A simple text classification pipeline.
    """

    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = LogisticRegression(max_iter=1000)
        logging.info("TextClassifier initialized.")

    def preprocess_text(self, text: str) -> str:
        """Cleans and normalizes a single text string.
        - Removes HTML tags
        - Removes URLs
        - Normalizes accented characters to ASCII
        - Lowercases text
        - Removes non-alphabetic characters (keeping spaces)
        - Collapses multiple spaces
        """
        import unicodedata

        logging.debug(f"Preprocessing text: '{text}'")

        # Step 1: Remove HTML tags (before lowercasing to handle <P> vs <p>)
        text = re.sub(r"<[^>]+>", "", text)

        # Step 2: Remove URLs (http, https, www)
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # Step 3: Lowercase
        text = text.lower()

        # Step 4: Normalize accented characters to ASCII (é → e, ñ → n)
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("utf-8")

        # Step 5: Remove non-alphabetical characters (keeping spaces)
        non_alphabetical_characters = r"[^a-z\s]"
        text = re.sub(non_alphabetical_characters, "", text)

        # Step 6: Collapse multiple spaces and trim
        text = " ".join(text.split())
        logging.debug(f"Preprocessed text: '{text}'")
        return text

    def train(self, texts: list[str], labels: list[str]):
        """Trains the classification model with MLflow experiment tracking."""
        logging.info("Starting model training.")

        # Preprocess and vectorize
        processed_texts = [self.preprocess_text(text) for text in texts]
        X = self.vectorizer.fit_transform(processed_texts)
        y = labels

        # Train the model
        self.model.fit(X, y)
        # Calculate training accuracy
        train_accuracy = self.model.score(X, y)

        # ---- MLflow experiment tracking ----
        try:
            import mlflow
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment("text-classifier")

            with mlflow.start_run():
                # Log hyperparameters (what we configured)
                mlflow.log_param("max_iter", self.model.max_iter)
                mlflow.log_param("model_type", type(self.model).__name__)
                mlflow.log_param("vectorizer_type", type(self.vectorizer).__name__)
            # Log metrics (what resulted)
            mlflow.log_metric("train_samples", len(texts))
            mlflow.log_metric("vocab_size", len(self.vectorizer.vocabulary_))
            mlflow.log_metric("train_accuracy", train_accuracy)

        except Exception as e:
            logging.info(f"MLflow logging skipped: {e}")

        logging.info(f"Model training completed. Training accuracy: {train_accuracy:.4f}")

    def predict(self, texts: list[str]) -> list[str]:
        """
        Makes predictions on new text data.
        """
        logging.info("Starting prediction.")
        # Preprocess new texts
        processed_texts = [self.preprocess_text(text) for text in texts]

        # Transform texts using the fitted vectorizer
        X_new = self.vectorizer.transform(processed_texts)

        # Make predictions
        predictions = self.model.predict(X_new).tolist()
        logging.info("Prediction completed.")
        return predictions

    def evaluate(self, texts: list[str], true_labels: list[str]) -> float:
        """
        Evaluates the model's accuracy.
        """
        logging.info("Starting model evaluation.")
        predictions = self.predict(texts)
        score = accuracy_score(true_labels, predictions)
        logging.info(f"Model accuracy: {score:.4f}")
        return score
