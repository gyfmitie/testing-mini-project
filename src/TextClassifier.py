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
        """
        Cleans and normalizes a single text string.
        - Lowercases text
        - Removes non-alphanumeric characters (keeping spaces)
        """
    #   logging.debug(f"Preprocessing text: '{text}'")
        text = text.lower()
        non_alphabetical_characters = r"[^a-z\s]"
        text = re.sub(non_alphabetical_characters, "", text)
        text = " ".join(text.split())
    #   logging.debug(f"Preprocessed text: '{text}'")
        return text

    def train(self, texts: list[str], labels: list[str]):
        """
        Trains the classification model.
        """
        logging.info("Starting model training.")
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in texts]

        # Fit vectorizer and transform texts
        X = self.vectorizer.fit_transform(processed_texts)
        y = labels

        # Train the model
        self.model.fit(X, y)
        logging.info("Model training completed.")

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
