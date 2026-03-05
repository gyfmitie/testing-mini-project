# Testing Mini Project

This project demonstrates a basic Machine Learning pipeline for text classification, specifically designed to categorise customer reviews as 'positive' or 'negative'. It showcases a common project structure, automated testing (unit, regression, and integration), and a simple application runner.

> You run an e-commerce platform filled with product reviews. Your `TextClassifier` is your smart helper, enabling you to tell positive from negative reviews. To ensure this smart helper can adapt and respond to new features and changes, you introduce automated testing into your project.

## Getting Started

### Option A: Dev Container (recommended)

1. Open the project in VS Code.
1. Run `Dev Containers: Reopen in Container`.
1. Wait for container setup to finish.

Dependencies are installed automatically from `requirements.txt` during container setup.

Run the app:

```sh
python app.py
```

Run tests:

```sh
pytest -v
```

### Option B: Local Environment

1. Open the project in your IDE and open the integrated terminal.

1. Create a virtual environment and then activate it.

   - Create with `venv`

   ```sh
    # Create
    python -m venv venv
    # Activate with Windows
    .\venv\Scripts\activate
    # Activate with macOS/Linux
    source venv/bin/activate
   ```

   - Create with `conda`

   ```sh
   # Create
   conda create -n text_classifier_env python=3.8 # Or your preferred Python version
   conda activate text_classifier_env
   ```

1. Install Dependencies

   ```sh
   pip install -r requirements.txt
   ```

1. Run the Application (This demo will train the classifier on the provided CSV data and output predictions and evaluation results.)

   ```sh
   python app.py
   ```

1. Run Tests (-v flagProvides verbose output, showing individual test results.)

   ```sh
   # Run all
   pytest -v
   # Run specific
   pytest -v tests/test_TextClassifier_unit.py
   ```

## Project Structure

```
testing-mini-project/
├── data/
│   └── raw/
│       └── text-label.csv
├── src/
│   └── TextClassifier.py
├── tests/
│   └── conftest.py
│   └── test_TextClassifier_integration.py
│   └── test_TextClassifier_regression.py
│   └── test_TextClassifier_unit.py
├── app.py
├── requirements.txt
└── README.md
```

---
