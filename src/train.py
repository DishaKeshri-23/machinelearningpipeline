
import os
import pickle
import importlib
from urllib.parse import urlparse

import pandas as pd
import yaml
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Import infer_signature
infer_signature = importlib.import_module(
    "mlflow.models"
).infer_signature


# -------------------------------
# 1. Load parameters
# -------------------------------
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)["train"]


# -------------------------------
# 2. Hyperparameter tuning
# -------------------------------
def hyperparameter_tuning(X_train, y_train, param_grid, random_state):
    rf = RandomForestClassifier(random_state=random_state)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2,
        scoring="accuracy"
    )

    grid_search.fit(X_train, y_train)
    return grid_search


# -------------------------------
# 3. Train the model
# -------------------------------
def train(data_path, model_path, random_state):

    # Load dataset
    columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

    data = pd.read_csv(data_path, header=None, names=columns)
    ##print("CSV columns:", data.columns.tolist())
    ##print(data.head())
    # Separate features and target
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    # Configure MLflow tracking
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        raise ValueError(
            "Please set the MLFLOW_TRACKING_URI environment variable."
        )

    mlflow.set_tracking_uri(tracking_uri)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state
    )

    # Define hyperparameter grid
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    # Start MLflow run
    with mlflow.start_run():

        # Perform hyperparameter tuning
        grid_search = hyperparameter_tuning(
            X_train, y_train, param_grid, random_state
        )

        # Get the best model
        best_model = grid_search.best_estimator_

        print("Best parameters:", grid_search.best_params_)

        # Predict test results
        y_pred = best_model.predict(X_test)

        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        print("Confusion Matrix:")
        print(cm)

        print("Classification Report:")
        print(cr)

        # Log parameters and metrics
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", accuracy)

        # Log reports as artifacts
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(cr, "classification_report.txt")

        # Create model signature using actual predictions
        signature = infer_signature(
            X_train, best_model.predict(X_train)
        )

        # Log model to MLflow
       
        mlflow.sklearn.log_model(
    sk_model=best_model,
    name="model",
    signature=signature,
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)

        # Save model locally
        output_dir = os.path.dirname(model_path) or "."
        os.makedirs(output_dir, exist_ok=True)

        with open(model_path, "wb") as file:
            pickle.dump(best_model, file)

        print(f"Model saved to: {model_path}")

        return best_model


# -------------------------------
# 4. Run the training script
# -------------------------------
if __name__ == "__main__":
    train(
        data_path=params["data_path"],
        model_path=params["model_path"],
        random_state=params.get("random_state", 42)
    )
