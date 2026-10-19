import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

HEART_DISEASE_UCIREPO_ID = 45


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Funkcja aktywacji sigmoidalna z zabezpieczeniem przed overflow."""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def compute_loss(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    """Oblicza funkcję kosztu – entropię krzyżową (Binary Cross-Entropy)."""
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


class CustomLogisticRegression:
    def __init__(
        self,
        alpha: float = 0.01,
        epochs: int = 1000,
        init_method: str = "zeros",
        random_state: int = 42,
    ):
        self.alpha = alpha
        self.epochs = epochs
        self.init_method = init_method
        self.random_state = random_state
        self.W: np.ndarray | None = None
        self.b: float = 0.0
        self.history = {
            "loss": [],
            "f1": [],
            "precision": [],
            "recall": [],
            "accuracy": [],
        }

    def _initialize_weights(self, n_features: int) -> None:
        rng = np.random.default_rng(self.random_state)
        if self.init_method == "zeros":
            self.W = np.zeros(n_features)
        elif self.init_method == "normal":
            self.W = rng.normal(loc=0.0, scale=0.01, size=n_features)
        elif self.init_method == "uniform":
            self.W = rng.uniform(low=-0.01, high=0.01, size=n_features)
        elif self.init_method == "large_normal":
            self.W = rng.normal(loc=0.0, scale=1.0, size=n_features)
        else:
            raise ValueError(f"Nieznana metoda inicjalizacji: {self.init_method}")
        self.b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> CustomLogisticRegression:
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)

        self.history = {
            "loss": [],
            "f1": [],
            "precision": [],
            "recall": [],
            "accuracy": [],
        }

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.W) + self.b
            p = sigmoid(linear_model)

            # Obliczenie funkcji kosztu
            loss = compute_loss(y, p)
            self.history["loss"].append(loss)

            # Predykcja i metryki dla bieżącej epoki
            y_pred = (p >= 0.5).astype(int)
            self.history["accuracy"].append(accuracy_score(y, y_pred))
            self.history["f1"].append(f1_score(y, y_pred, zero_division=0))
            self.history["precision"].append(precision_score(y, y_pred, zero_division=0))
            self.history["recall"].append(recall_score(y, y_pred, zero_division=0))

            # Gradienty
            dw = -np.dot(X.T, (y - p)) / n_samples
            db = -np.sum(y - p) / n_samples

            # Aktualizacja wag
            self.W -= self.alpha * dw
            self.b -= self.alpha * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        linear_model = np.dot(X, self.W) + self.b
        return sigmoid(linear_model)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def prepare_data():
    heart_disease = fetch_ucirepo(id=HEART_DISEASE_UCIREPO_ID)
    X = heart_disease.data.features
    vars_df = heart_disease.variables

    features = vars_df[vars_df["role"] == "Feature"]
    numeric_features = features[features["type"] == "Integer"]["name"].tolist()
    categorical_features = features[features["type"] == "Categorical"]["name"].tolist()

    y_original = heart_disease.data.targets.iloc[:, 0]
    y_binary = (y_original > 0).astype(int)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_binary,
        test_size=0.2,
        random_state=42,
        stratify=y_binary,
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return (
        preprocessor,
        X_train,
        X_test,
        X_train_processed,
        X_test_processed,
        y_train.to_numpy(),
        y_test.to_numpy(),
    )


def run_baseline(preprocessor, X_train, X_test, y_train, y_test):
    print("\n" + "=" * 50)
    print(" BASELINE (scikit-learn LogisticRegression)")
    print("=" * 50)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.3f}")
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))


def plot_learning_curves(
    results_dict: dict,
    title: str,
    filename: str,
    metric_name: str = "loss",
    ylabel: str = "Cost Function (Cross-Entropy)",
):
    plt.figure(figsize=(10, 6))
    for label, history in results_dict.items():
        plt.plot(history[metric_name], label=label)

    plt.title(title)
    plt.xlabel("Epoka / Iteracja")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Zapisano wykres: {filename}")


def main() -> None:
    os.makedirs("plots", exist_ok=True)

    (
        preprocessor,
        X_train,
        X_test,
        X_train_proc,
        X_test_proc,
        y_train,
        y_test,
    ) = prepare_data()

    # 1. Baseline
    run_baseline(preprocessor, X_train, X_test, y_train, y_test)

    # 2. Eksperyment A: Wpływ współczynnika uczenia (alpha)
    print("\n" + "=" * 50)
    print(" EKSPERYMENT 1: Wpływ współczynnika uczenia (alpha)")
    print("=" * 50)

    alphas = [0.5, 0.1, 0.01, 0.001, 0.0001]
    results_alpha = {}

    for alpha in alphas:
        model = CustomLogisticRegression(alpha=alpha, epochs=1000, init_method="zeros")
        model.fit(X_train_proc, y_train)
        results_alpha[f"alpha = {alpha}"] = model.history

        y_pred = model.predict(X_test_proc)
        print(
            f"Alpha {alpha:6.4f} | Final Loss: {model.history['loss'][-1]:.4f} | "
            f"Test F1: {f1_score(y_test, y_pred):.3f} | "
            f"Test Prec: {precision_score(y_test, y_pred, zero_division=0):.3f} | "
            f"Test Rec: {recall_score(y_test, y_pred, zero_division=0):.3f}"
        )

    plot_learning_curves(
        results_alpha,
        "Wpływ współczynnika uczenia (alpha) na funkcję kosztu",
        "plots/learning_curve_alpha_loss.png",
        metric_name="loss",
        ylabel="Cross-Entropy Loss",
    )
    plot_learning_curves(
        results_alpha,
        "Wpływ współczynnika uczenia (alpha) na F1-score (trening)",
        "plots/learning_curve_alpha_f1.png",
        metric_name="f1",
        ylabel="F1-Score",
    )

    # 3. Eksperyment B: Wpływ inicjalizacji wag
    print("\n" + "=" * 50)
    print(" EKSPERYMENT 2: Wpływ rozkładu inicjalizacji parametrów")
    print("=" * 50)

    init_methods = {
        "Zeros": "zeros",
        "Normal (std=0.01)": "normal",
        "Uniform (-0.01, 0.01)": "uniform",
        "Large Normal (std=1.0)": "large_normal",
    }
    results_init = {}

    for label, method in init_methods.items():
        model = CustomLogisticRegression(alpha=0.1, epochs=1000, init_method=method)
        model.fit(X_train_proc, y_train)
        results_init[label] = model.history

        y_pred = model.predict(X_test_proc)
        print(
            f"Init: {label:22s} | Final Loss: {model.history['loss'][-1]:.4f} | Test F1: {f1_score(y_test, y_pred):.3f}"
        )

    plot_learning_curves(
        results_init,
        "Wpływ inicjalizacji wag na funkcję kosztu (alpha=0.1)",
        "plots/learning_curve_init_loss.png",
        metric_name="loss",
        ylabel="Cross-Entropy Loss",
    )

    # 4. Eksperyment C: Wpływ liczby epok / warunku stopu
    print("\n" + "=" * 50)
    print(" EKSPERYMENT 3: Ocena zbieżności najlepszego modelu")
    print("=" * 50)

    best_model = CustomLogisticRegression(alpha=0.1, epochs=2000, init_method="zeros")
    best_model.fit(X_train_proc, y_train)
    y_pred_best = best_model.predict(X_test_proc)

    print("Najlepszy własny model (alpha=0.1, epochs=2000, init=zeros):")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred_best):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred_best):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred_best):.3f}")
    print(f"F1-score:  {f1_score(y_test, y_pred_best):.3f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred_best))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_best))


if __name__ == "__main__":
    main()
