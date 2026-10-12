import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

ROUNDING = 2
HEART_DISEASE_UCIREPO_ID = 45


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def main() -> None:
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

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nBaseline results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.3f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()

    n_features = X_train_processed.shape[1]
    W = np.zeros(n_features)
    b = 0.0

    alpha = 0.01
    epochs = 1000

    for epoch in range(epochs):
        linear_model = np.dot(X_train_processed, W) + b
        p = sigmoid(linear_model)

        dw = -np.dot(X_train_processed.T, (y_train_np - p)) / len(y_train_np)
        db = -np.sum(y_train_np - p) / len(y_train_np)

        W = W - alpha * dw
        b = b - alpha * db

    test_linear = np.dot(X_test_processed, W) + b
    y_pred_probs = sigmoid(test_linear)
    y_pred = (y_pred_probs >= 0.5).astype(int)

    print(f"F1-score własnego modelu: {f1_score(y_test_np, y_pred):.3f}")
