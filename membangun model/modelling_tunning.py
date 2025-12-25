import pandas as pd
import os
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================================================
# PATH FILE
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_FILE = os.path.join(BASE_DIR, "telecom_churn_train.csv")
TEST_FILE  = os.path.join(BASE_DIR, "telecom_churn_test.csv")

TARGET_COLUMN = "Churn"
# =========================================================

# =========================================================
# 🔗 DAGSHUB CONFIG (WAJIB UNTUK ADVANCED)
dagshub.init(
    repo_owner="Farisss-4357",
    repo_name="telecom-churn-mlflow",
    mlflow=True
)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Telecom Churn - Kriteria 2 Advanced")
# =========================================================

# Matikan autolog (WAJIB untuk Skilled & Advanced)
mlflow.sklearn.autolog(disable=True)

# Hyperparameter Grid
param_grid = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "solver": ["liblinear", "lbfgs"],
    "penalty": ["l1", "l2"]
}

def load_data():
    try:
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)

        X_train = train_df.drop(TARGET_COLUMN, axis=1)
        y_train = train_df[TARGET_COLUMN]

        X_test = test_df.drop(TARGET_COLUMN, axis=1)
        y_test = test_df[TARGET_COLUMN]

        print("✅ Data train & test berhasil dimuat")
        return X_train, X_test, y_train, y_test

    except FileNotFoundError as e:
        print(f"❌ File tidak ditemukan: {e.filename}")
        return None, None, None, None


def hyperparameter_tuning():
    X_train, X_test, y_train, y_test = load_data()
    if X_train is None:
        return

    run_id = 1

    for C in param_grid["C"]:
        for solver in param_grid["solver"]:
            for penalty in param_grid["penalty"]:

                # Validasi kombinasi
                if penalty == "l1" and solver != "liblinear":
                    print(f"[{run_id}] SKIP kombinasi tidak valid: {solver} + {penalty}")

                    with mlflow.start_run(run_name=f"Run-{run_id}-SKIPPED"):
                        mlflow.log_param("C", C)
                        mlflow.log_param("solver", solver)
                        mlflow.log_param("penalty", penalty)
                        mlflow.log_metric("test_accuracy", 0.0)

                    run_id += 1
                    continue

                with mlflow.start_run(run_name=f"Run-{run_id}-{solver}-{penalty}"):

                    print(f"[{run_id}] Training: C={C}, solver={solver}, penalty={penalty}")

                    model = LogisticRegression(
                        C=C,
                        solver=solver,
                        penalty=penalty,
                        max_iter=500,
                        random_state=42
                    )

                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)

                    # ===== Manual Logging =====
                    mlflow.log_param("C", C)
                    mlflow.log_param("solver", solver)
                    mlflow.log_param("penalty", penalty)
                    mlflow.log_metric("test_accuracy", acc)

                    # Artefak tambahan (WAJIB ADVANCED)
                    mlflow.log_text(
                        f"Model LogisticRegression dengan C={C}, solver={solver}, penalty={penalty}",
                        "model_description.txt"
                    )

                    mlflow.sklearn.log_model(model, artifact_path="model")

                    print(f"    → Accuracy: {acc:.4f}\n")

                run_id += 1


if __name__ == "__main__":
    hyperparameter_tuning()