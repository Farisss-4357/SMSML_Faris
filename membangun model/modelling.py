import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import os

# =========================================================
# KONFIGURASI PATH FILE
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_FILE = os.path.join(BASE_DIR, "telecom_churn_train.csv")
TEST_FILE  = os.path.join(BASE_DIR, "telecom_churn_test.csv")

TARGET_COLUMN = "Churn"
# =========================================================
# SETUP MLFLOW
# =========================================================
mlflow.set_experiment("Telecom Churn - Kriteria 2 Basic")
mlflow.sklearn.autolog()

def load_data():
    """Memuat data train dan test hasil preprocessing (Kriteria 1)."""
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
        print("❌ FILE TIDAK DITEMUKAN:", e.filename)
        return None, None, None, None


def train_model():
    X_train, X_test, y_train, y_test = load_data()
    if X_train is None:
        return

    with mlflow.start_run(run_name="LogisticRegression_Churn"):

        # Inisialisasi model
        model = LogisticRegression(
            solver="liblinear",
            max_iter=500,
            random_state=42
        )

        print("🚀 Training model...")
        model.fit(X_train, y_train)

        # Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Logging metric tambahan
        mlflow.log_metric("test_accuracy", acc)

        print(f"✅ Test Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("\n📊 Cek MLflow UI untuk hasil training")


if __name__ == "__main__":
    train_model()