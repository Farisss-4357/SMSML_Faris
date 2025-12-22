import pandas as pd
import numpy as np
import os
import mlflow
import dagshub # Diperlukan untuk Kriteria Advanced
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import io

# =========================================================================
# === KONFIGURASI PATH FINAL (WAJIB SESUAIKAN DENGAN PATH ASLI) ===
# Jika data Anda berada di lokasi spesifik, path harus lengkap!
FULL_PROJECT_PATH = "C:/Users/ASUS/Downloads/SMSML_Faris" 
DATA_FOLDER_NAME = "Eksperimen_SML_Faris/preprocessing" 
TARGET_COLUMN = "species" 

DATA_PATH = os.path.join(FULL_PROJECT_PATH, DATA_FOLDER_NAME)
TRAIN_FILE = os.path.join(DATA_PATH, "train_data.csv")
TEST_FILE = os.path.join(DATA_PATH, "test_data.csv")
# =========================================================================

# --- KONFIGURASI DAGSHUB (Wajib untuk Kriteria Advanced) ---
# GANTI DENGAN NAMA PENGGUNA DAN REPO ANDA YANG BENAR
dagshub.init(repo_owner='Farisss-4357', repo_name='SMSML_Faris_MLOps', mlflow=True) 

# --- Setup MLflow ---
mlflow.sklearn.autolog(disable=True) 
mlflow.set_experiment("Kriteria 3 - Hyperparameter Tuning (Grid Search)")

# --- DEFINISI HYPERPARAMETER GRID ---
param_grid = {
    'C': [0.1, 1.0, 10.0],
    'solver': ['liblinear', 'lbfgs', 'newton-cg'],
    'penalty': ['l1', 'l2']
}

# -------------------------------------------------------------------------
# === FUNGSI ARTEFAK TAMBAHAN (Wajib untuk Kriteria Advanced) ===
# -------------------------------------------------------------------------

def log_custom_artifacts(model, X_test, y_test, y_pred, feature_names):
    """Mencatat Confusion Matrix dan Feature Importance Plot sebagai Artefak."""
    
    # 1. ARTEFAK 1: Confusion Matrix Plot
    try:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 6))
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Simpan plot ke memori dan catat
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close(fig)
    except Exception as e:
        print(f"Gagal mencatat Confusion Matrix: {e}")

    # 2. ARTEFAK 2: Feature Importance Plot (Hanya untuk LogisticRegression)
    try:
        if hasattr(model, 'coef_'):
            importances = model.coef_[0]
            fig, ax = plt.subplots(figsize=(8, 4))
            plt.bar(feature_names, importances)
            plt.title("Feature Importance")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            plt.savefig("feature_importance.png")
            mlflow.log_artifact("feature_importance.png")
            plt.close(fig)
    except Exception as e:
        print(f"Gagal mencatat Feature Importance: {e}")


# -------------------------------------------------------------------------
# === FUNGSI UTAMA TUNING ===
# -------------------------------------------------------------------------

def load_data():
    """Memuat data training dan testing."""
    try:
        # Menggunakan path lengkap yang sudah dikoreksi
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)
        
        X_train = train_df.drop(TARGET_COLUMN, axis=1)
        y_train = train_df[TARGET_COLUMN]
        X_test = test_df.drop(TARGET_COLUMN, axis=1)
        y_test = test_df[TARGET_COLUMN]
        
        print("Data training dan testing berhasil dimuat.")
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        # Jika masih gagal, mungkin file CSV ada di working directory
        print(f"ERROR: File data tidak ditemukan di path: {TRAIN_FILE}")
        return None, None, None, None


def perform_tuning():
    X_train, X_test, y_train, y_test = load_data()
    if X_train is None:
        return

    run_counter = 1
    
    for C in param_grid['C']:
        for solver in param_grid['solver']:
            for penalty in param_grid['penalty']:
                
                if penalty == 'l1' and solver not in ('liblinear', 'saga'):
                    continue
                if penalty == 'l1' and solver in ('lbfgs', 'newton-cg'):
                    continue
                
                with mlflow.start_run(run_name=f"Run-{run_counter}_Solver-{solver}"):
                    
                    try:
                        print(f"[{run_counter}] Melatih dengan C={C}, Solver={solver}, Penalty={penalty}...")
                        
                        # 1. Inisialisasi dan Latih Model
                        tuned_model = LogisticRegression(C=C, solver=solver, penalty=penalty, max_iter=500, random_state=42)
                        tuned_model.fit(X_train, y_train)
                        
                        # 2. Catat Hyperparameter secara Manual
                        mlflow.log_param("C", C)
                        mlflow.log_param("solver", solver)
                        mlflow.log_param("penalty", penalty)
                        
                        # 3. Evaluasi Metrik dan Catat
                        y_pred = tuned_model.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)
                        mlflow.log_metric("test_accuracy", acc)
                        print(f"-> Accuracy: {acc:.4f}\n")

                        # 4. LOGGING ARTEFAK TAMBAHAN (2 Artefak)
                        log_custom_artifacts(tuned_model, X_test, y_test, y_pred, X_train.columns)

                        # 5. Simpan Model
                        mlflow.sklearn.log_model(tuned_model, "model")
                        
                        run_counter += 1

                    except Exception as e:
                        print(f"[{run_counter}] GAGAL karena konfigurasi tidak valid: {C}, {solver}, {penalty}. Error: {e}")
                        mlflow.log_metric("test_accuracy", 0.0)
                        run_counter += 1


if __name__ == "__main__":
    perform_tuning()