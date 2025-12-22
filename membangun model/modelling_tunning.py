import pandas as pd
import numpy as np
import os
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================================================================
TRAIN_FILE = os.path.join("train_data.csv")
TEST_FILE = os.path.join("test_data.csv")
TARGET_COLUMN = "species"
# =========================================================================

# --- Setup MLflow ---
# 1. Matikan Autologging (Kriteria 3 wajib Manual Logging)
mlflow.sklearn.autolog(disable=True) 

# 2. Set Eksperimen untuk Tuning
mlflow.set_experiment("Kriteria 3 - Hyperparameter Tuning (Grid Search)")

# --- DEFINISI HYPERPARAMETER GRID ---
# Kita akan mencoba 3 * 3 * 2 = 18 kombinasi unik
param_grid = {
    'C': [0.1, 1.0, 10.0],
    'solver': ['liblinear', 'lbfgs', 'newton-cg'],
    'penalty': ['l1', 'l2']
}

def load_data():
    """Memuat data training dan testing."""
    try:
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)
        
        X_train = train_df.drop(TARGET_COLUMN, axis=1)
        y_train = train_df[TARGET_COLUMN]
        X_test = test_df.drop(TARGET_COLUMN, axis=1)
        y_test = test_df[TARGET_COLUMN]
        
        print("Data training dan testing berhasil dimuat.")
        return X_train, X_test, y_train, y_test
    except FileNotFoundError as e:
        print(f"ERROR: File data tidak ditemukan. Cek path: {e.filename}")
        return None, None, None, None

def perform_tuning():
    X_train, X_test, y_train, y_test = load_data()
    if X_train is None:
        return

    run_counter = 1
    
    # Loop melalui SEMUA KOMBINASI di grid secara manual
    for C in param_grid['C']:
        for solver in param_grid['solver']:
            for penalty in param_grid['penalty']:
                
                # Cek kompatibilitas (L1 tidak bisa dengan 'lbfgs' atau 'newton-cg')
                if penalty == 'l1' and solver not in ('liblinear', 'saga'):
                    continue
                if penalty == 'l2' and solver == 'liblinear':
                    pass
                if penalty == 'l1' and solver in ('lbfgs', 'newton-cg'):
                    continue
                
                # --- MEMULAI RUN MLFLOW untuk SETIAP KOMBINASI ---
                with mlflow.start_run(run_name=f"Run-{run_counter}_Solver-{solver}"):
                    
                    try:
                        print(f"[{run_counter}] Melatih dengan C={C}, Solver={solver}, Penalty={penalty}...")
                        
                        # 1. Inisialisasi dan Latih Model
                        tuned_model = LogisticRegression(
                            C=C, 
                            solver=solver, 
                            penalty=penalty, 
                            max_iter=500, 
                            random_state=42
                        )
                        tuned_model.fit(X_train, y_train)
                        
                        # 2. Catat Hyperparameter secara Manual (KRITERIA 3: MANUAL LOGGING)
                        mlflow.log_param("C", C)
                        mlflow.log_param("solver", solver)
                        mlflow.log_param("penalty", penalty)
                        
                        # 3. Evaluasi Metrik
                        y_pred = tuned_model.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)
                        
                        # 4. Catat Metrik secara Manual
                        mlflow.log_metric("test_accuracy", acc)
                        
                        print(f"-> Accuracy: {acc:.4f}\n")

                        # 5. Simpan Model untuk run ini (Manual Logging Model)
                        mlflow.sklearn.log_model(tuned_model, "model")
                        
                        run_counter += 1

                    except Exception as e:
                        print(f"[{run_counter}] GAGAL karena konfigurasi tidak valid: {C}, {solver}, {penalty}. Error: {e}")
                        mlflow.log_metric("test_accuracy", 0.0)
                        run_counter += 1


if __name__ == "__main__":
    perform_tuning()