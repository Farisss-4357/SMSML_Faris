import pandas as pd
import numpy as np
import os
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================================================================
# Asumsi file-file ini ada di direktori yang sama
TRAIN_FILE = os.path.join("train_data.csv")
TEST_FILE = os.path.join("test_data.csv")
TARGET_COLUMN = "species"
# =========================================================================

# --- Setup MLflow ---
mlflow.sklearn.autolog(disable=True)
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
                
                # --- Pengecekan Kompatibilitas Scikit-learn ---
                # L1 penalty HANYA didukung oleh 'liblinear' (dan 'saga', yang tidak ada di grid ini)
                is_incompatible = (penalty == 'l1' and solver not in ('liblinear',))
                
                if is_incompatible:
                    print(f"[{run_counter}] Melewatkan kombinasi tidak valid (L1 tidak kompatibel dengan {solver}).")
                    # Tetap mencatat run yang dilewati di MLflow
                    with mlflow.start_run(run_name=f"Run-{run_counter}_Solver-{solver}-SKIPPED"):
                        mlflow.log_param("C", C)
                        mlflow.log_param("solver", solver)
                        mlflow.log_param("penalty", penalty)
                        mlflow.log_metric("test_accuracy", 0.0) # Log 0.0 untuk kombinasi gagal/dilewati
                    run_counter += 1
                    continue
                
                # --- MEMULAI RUN MLFLOW untuk SETIAP KOMBINASI VALID ---
                with mlflow.start_run(run_name=f"Run-{run_counter}_Solver-{solver}_P-{penalty}"):
                    
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
                        # Memberi tag unik pada model
                        mlflow.sklearn.log_model(tuned_model, "model")
                        
                        run_counter += 1

                    except Exception as e:
                        # Ini menangani kasus lain yang tidak terduga, mis. max_iter tidak cukup
                        print(f"[{run_counter}] GAGAL saat fitting: {C}, {solver}, {penalty}. Error: {e}")
                        mlflow.log_metric("test_accuracy", 0.0)
                        run_counter += 1


if __name__ == "__main__":
    perform_tuning()