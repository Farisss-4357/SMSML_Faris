import pandas as pd
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn

mlflow.sklearn.autolog() 

# Ganti dengan path ke folder dataset Anda
DATASET_PATH = "namadataset_preprocessing/" 
TRAIN_FILE = DATASET_PATH + "train_data.csv"
TEST_FILE = DATASET_PATH + "test_data.csv"
TARGET_COLUMN = 'species' # Kolom target berdasarkan data Iris Anda

def load_data():
    """Memuat data training dan testing yang sudah dipisahkan"""
    try:
        train_df = pd.read_csv(TRAIN_FILE)
        X_train = train_df.drop(TARGET_COLUMN, axis=1)
        y_train = train_df[TARGET_COLUMN]
        
        test_df = pd.read_csv(TEST_FILE)
        X_test = test_df.drop(TARGET_COLUMN, axis=1)
        y_test = test_df[TARGET_COLUMN]
        
        print("Data training dan testing berhasil dimuat.")
        return X_train, X_test, y_train, y_test
    
    except FileNotFoundError as e:
        print(f"ERROR: File dataset tidak ditemukan. Pastikan data berada di path yang benar: {e}")
        return None, None, None, None

def train_model():
    X_train, X_test, y_train, y_test = load_data()
    
    if X_train is None:
        return

    # --- Memulai MLflow Run ---
    with mlflow.start_run():
        print("Memulai pelatihan model (Logistic Regression)...")
        
        # Inisialisasi Model
        model = LogisticRegression(
            solver='lbfgs', 
            multi_class='ovr', 
            max_iter=1000, 
            random_state=42
        )
        
        # Latih Model
        model.fit(X_train, y_train)

        # Log parameter tambahan
        mlflow.log_param("data_source", "iris_processed_files")
        
        print("Pelatihan selesai. Metrik dan Model telah dicatat oleh MLflow Autolog.")


if __name__ == "__main__":
    train_model()