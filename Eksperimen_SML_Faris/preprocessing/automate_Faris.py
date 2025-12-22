import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Fungsi untuk melakukan preprocessing data secara otomatis
def preprocess_data(file_path, output_path, train_output_path, test_output_path):
    # 1. Memuat dataset
    df = pd.read_csv(file_path)
    print(f"Dataset dimuat dari {file_path}")

    # 2. Periksa nilai unik di kolom 'species'
    print("Unik values di kolom 'species' sebelum pembersihan:")
    print(df['species'].unique())

    # 3. Mengatasi missing values
    # PERBAIKAN: Tambahkan numeric_only=True agar tidak error saat bertemu kolom teks
    df.fillna(df.mean(numeric_only=True), inplace=True) 
    print("Missing values pada kolom numerik telah diisi dengan rata-rata")

    # 4. Bersihkan kolom 'species' dari spasi ekstra
    df['species'] = df['species'].astype(str).str.strip().replace(r'\s+', ' ', regex=True)
    
    # 5. Menampilkan nilai unik setelah pembersihan
    print("Unik values di kolom 'species' setelah pembersihan:")
    print(df['species'].unique())

    # 6. Encoding variabel kategorikal 'species' menjadi numerik
    # Menggunakan map sesuai kode Anda
    mapping = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    df['species'] = df['species'].map(mapping)
    
    # Tambahan: Cek jika ada mapping yang gagal (NaN) karena typo di dataset
    if df['species'].isnull().any():
        print("Peringatan: Ada nilai di kolom species yang tidak terpetakan!")
        # Mengisi nilai null hasil mapping yang gagal dengan nilai default atau hapus
        df.dropna(subset=['species'], inplace=True)

    print("Kolom 'species' telah diencoding menjadi numerik")

    # 7. Standarisasi fitur numerik menggunakan StandardScaler
    scaler = StandardScaler()
    kolom_fitur = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    df[kolom_fitur] = scaler.fit_transform(df[kolom_fitur])
    print("Fitur numerik telah distandarisasi")

    # 8. Menyimpan data yang sudah diproses ke file CSV
    df.to_csv(output_path, index=False)
    print(f"Data yang sudah diproses disimpan di {output_path}")

    # 9. Pisahkan fitur dan target
    X = df.drop('species', axis=1)
    y = df['species']

    # 10. Bagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 11. Simpan data latih
    train_data = pd.concat([X_train, y_train], axis=1)
    train_data.to_csv(train_output_path, index=False)
    print(f"Data latih disimpan di {train_output_path}")

    # 12. Simpan data uji
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.to_csv(test_output_file, index=False)
    print(f"Data uji disimpan di {test_output_file}")

    return df

if __name__ == "__main__":
    input_file = "../namadataset_raw/iris.csv"
    output_file = "iris_preprocessing.csv"
    train_output_file = "train_data.csv"
    test_output_file = "test_data.csv"

    preprocess_data(input_file, output_file, train_output_file, test_output_file)