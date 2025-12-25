import requests
import time
import random

PREDICT_URL = "http://localhost:8000/predict"

def send_request():
    """
    Mengirim request prediksi ke inference server
    untuk mensimulasikan traffic user
    """

    data = {
        "instances": [
            [
                random.uniform(-3, 3),  # feature 1
                random.uniform(-4, 4),  # feature 2
                random.uniform(-4, 4),  # feature 3
                random.uniform(-3, 3),  # feature 4
                random.uniform(-2, 2),  # feature 5
                random.uniform(-1, 1),  # feature 6
                random.uniform(-5, 5),  # feature 7
                random.uniform(-6, 6),  # feature 8
                random.uniform(-3, 3),  # feature 9
                random.uniform(-4, 4),  # feature 10
            ]
        ]
    }


    try:
        response = requests.post(PREDICT_URL, json=data)

        if response.status_code == 200:
            print("Prediksi berhasil:", response.json())
        else:
            print(
                f"Error prediksi | "
                f"Status: {response.status_code} | "
                f"Pesan: {response.text}"
            )

    except requests.exceptions.ConnectionError:
        print("❌ Inference server belum jalan (port 8000)")
    except Exception as e:
        print("❌ Error tak terduga:", e)


if __name__ == "__main__":
    print("🚀 Mulai simulasi traffic inference...")
    while True:
        send_request()
        time.sleep(random.uniform(1, 2))