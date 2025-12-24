import time
import mlflow
import mlflow.pyfunc
import pandas as pd
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# =========================
# MLFLOW DAGSHUB CONFIG
# =========================
mlflow.set_tracking_uri(
    "https://dagshub.com/Farisss-4357/SMSML_Faris_MLOps.mlflow"
)

MODEL_URI = "runs:/d1225e7863f34591a48ebe53b8938ac9/model"

print("Loading model from MLflow...")
model = mlflow.pyfunc.load_model(MODEL_URI)
print("Model loaded successfully!")

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

# =========================
# PROMETHEUS METRICS
# =========================
REQUEST_COUNT = Counter(
    "inference_request_total",
    "Total number of inference requests"
)

INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference latency in seconds"
)

ERROR_COUNT = Counter(
    "inference_error_total",
    "Total inference errors"
)

# =========================
# PREDICT ENDPOINT
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()

    try:
        payload = request.get_json()

        if "instances" not in payload:
            ERROR_COUNT.inc()
            return jsonify({"error": "Key 'instances' not found"}), 400

        df = pd.DataFrame(payload["instances"])
        prediction = model.predict(df)

        INFERENCE_LATENCY.observe(time.time() - start_time)

        return jsonify({
            "prediction": prediction.tolist()
        })

    except Exception as e:
        ERROR_COUNT.inc()
        return jsonify({"error": str(e)}), 500

# =========================
# METRICS ENDPOINT
# =========================
@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {
        "Content-Type": CONTENT_TYPE_LATEST
    }

# =========================
# SERVER START
# =========================
if __name__ == "__main__":
    print("Inference server running on http://localhost:8000")
    app.run(host="0.0.0.0", port=8000)