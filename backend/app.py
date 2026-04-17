from fastapi import FastAPI
import torch
import pandas as pd

from graph.graph_builder import build_graph_from_window
from models.graphcnn import GraphCNN

app = FastAPI(title="RANK Graph-CNN IDS API")

ATTACK_TYPE_LABELS = {
    "DoS": "DoS (Denial of Service)",
    "Reconnaissance": "Recon (Reconnaissance)",
    "Exploits": "Exploit Attempt",
    "Generic": "Generic Attack",
}

# -------------------------------
# LOAD EVERYTHING ON STARTUP
# -------------------------------

print("Loading model and data...")

CSV_PATH = "data/processed_data.csv"
MODEL_PATH = "models/graphcnn_model.pth"

# Load dataset ONCE and take a small 100-row window
df = pd.read_csv(CSV_PATH)
sample_window = df.iloc[:100]
sample_graph = build_graph_from_window(sample_window)

# Load model ONCE
model = GraphCNN(num_features=sample_graph.x.shape[1])
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

print("Backend ready!")

# -------------------------------
# ROOT
# -------------------------------

@app.get("/")
def root():
    return {"status": "API is running"}

# -------------------------------
# FAST PREDICTION (cached graph)
# -------------------------------

@app.get("/predict/sample")
def predict_sample():
    try:
        with torch.no_grad():
            out = model(sample_graph).item()

        prediction = "Attack" if out > 0.5 else "Normal"
        confidence = out if prediction == "Attack" else 1 - out
        raw_attack_type = str(getattr(sample_graph, "attack_cat", "Unknown"))
        attack_type = (
            ATTACK_TYPE_LABELS.get(raw_attack_type, raw_attack_type)
            if prediction == "Attack"
            else "Normal"
        )

        return {
            "prediction": prediction,
            "confidence": round(float(confidence), 4),
            "nodes": sample_graph.x.shape[0],
            "edges": sample_graph.edge_index.shape[1],
            "attack_type": attack_type,
        }
    except Exception as e:
        return {"error": str(e)}
