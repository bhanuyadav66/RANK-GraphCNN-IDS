from fastapi import FastAPI
import torch

app = FastAPI(title="RANK Graph-CNN IDS API")

CSV_PATH = "data/processed_data.csv"
MODEL_PATH = "models/graphcnn_model.pth"

model = None
graphs = None


@app.get("/")
def root():
    return {"status": "API is running"}


@app.get("/predict/sample")
def predict_sample():
    global model, graphs

    try:
        from graph.graph_builder import build_graph_dataset
        from models.graphcnn import GraphCNN

        # Build graphs ONLY ONCE
        if graphs is None:
            graphs = build_graph_dataset(CSV_PATH)

        # Load model ONLY ONCE
        if model is None:
            model = GraphCNN(num_features=graphs[0].x.shape[1])
            model.load_state_dict(
                torch.load(MODEL_PATH, map_location="cpu")
            )
            model.eval()

        # ⚡ Use a PRE-CACHED graph (fast)
        graph = graphs[0]  # normal or attack sample

        with torch.no_grad():
            score = model(graph).item()

        return {
            "prediction": "Attack" if score > 0.5 else "Normal",
            "confidence": round(float(score), 4),
            "nodes": graph.x.shape[0],
            "edges": graph.edge_index.shape[1] // 2
        }

    except Exception as e:
        return {"error": str(e)}
