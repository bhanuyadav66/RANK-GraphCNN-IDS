import torch
from torch_geometric.loader import DataLoader
from graph.graph_builder import build_graph_dataset
from models.graphcnn import GraphCNN
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CSV_PATH = "C:/Users/BUNNY YADAV/RANK-GraphCNN/data/processed_data.csv"
MODEL_PATH = "models/graphcnn_model.pth"
BATCH_SIZE = 16


def evaluate():
    # ---------------- Load graphs ----------------
    graphs = build_graph_dataset(CSV_PATH)

    # ---------------- Separate by class ----------------
    normal_graphs = [g for g in graphs if g.y.item() == 0]
    attack_graphs = [g for g in graphs if g.y.item() == 1]

    print("Total graphs:", len(graphs))
    print("Normal graphs:", len(normal_graphs))
    print("Attack graphs:", len(attack_graphs))

    # ---------------- Manual stratified test set ----------------
    test_graphs = []

    # Force at least ONE normal graph into test
    if len(normal_graphs) > 0:
        test_graphs.append(normal_graphs[0])

    # Add ~20% attack graphs to test
    test_attack_count = max(1, int(0.2 * len(attack_graphs)))
    test_graphs.extend(attack_graphs[:test_attack_count])

    # ---------------- DataLoader ----------------
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE)

    # ---------------- Load model ----------------
    model = GraphCNN(num_features=test_graphs[0].x.shape[1])
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # ---------------- Evaluation ----------------
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in test_loader:
            out = model(data).view(-1)
            preds = (out > 0.5).int().tolist()
            y_pred.extend(preds)
            y_true.extend(data.y.tolist())

    # ---------------- Metrics ----------------
    print("\nGraph-CNN Evaluation Results (IDS Mode)")
    print("Accuracy:", accuracy_score(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=[0, 1],
                                target_names=["Normal", "Attack"]))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))


if __name__ == "__main__":
    evaluate()
