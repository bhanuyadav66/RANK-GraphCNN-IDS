import torch
from torch_geometric.loader import DataLoader
from graph.graph_builder import build_graph_dataset
from models.graphcnn import GraphCNN
import random


CSV_PATH = "C:\\Users\\BUNNY YADAV\\RANK-GraphCNN\\data\\processed_data.csv"
BATCH_SIZE = 16
EPOCHS = 20
LR = 0.001


def train():
    graphs = build_graph_dataset(CSV_PATH)
    random.shuffle(graphs)

    # Split data
    train_size = int(0.8 * len(graphs))
    train_graphs = graphs[:train_size]
    test_graphs = graphs[train_size:]

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE)

    model = GraphCNN(num_features=train_graphs[0].x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.BCELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for data in train_loader:
            optimizer.zero_grad()
            out = model(data).view(-1)
            loss = criterion(out, data.y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "models/graphcnn_model.pth")
    print("Model saved to models/graphcnn_model.pth")


if __name__ == "__main__":
    train()
