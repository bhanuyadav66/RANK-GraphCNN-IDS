from graph.graph_builder import build_graph_dataset
from collections import Counter

graphs = build_graph_dataset("data/processed_data.csv")

labels = [int(g.y.item()) for g in graphs]
print("Graph label distribution:", Counter(labels))
