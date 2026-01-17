import pandas as pd
import torch
from torch_geometric.data import Data

# ---------------- CONFIG ----------------
WINDOW_SIZE = 100
STRIDE = 50
MIN_NORMAL_NODES = 2


GRAPH_ID_COLS = ['srcip', 'dstip', 'sport', 'dsport', 'Stime']
LABEL_COLS = ['Label', 'attack_cat']
# ----------------------------------------


def build_graph_from_window(df_window):
    """
    Converts a dataframe window into a PyTorch Geometric graph
    """

    # -------- Node features --------
    x = df_window.drop(columns=GRAPH_ID_COLS + LABEL_COLS)
    x = torch.tensor(x.values, dtype=torch.float)

    # -------- Edge construction --------
    edge_index = []

    for i in range(len(df_window)):
        for j in range(i + 1, len(df_window)):
            if (
                df_window.iloc[i]['srcip'] == df_window.iloc[j]['srcip'] or
                df_window.iloc[i]['dstip'] == df_window.iloc[j]['dstip'] or
                df_window.iloc[i]['dsport'] == df_window.iloc[j]['dsport']
            ):
                edge_index.append([i, j])
                edge_index.append([j, i])

    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()

    # -------- Graph label (temporary, overwritten later) --------
    attack_ratio = df_window['Label'].mean()
    binary_label = 1 if attack_ratio >= 0.3 else 0
    attack_cat = df_window['attack_cat'].mode()[0]

    data = Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([binary_label], dtype=torch.long)
    )

    data.attack_cat = attack_cat
    return data


def build_graph_dataset(csv_path):
    """
    Builds a balanced graph dataset:
    - Attack graphs from attack flows (sliding windows)
    - Normal graphs from normal flows (adaptive window)
    """

    df = pd.read_csv(csv_path)

    normal_df = df[df['Label'] == 0]
    attack_df = df[df['Label'] == 1]

    graphs = []

    # -------- NORMAL GRAPHS (adaptive window) --------
    normal_size = min(WINDOW_SIZE, len(normal_df))

    if normal_size >= MIN_NORMAL_NODES:
        window = normal_df.iloc[:normal_size]
        graph = build_graph_from_window(window)
        graph.y = torch.tensor([0], dtype=torch.long)
        graphs.append(graph)

    # -------- ATTACK GRAPHS (sliding windows) --------
    for start in range(0, len(attack_df) - WINDOW_SIZE, STRIDE):
        window = attack_df.iloc[start:start + WINDOW_SIZE]
        graph = build_graph_from_window(window)
        graph.y = torch.tensor([1], dtype=torch.long)
        graphs.append(graph)

    return graphs


# ---------------- DEBUG / STANDALONE RUN ----------------
if __name__ == "__main__":
    graphs = build_graph_dataset(
        "C:/Users/BUNNY YADAV/RANK-GraphCNN/data/processed_data.csv"
    )

    print("Total graphs created:", len(graphs))

    labels = [int(g.y.item()) for g in graphs]
    from collections import Counter
    print("Graph label distribution:", Counter(labels))

    print("Sample graph:")
    print(graphs[0])

    # Optional visualization
    try:
        import networkx as nx
        import matplotlib.pyplot as plt

        def visualize_graph(data):
            G = nx.Graph()
            edge_list = data.edge_index.t().tolist()
            G.add_edges_from(edge_list)

            plt.figure(figsize=(6, 6))
            nx.draw(G, node_size=30)
            plt.title("Incident Graph")
            plt.show()

        visualize_graph(graphs[0])
    except ImportError:
        pass
