import networkx as nx
from pyvis.network import Network
import tempfile
import os


def visualize_graph_pyg(data, prediction_label):
    """
    Visualize a PyG graph with node coloring based on connectivity.
    High-degree nodes = attack-dominant (red)
    Low-degree nodes = normal (blue)
    """

    G = nx.Graph()
    edge_index = data.edge_index.t().tolist()

    # Add nodes
    for i in range(data.x.shape[0]):
        G.add_node(i)

    # Add edges
    for src, dst in edge_index:
        G.add_edge(src, dst)

    # Degree-based coloring
    degrees = dict(G.degree())
    avg_degree = sum(degrees.values()) / len(degrees)

    net = Network(
        height="500px",
        width="100%",
        bgcolor="#0e1117",
        font_color="white"
    )

    for node, degree in degrees.items():
        if prediction_label == "Attack" and degree >= avg_degree:
            color = "#ff4b4b"   # 🔴 attack-related
        else:
            color = "#4da6ff"   # 🔵 normal-related

        net.add_node(
            node,
            label=f"Alert {node}",
            color=color,
            size=8 + (degree * 0.3)
        )

    for src, dst in G.edges():
        net.add_edge(src, dst, color="#888888")

    net.force_atlas_2based()

    tmp_dir = tempfile.gettempdir()
    html_path = os.path.join(tmp_dir, "incident_graph.html")
    net.save_graph(html_path)

    return html_path, avg_degree
