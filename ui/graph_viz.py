import os
from pathlib import Path
import tempfile

import networkx as nx
from pyvis.network import Network


def visualize_graph_pyg(data, prediction):
    G = nx.Graph()

    num_nodes = int(getattr(data, "num_nodes", 0) or 0)
    if num_nodes:
        G.add_nodes_from(range(num_nodes))

    edges = data.edge_index.t().tolist()
    G.add_edges_from(edges)

    net = Network(
        height="500px",
        width="100%",
        bgcolor="#111",
        font_color="white",
        notebook=False,
        cdn_resources="in_line",
    )

    degrees = dict(G.degree())
    avg_degree = (sum(degrees.values()) / len(degrees)) if degrees else 0

    for node in G.nodes():
        degree = degrees.get(node, 0)

        if prediction == "Attack" and degree >= avg_degree:
            color = "#ff4b4b"
            size = 13
            title = f"Alert {node}: high-correlation node (degree={degree})"
        else:
            color = "#4da6ff"
            size = 8
            title = f"Alert {node}: background node (degree={degree})"

        net.add_node(node, color=color, size=size, title=title)

    net.add_edges(list(G.edges()))
    net.force_atlas_2based(gravity=-45)

    fd, html_path = tempfile.mkstemp(suffix=".html")
    os.close(fd)
    Path(html_path).unlink(missing_ok=True)

    html_content = net.generate_html(notebook=False)
    with open(html_path, "w", encoding="utf-8") as html_file:
        html_file.write(html_content)

    return html_path, avg_degree
