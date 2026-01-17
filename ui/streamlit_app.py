import streamlit as st
import requests
import streamlit.components.v1 as components
import pandas as pd

from graph.graph_builder import build_graph_from_window
from ui.graph_viz import visualize_graph_pyg

# =====================================================
# Configuration
# =====================================================
API_URL = "http://localhost:8000/predict/sample"
DATA_PATH = "data/processed_data.csv"

st.set_page_config(
    page_title="RANK Graph-CNN IDS",
    layout="centered"
)

# =====================================================
# Session State Initialization
# =====================================================
for key in ["prediction", "confidence", "nodes", "edges", "density", "graph_html"]:
    if key not in st.session_state:
        st.session_state[key] = None

# =====================================================
# Cache: Load Graph ONCE (heavy operation)
# =====================================================
@st.cache_resource
def load_sample_graph_once():
    df = pd.read_csv(DATA_PATH)
    window = df.iloc[:100]  # one incident-sized window
    graph = build_graph_from_window(window)
    return graph

# =====================================================
# Cache: Build Graph HTML (label-aware)
# IMPORTANT: leading underscore avoids hashing PyG graph
# =====================================================
@st.cache_resource
def load_graph_html(_graph, label):
    html_path, avg_degree = visualize_graph_pyg(_graph, label)
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    return html, avg_degree

# =====================================================
# Preload graph ONCE
# =====================================================
sample_graph = load_sample_graph_once()

# =====================================================
# UI Header
# =====================================================
st.title("🔐 RANK: Graph-CNN Intrusion Detection System")

st.markdown("""
This demo shows **incident-level intrusion detection**
using a **Graph Convolutional Neural Network (Graph-CNN)**.
""")

st.caption(
    "ℹ️ Graph construction and visualization are precomputed for demo stability. "
    "Model inference itself runs in near real-time."
)

# =====================================================
# Run IDS Button (FAST)
# =====================================================
if st.button("Run IDS on Sample Incident"):
    with st.spinner("Running Graph-CNN IDS inference..."):
        response = requests.get(API_URL)

    if response.status_code == 200:
        data = response.json()

        st.session_state.prediction = data["prediction"]
        st.session_state.confidence = data["confidence"]
        st.session_state.nodes = data["nodes"]
        st.session_state.edges = data["edges"]

        st.session_state.density = round(
            (2 * data["edges"]) / (data["nodes"] * (data["nodes"] - 1)),
            3
        )

        # Build graph visualization ONCE per prediction
        graph_html, _ = load_graph_html(sample_graph, st.session_state.prediction)
        st.session_state.graph_html = graph_html

    else:
        st.error("❌ Failed to contact IDS API")

# =====================================================
# Prediction Output
# =====================================================
if st.session_state.prediction is not None:

    st.success(f"Prediction: **{st.session_state.prediction}**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Confidence", f"{st.session_state.confidence:.2f}")
    col2.metric("Nodes", st.session_state.nodes)
    col3.metric("Edges", st.session_state.edges)

    st.markdown("---")
    st.subheader("🛡 IDS Interpretation")

    if st.session_state.prediction == "Attack":
        st.write("⚠️ Suspicious correlated activity detected.")
    else:
        st.write("✅ Behavior consistent with normal network traffic.")

# =====================================================
# Graph Visualization (INSTANT)
# =====================================================
st.markdown("---")
st.subheader("🕸️ Incident Correlation Graph")

if st.session_state.graph_html:
    components.html(st.session_state.graph_html, height=550, scrolling=True)

    st.markdown("""
**Node Color Legend**
- 🔴 **Red nodes**: Attack-related or highly correlated alerts  
- 🔵 **Blue nodes**: Normal background traffic  
Edges indicate shared attributes such as IP, port, or temporal proximity.
""")
else:
    st.info("Click **Run IDS on Sample Incident** to visualize the incident graph.")

# =====================================================
# Auto-generated Explanation
# =====================================================
if st.session_state.prediction is not None:

    st.markdown("###  Graph-based Explanation")

    if st.session_state.prediction == "Attack":
        explanation = (
            f"The incident graph contains **{st.session_state.nodes} alerts** "
            f"with **high interconnectivity** (density = {st.session_state.density}). "
            f"Several highly connected red nodes indicate coordinated activity, "
            f"which the Graph-CNN has learned to associate with attack behavior."
        )
    else:
        explanation = (
            f"The incident graph shows **low connectivity** "
            f"(density = {st.session_state.density}), indicating mostly independent alerts. "
            f"This pattern is consistent with normal network behavior."
        )

    st.write(explanation)

# =====================================================
# Footer
# =====================================================
st.markdown("---")
st.caption("RANK Graph-CNN IDS | Final Year Project Demo")
