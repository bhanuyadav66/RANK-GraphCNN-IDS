import streamlit as st
import requests
import streamlit.components.v1 as components
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph.graph_builder import build_graph_from_window
from ui.graph_viz import visualize_graph_pyg

# =====================================================
# CONFIG
# =====================================================
API_URL = "http://localhost:8000/predict/sample"

st.set_page_config(
    page_title="RANK Graph-CNN IDS",
    layout="wide",
)

st.markdown(
    """
<style>
html, body, [class*="css"] {
    font-size: 17px;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1 {
    font-size: 2.2rem !important;
    line-height: 1.2 !important;
}

h2, h3 {
    font-size: 1.45rem !important;
    line-height: 1.3 !important;
}

p, li, div[data-testid="stMarkdownContainer"] p {
    font-size: 1rem !important;
    line-height: 1.6 !important;
}

[data-testid="stMetric"] {
    padding: 0.85rem 1rem;
}

[data-testid="stMetricValue"] {
    font-size: 1.9rem !important;
    line-height: 1.2 !important;
}

[data-testid="stMetricLabel"] {
    font-size: 1rem !important;
}

[data-testid="stCaptionContainer"] {
    font-size: 0.98rem !important;
    line-height: 1.55 !important;
}

button {
    font-size: 1rem !important;
}

.graph-note {
    margin-top: 0.9rem;
    padding: 0.95rem 1rem;
    background: rgba(49, 51, 63, 0.06);
    border-radius: 0.6rem;
    font-size: 1rem;
    line-height: 1.65;
}

.graph-note strong {
    font-size: 1rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# =====================================================
# SESSION STATE
# =====================================================
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.confidence = None
    st.session_state.nodes = None
    st.session_state.edges = None
    st.session_state.density = None
    st.session_state.attack_type = None


# =====================================================
# LOAD GRAPH (ONCE)
# =====================================================
@st.cache_resource
def load_sample_graph_once():
    import pandas as pd

    df = pd.read_csv("data/processed_data.csv")
    window = df.iloc[:100]
    return build_graph_from_window(window)


@st.cache_resource
def load_graph_html(_graph):
    html_path, avg_degree = visualize_graph_pyg(_graph, "Attack")
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    return html, avg_degree


sample_graph = load_sample_graph_once()
graph_html, avg_degree = load_graph_html(sample_graph)

# =====================================================
# HEADER
# =====================================================
st.title("RANK: Graph-CNN Intrusion Detection System")

st.markdown(
    """
This demo shows **incident-level intrusion detection**
using a **Graph Convolutional Neural Network (Graph-CNN)**.
"""
)

st.caption("Inference runs in near real-time using a preloaded Graph-CNN model.")

# =====================================================
# SIDEBAR BUTTON
# =====================================================
if st.sidebar.button("Run IDS on Sample Incident"):
    try:
        with st.spinner("Running Graph-CNN IDS inference..."):
            response = requests.get(API_URL, timeout=5)

        if response.status_code == 200:
            data = response.json()

            st.session_state.prediction = data["prediction"]
            st.session_state.confidence = data["confidence"]
            st.session_state.nodes = data["nodes"]
            st.session_state.edges = data["edges"]
            st.session_state.attack_type = data.get("attack_type", "Unknown")

            if data["nodes"] > 1:
                st.session_state.density = round(
                    (2 * data["edges"]) / (data["nodes"] * (data["nodes"] - 1)),
                    3,
                )
            else:
                st.session_state.density = 0.0
        else:
            st.error("Failed to contact backend")

    except Exception:
        st.error("Backend not running on port 8000")

# =====================================================
# OUTPUT
# =====================================================
if st.session_state.prediction:
    if st.session_state.prediction == "Attack":
        st.error("High Risk Attack Detected", icon="🚨")
    else:
        st.success("Normal Traffic", icon="✅")

    st.info(f"Prediction: {st.session_state.prediction}")

    confidence_pct = min(st.session_state.confidence * 100, 99.9)

    col1, col2, col3, col4 = st.columns([1, 1, 1.4, 1])
    col1.metric("Confidence", f"{confidence_pct:.1f}%")
    col2.metric("Nodes", st.session_state.nodes)
    col3.metric("Edges", st.session_state.edges)
    col4.metric("Attack Type", st.session_state.attack_type)
    st.caption("Attack type is inferred from dataset labels and not directly predicted by the model.")

    if st.session_state.prediction == "Attack":
        density_text = "Highly connected structure suggests coordinated attack behavior."
    else:
        density_text = "Low connectivity suggests normal background traffic."

    st.caption(
        f"Graph Density: {st.session_state.density} | {density_text}"
    )

    st.markdown("---")

    left_col, right_col = st.columns([0.95, 1.25], gap="large")

    with left_col:
        st.subheader("IDS Interpretation")

        if st.session_state.prediction == "Attack":
            st.write(
                "Multiple highly correlated alerts detected, indicating coordinated malicious activity."
            )
        else:
            st.write("Behavior is consistent with normal background traffic.")

    with right_col:
        st.subheader("Incident Correlation Graph")
        components.html(graph_html, height=560, scrolling=True)
        st.markdown(
            """
            <div class="graph-note">
                <strong>Red</strong> = suspicious nodes |
                <strong>Blue</strong> = normal traffic<br>
                Node colors are heuristic visual cues based on graph connectivity, not per-node model predictions.
                Edges indicate shared attributes such as IP, port, or temporal proximity.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Graph-based Explanation")

    explanation = (
        f"The incident graph contains **{st.session_state.nodes} alerts** "
        f"with density = {st.session_state.density} and average node degree of {avg_degree:.2f}. "
        f"Highly connected nodes indicate correlated activity, which the Graph-CNN associates with attack behavior."
    )

    st.write(explanation)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption("RANK Graph-CNN IDS | Final Year Project Demo")
