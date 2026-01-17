# 🔐 RANK: Graph-CNN Based Intrusion Detection System

RANK is an **incident-level Intrusion Detection System (IDS)** that models network traffic as graphs and applies a **Graph Convolutional Neural Network (Graph-CNN)** to detect coordinated attack behavior.  
Unlike traditional flow-based IDS approaches, RANK captures **relationships between alerts** using graph structures, improving detection of multi-stage and correlated attacks.

---

## 📖 Project Motivation

Conventional intrusion detection systems analyze network flows independently, which limits their ability to detect **complex, multi-step attacks** such as reconnaissance followed by exploitation.  
This project addresses that limitation by:
<<<<<<< HEAD

- Representing alerts as **incident correlation graphs**
- Learning structural attack patterns using **Graph Neural Networks**
- Providing **visual explanations** to support analyst interpretation

---

## 🏗️ System Architecture

<p align="center">
  <img src="" width="750"/>
</p>

**Pipeline Overview:**

1. Network traffic datasets (UNSW-NB15 / DARPA) are preprocessed
2. Alerts are grouped into sliding windows
3. Incident graphs are constructed using shared IP, port, and time correlations
4. A Graph-CNN classifies each incident as **Attack** or **Normal**
5. Results are served through a **FastAPI backend**
6. A **Streamlit dashboard** visualizes predictions and correlation graphs

---

## 🧠 Graph-CNN Model Design

<p align="center">
  <img src="docs/graphcnn_architecture.png" width="450"/>
</p>

**Model Components:**

- Two GCN layers with ReLU activation
- Global pooling for graph-level embedding
- Fully connected classification head
- Binary output: `Attack` / `Normal`

The model is implemented using **PyTorch Geometric**.

---

## 📊 Datasets Used

| Dataset | Description |
|------|------------|
| UNSW-NB15 | Modern synthetic intrusion dataset |
| DARPA | Benchmark intrusion detection dataset |

Preprocessed CSV files are stored in the `data/` directory.

---

## 🚀 How to Run the Project

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```
2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
3️⃣ Preprocess Dataset
```bash
python preprocessing/prepare_features.py
```
4️⃣ Train Graph-CNN
```bash
python -m training.train_graphcnn
```
5️⃣ Start Backend API
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
6️⃣ Launch Dashboard
```bash
streamlit run ui/streamlit_app.py
```
Open browser at:
👉 http://localhost:8501

🖥️ Streamlit Dashboard
<p align="center"> <img src="" width="750"/> </p>
Dashboard Features:

One-click IDS inference

Incident-level confidence score

Interactive correlation graph

Graph-based explanation panel

🔍 Graph Visualization Explanation
🔴 Red Nodes → Alerts contributing strongly to attack prediction

🔵 Blue Nodes → Normal or low-risk alerts

Dense connectivity indicates coordinated attack behavior

📈 Experimental Results
Metric	Value
Accuracy	98.8%
Precision	0.99
Recall	1.00
F1-Score	0.99

Results demonstrate strong performance on incident-level detection.

🧪 Technologies Used
Python 3.9
PyTorch
PyTorch Geometric
FastAPI
Streamlit
NetworkX
PyVis
Scikit-learn

📂 Project Structure
RANK-GraphCNN/
│
├── backend/          # FastAPI server
├── preprocessing/    # Feature extraction scripts
├── graph/            # Graph construction logic
├── models/           # Graph-CNN model
├── training/         # Training and evaluation
├── ui/               # Streamlit dashboard
├── data/             # Datasets (ignored in Git)
├── notebooks/        # Research notebooks
├── dashboard/             # Architecture & result images
└── README.md
🎓 Academic Context
📜 License


=======

- Representing alerts as **incident correlation graphs**
- Learning structural attack patterns using **Graph Neural Networks**
- Providing **visual explanations** to support analyst interpretation

---

## 🏗️ System Architecture

<p align="center">
  <img src="dashboard/Architecture diagram.png" width="750"/>
</p>

**Pipeline Overview:**

1. Network traffic datasets (UNSW-NB15 / DARPA) are preprocessed
2. Alerts are grouped into sliding windows
3. Incident graphs are constructed using shared IP, port, and time correlations
4. A Graph-CNN classifies each incident as **Attack** or **Normal**
5. Results are served through a **FastAPI backend**
6. A **Streamlit dashboard** visualizes predictions and correlation graphs

---

## 🧠 Graph-CNN Model Design

<p align="center">
  <img src="dashboard\Graph flowchart.png" width="450"/>
</p>

**Model Components:**

- Two GCN layers with ReLU activation
- Global pooling for graph-level embedding
- Fully connected classification head
- Binary output: `Attack` / `Normal`

The model is implemented using **PyTorch Geometric**.

---

## 📊 Datasets Used

| Dataset | Description |
|------|------------|
| UNSW-NB15 | Modern synthetic intrusion dataset |
| DARPA | Benchmark intrusion detection dataset |

Preprocessed CSV files are stored in the `data/` directory.

---

## 🚀 How to Run the Project

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```
2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
3️⃣ Preprocess Dataset
```bash
python preprocessing/prepare_features.py
```
4️⃣ Train Graph-CNN
```bash
python -m training.train_graphcnn
```
5️⃣ Start Backend API
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
6️⃣ Launch Dashboard
```bash
streamlit run ui/streamlit_app.py
```
Open browser at:
👉 http://localhost:8501

🖥️ Streamlit Dashboard
<p align="center"> <img src="dashboard\output.png" width="750"/> </p>
Dashboard Features:

One-click IDS inference

Incident-level confidence score

Interactive correlation graph

Graph-based explanation panel

🔍 Graph Visualization Explanation
🔴 Red Nodes → Alerts contributing strongly to attack prediction

🔵 Blue Nodes → Normal or low-risk alerts

Dense connectivity indicates coordinated attack behavior

📈 Experimental Results

Metric	Value
Accuracy	98.8%
Precision	0.99
Recall	1.00
F1-Score	0.99

Results demonstrate strong performance on incident-level detection.

🧪 Technologies Used

Python 3.9
PyTorch
PyTorch Geometric
FastAPI
Streamlit
NetworkX
PyVis
Scikit-learn

📂 Project Structure

RANK-GraphCNN/
│
├── backend/          # FastAPI server
├── preprocessing/    # Feature extraction scripts
├── graph/            # Graph construction logic
├── models/           # Graph-CNN model
├── training/         # Training and evaluation
├── ui/               # Streamlit dashboard
├── data/             # Datasets (ignored in Git)
├── notebooks/        # Research notebooks
├── dashboard/             # Architecture & result images
└── README.md

🎓 Academic Context

This project was developed as a Final Year B.Tech Computer Science project
and is suitable for:
Academic evaluation
Research publication
IDS prototyping demonstrations

📜 License

This project is for academic and research use only.
>>>>>>> 4cd91fc (Initial release: Graph-CNN based IDS with dashboard)
