# 🔐 RANK: Graph-Based Intrusion Detection using Graph-CNN

🚀 A Graph Neural Network (GNN) based Intrusion Detection System that models network traffic as graphs and detects coordinated cyber-attacks using Graph Convolutional Networks (Graph-CNN).

---

## 📌 Project Overview

Traditional Intrusion Detection Systems (IDS) analyze network traffic as independent records, often missing correlated attack patterns.

This project introduces a **Graph-Based Intrusion Detection System (RANK)** that:
- Represents network alerts as **graphs**
- Captures relationships between alerts (IP, port, time)
- Uses **Graph Convolutional Networks (GCN)** for classification
- Detects **coordinated attacks at incident level**

---

## 🎯 Key Features

- ✅ Graph-based anomaly detection
- ✅ Incident-level attack classification (not just row-wise)
- ✅ Interactive Streamlit dashboard
- ✅ Real-time API-based prediction (FastAPI)
- ✅ Graph visualization of network behavior
- ✅ Attack vs Normal classification
- ✅ Attack type interpretation (DoS, Exploits, etc.)

---

## 🧠 How It Works

### Pipeline:

1. Load network dataset (UNSW-NB15 / DARPA)
2. Preprocess data (cleaning, encoding)
3. Create sliding windows (incident grouping)
4. Construct graph:
   - Nodes → Alerts
   - Edges → Shared IP / Port / Time
5. Apply Graph-CNN model
6. Predict:
   - Attack / Normal
   - Confidence score
7. Visualize graph + explanation in dashboard

---

## 🏗️ System Architecture

![Architecture](https://github.com/bhanuyadav66/RANK-GraphCNN-IDS/blob/main/dashoard/system%20architecture.png)

---

## 🔄 Workflow

![Workflow](dashboard/Flowchart.png)

---

## 🕸️ Graph Representation

![Graph Representation](dashboard/Graph%20flowchart.png)

---

## 📊 Dashboard Output

![Output](dashboard/output.png)

---

## 🛠️ Tech Stack

| Category        | Technology |
|----------------|-----------|
| Programming    | Python |
| ML Framework   | PyTorch |
| Graph ML       | PyTorch Geometric |
| Backend API    | FastAPI |
| Frontend       | Streamlit |
| Visualization  | PyVis |
| Data Handling  | Pandas, NumPy |

---

## 📂 Project Structure

RANK-GraphCNN/


├── backend/

│ └── app.py # FastAPI backend


├── dashboard/

│ ├── streamlit_app.py # UI dashboard

│ ├── Architecture diagram.png

│ ├── Flowchart.png

│ ├── Graph flowchart.png

│ └── output.png


├── graph/
│ └── graph_builder.py # Graph creation logic


├── model/
│ └── graph_model.py # Graph-CNN model


├── data/
│ └── processed_data.csv # Dataset


├── notebooks/

│ └── research_notebook.ipynb


├── requirements.txt

└── README.md

---


---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/bhanuyadav66/RANK-GraphCNN-IDS.git
cd RANK-GraphCNN-IDS
```

### 2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt
🚀 Running the Project
▶️ Start Backend (FastAPI)
uvicorn backend.app:app --reload

👉 Runs at: http://localhost:8000

▶️ Start Dashboard (Streamlit)
streamlit run ui/streamlit_app.py

👉 Runs at: http://localhost:8501

▶️ Run Demo
Open dashboard
Click "Run IDS on Sample Incident"
View:
  Prediction
  Graph visualization
  Explanation

📈 Model Output
Prediction: Attack / Normal
Confidence Score: Probability of prediction
Graph Metrics: Nodes, Edges, Density
Attack Type: Derived from dataset labels

⚠️ Important Note

The Graph-CNN model performs binary classification (Attack vs Normal).
Attack type labels (DoS, Exploit, etc.) are derived from dataset categories for interpretability.

📊 Dataset Used
UNSW-NB15 Dataset
DARPA Intrusion Detection Dataset

📌 Applications
Network Security Monitoring
Enterprise Threat Detection
SOC (Security Operations Center)
Cyber Attack Analysis

🔮 Future Improvements
Multi-class attack classification
Real-time streaming IDS
Explainable AI (XAI) integration
Deployment on cloud (AWS/GCP)
Integration with SIEM tools

👨‍💻 Author

Allam Bhanu Yadav
Final Year CSE Student
Specialization: AI / ML / Data Science

📜 License

This project is for academic and research purposes.
