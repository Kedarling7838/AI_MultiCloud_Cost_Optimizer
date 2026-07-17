# 🚀 AI-Driven Framework for Workload Placement and Cost–Performance Optimization in Multi-Cloud Systems

An intelligent machine learning framework that predicts cloud workload requirements using **XGBoost** and **Random Forest** and recommends the optimal cloud provider based on **cost**, **latency**, and **availability**.

---

## 📌 Overview

Cloud computing has become the backbone of modern applications; however, selecting the most suitable cloud provider while maintaining performance and minimizing operational cost remains a challenge. This project presents an AI-driven framework that predicts workload requirements and performs intelligent cloud provider selection using machine learning and multi-objective optimization.

The proposed framework combines workload prediction with cloud placement to improve resource utilization and support cost-effective deployment decisions.

---

## ✨ Features

* 📊 Workload Prediction using XGBoost and Random Forest
* ☁️ Intelligent Multi-Cloud Provider Selection
* 💰 Cost–Performance Optimization
* ⚡ Latency and Availability Aware Scheduling
* 📈 Automatic Performance Evaluation
* 📉 Research Paper Quality Visualizations
* 📁 CSV Result Generation
* 🔍 Feature Importance Analysis

---

## 🛠️ Technology Stack

| Category             | Technology             |
| -------------------- | ---------------------- |
| Programming Language | Python 3.x             |
| Machine Learning     | XGBoost, Random Forest |
| Data Processing      | Pandas, NumPy          |
| Visualization        | Matplotlib             |
| Model Evaluation     | Scikit-learn           |
| IDE                  | Visual Studio Code     |

---

## 📂 Dataset

The project uses four datasets:

1. **Batch Task Dataset**

   * Batch workload information
   * CPU requirements
   * Memory requirements

2. **Machine Metadata**

   * CPU cores
   * Memory capacity
   * Disk size
   * Machine status

3. **Cloud Pricing Dataset**

   * AWS pricing
   * Azure pricing
   * GCP pricing

4. **Latency & Availability Dataset**

   * Network latency
   * Service availability
   * Provider information

---

## 📁 Project Structure

```text
AI_MultiCloud_Project/
│
├── datasets/
│   ├── batch_task_20000_with_headers.csv
│   ├── machine_meta_with_headers.csv
│   ├── cloud_pricing_large.csv
│   └── latency_availability_large_5000.csv
│
├── results/
│   ├── graphs/
│   ├── model_results.csv
│   ├── predictions.csv
│   └── cloud_provider_ranking.csv
│
├── preprocessing.py
├── feature_engineering.py
├── model.py
├── evaluation.py
├── cloud_selection.py
├── visualization.py
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/AI_MultiCloud_Project.git

cd AI_MultiCloud_Project
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Execute:

```bash
python main.py
```

---

## 🔄 Workflow

```text
Datasets
     │
     ▼
Data Preprocessing
     │
     ▼
Feature Engineering
     │
     ▼
Train XGBoost & Random Forest
     │
     ▼
Performance Evaluation
     │
     ▼
Cloud Placement Algorithm
     │
     ▼
Recommended Cloud Provider
     │
     ▼
Graphs & Reports
```

---

## 📊 Evaluation Metrics

The models are evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score
* Mean Absolute Percentage Error (MAPE)

---

## 📈 Generated Outputs

After execution, the following files are generated:

```text
results/
│
├── model_results.csv
├── predictions.csv
├── cloud_provider_ranking.csv
│
└── graphs/
    ├── feature_importance.png
    ├── actual_vs_predicted.png
    ├── rmse_comparison.png
    ├── cpu_distribution.png
    ├── correlation_matrix.png
    └── cloud_cost_comparison.png
```

---

## 🧠 Machine Learning Models

* XGBoost Regressor
* Random Forest Regressor

The models predict workload demand based on machine configuration and task characteristics.

---

## ☁️ Cloud Placement Strategy

The cloud provider is selected using the scoring function:

```text
Provider Score =
0.5 × Cost +
0.3 × Latency +
0.2 × (100 − Availability)
```

The provider with the lowest score is recommended for workload deployment.

---

## 📌 Applications

* Cloud Resource Scheduling
* Multi-Cloud Optimization
* Intelligent Workload Placement
* Cost Optimization
* Resource Provisioning
* Cloud Infrastructure Planning

---

## 📚 Research Paper

**Title:**

**An AI-Driven Framework for Workload Placement and Cost–Performance Optimization in Multi-Cloud Systems**

---

## 🔮 Future Enhancements

* Deep Learning (LSTM, GRU, Transformer)
* Reinforcement Learning-based Scheduling
* Kubernetes Integration
* Real-Time Cloud Monitoring
* Hybrid Cloud Support
* Edge and Fog Computing
* Carbon-Aware Resource Scheduling

---

## 👨‍💻 Author

**Kedarling Ashok Kanade**

Master of Computer Applications (MCA)

GM University, Davangere

---

## 📄 License

This project is developed for academic and research purposes. Feel free to use and extend it with appropriate attribution.

