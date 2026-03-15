# 🌡️ PMSM Rotor Temperature Anomaly Detection

> A machine learning pipeline to detect anomalies in **Permanent Magnet Synchronous Motor (PMSM)** sensor data using **Isolation Forest** and **Decision Tree** models — designed for industrial predictive maintenance.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Methodology](#methodology)
- [Results](#results)
- [How to Run](#how-to-run)
- [Output Files](#output-files)
- [Key Insights](#key-insights)
- [Conclusion](#conclusion)
- [References](#references)

---

## 📖 Overview

This project performs **unsupervised anomaly detection** on PMSM motor sensor data. The goal is to identify abnormal operating conditions — such as unusual current, voltage, torque, or temperature readings — that could indicate potential motor faults.

The pipeline combines:
- **Isolation Forest** → Detects anomalies in unlabeled sensor data
- **Decision Tree** → Interprets and explains what causes anomalies (explainable AI)

This dual-model approach provides both **detection** and **explainability**, making it practical for industrial monitoring and predictive maintenance teams.

---

## 📂 Dataset

| Property | Detail |
|----------|--------|
| **Source** | [Kaggle – PMSM Regression Analysis](https://www.kaggle.com/code/jocelyndumlao/pmsm-regression-analysis) |
| **File** | `measures_v2.csv` |
| **Target Variable** | `pm` — Permanent Magnet (Rotor) Temperature |
| **Features** | Motor sensor readings: current (`i_d`, `i_q`), voltage (`u_d`, `u_q`), torque, speed, stator temperatures, etc. |

### Preprocessing Steps
- Dropped rows with `NaN` values (last row contained all nulls)
- Removed duplicate records
- Dropped `profile_id` (irrelevant identifier column)
- Moved target variable `pm` to the last column as `pm_target`
- Applied **StandardScaler** for feature normalization before modeling

---

## 🗂️ Project Structure
```
pmsm-anomaly-detection/
│
├── measures_v2.csv              # Raw input dataset
├── anomaly_detection.ipynb      # Main Jupyter Notebook
├── detected_anomalies.csv       # Output: records flagged as anomalies
└── README.md                    # Project documentation
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|--------|---------|
| `pandas` | Data loading, cleaning, manipulation |
| `matplotlib` | Plotting and visualization |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | Isolation Forest, Decision Tree, scaling, evaluation |

**Python Version:** 3.8+

Install dependencies:
```bash
pip install pandas matplotlib seaborn scikit-learn
```

---

## 🔬 Methodology

### 1. 📊 Exploratory Data Analysis (EDA)
- Distribution plots of rotor temperature (`pm_target`) with mean and median overlays
- Box plots for all features to detect outlier patterns
- **Correlation heatmap** to assess feature relationships with the target variable

### 2. ⚙️ Feature Engineering
- Correlation matrix analysis to verify feature relevance
- StandardScaler applied to all features before model training

### 3. 🌲 Model A — Isolation Forest (Anomaly Detection)
```python
IsolationForest(n_estimators=100, contamination=0.2, random_state=42)
```

- Trained on the scaled dataset (unsupervised)
- `contamination=0.2` → assumes ~20% of data may be anomalous
- Predictions: `1` = Normal, `-1` = Anomaly (remapped to `0` and `1`)
- Detected anomalies exported to `detected_anomalies.csv`

### 4. 🌳 Model B — Decision Tree Classifier (Explainability)
```python
DecisionTreeClassifier(max_depth=5, random_state=42)
```

- Uses Isolation Forest labels as the target (`anomally`)
- 80/20 train-test split
- Evaluated via `classification_report`
- Visualized the full decision tree
- **Feature importance** extracted to identify key anomaly drivers

---

## 📈 Results

### Anomaly Detection (Isolation Forest)
- Anomalies are **not** limited to extreme temperature spikes
- Red points appear across the full time index, indicating anomalies involve **unusual combinations** of sensor readings — not just single-feature outliers

### Decision Tree — Feature Importance

| Feature | Importance |
|---------|------------|
| `torque` | ⭐ Highest |
| `stator_motor` (stator temperature) | ⭐ High |
| Other sensor features | Lower |

> **Key Finding:** Abnormal torque and stator temperature readings are the primary drivers of detected anomalies.

### Sample Visualization

The anomaly scatter plot highlights:
- 🔵 **Blue line** → Normal operating temperature readings
- 🔴 **Red points** → Flagged anomalies across the time index

---

## ▶️ How to Run

1. **Clone the repository**
```bash
   git clone https://github.com/your-username/pmsm-anomaly-detection.git
   cd pmsm-anomaly-detection
```

2. **Install dependencies**
```bash
   pip install pandas matplotlib seaborn scikit-learn
```

3. **Add the dataset**
   - Download `measures_v2.csv` from [Kaggle](https://www.kaggle.com/code/jocelyndumlao/pmsm-regression-analysis)
   - Place it in the root project directory

4. **Run the notebook**
```bash
   jupyter notebook anomaly_detection.ipynb
```

5. **Check the output**
   - Anomaly records will be saved to `detected_anomalies.csv`

---

## 📤 Output Files

| File | Description |
|------|-------------|
| `detected_anomalies.csv` | All data records flagged as anomalies by Isolation Forest |

This file can be handed to **maintenance teams** for targeted inspection of specific motor operating conditions.

---

## 💡 Key Insights

- Anomalies in PMSM motors are **multi-dimensional** — they arise from unusual combinations of current, voltage, torque, and temperature, not just single sensor spikes.
- **Isolation Forest** effectively detects these patterns without requiring labeled training data.
- **Decision Tree** adds interpretability by revealing *which* features are responsible for anomalous behavior.
- `torque` and `stator_motor` temperature are the **most influential features** in explaining detected anomalies.

---

## 🧾 Conclusion

This project demonstrates a practical **two-stage ML pipeline** for industrial motor monitoring:

1. **Isolation Forest** detects anomalies unsupervised across the full sensor feature space.
2. **Decision Tree** interprets those anomalies, providing explainable rules for maintenance teams.

The combination enables both **automated fault detection** and **human-understandable diagnosis**, making it well-suited for real-world predictive maintenance applications.

---

## 📚 References

- Dataset: [PMSM Temperature Dataset – Kaggle](https://www.kaggle.com/code/jocelyndumlao/pmsm-regression-analysis)
- Scikit-learn: [IsolationForest Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- Scikit-learn: [DecisionTreeClassifier Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

---

## 🪪 License

This project is open-source and available under the [MIT License](LICENSE).

---

*Built for industrial predictive maintenance — detecting what the eye can't see.*
