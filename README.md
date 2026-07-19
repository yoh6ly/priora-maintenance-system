# 🔧 PRIORA — Predictive Maintenance System

<div align="center">

![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Azure ML](https://img.shields.io/badge/Azure%20ML-0089D6?style=for-the-badge&logo=microsoft-azure&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost-FF6B35?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-F37726?style=for-the-badge&logo=jupyter&logoColor=white)

**Advanced Predictive Maintenance System for Industrial Electric Motors**

[Overview](#-overview) • [Features](#-features) • [Results](#-results) • [Installation](#-installation) • [Usage](#-usage) • [Documentation](#-documentation)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Results & Performance](#-results--performance)
- [Datasets](#-datasets)
- [Installation](#-installation)
- [Usage](#-usage)
- [File Structure](#-file-structure)
- [Documentation](#-documentation)
- [Author](#-author)

---

## 🎯 Overview

**PRIORA** is an intelligent predictive maintenance system designed to detect faults and prioritize maintenance for industrial electric motors. The system automatically identifies motor type and predicts failure probability, assigning maintenance priority levels.

### **Key Capabilities:**
- 🔍 **Motor Type Classification** — Identifies AC 3-phase, AC single-phase, and DC brushed motors
- ⚠️ **Fault Prediction** — Detects potential failures before they occur
- 📊 **Priority Assignment** — Automatically assigns maintenance priority levels (Normal → Critical)
- ⚡ **High Accuracy** — 96-100% accuracy across motor types

---

## ✨ Features

| Feature | Description | Status |
|---------|-------------|--------|
| **Multi-Motor Support** | AC 3-phase, AC single-phase, DC brushed | ✅ |
| **Machine Learning Models** | Independent XGBoost models per motor type | ✅ |
| **Priority System** | 4-level maintenance prioritization | ✅ |
| **Data Preprocessing** | Automated feature scaling and normalization | ✅ |
| **Model Persistence** | Serialized models for production deployment | ✅ |
| **Azure ML Integration** | Cloud-based training and inference | ✅ |

---

## 🏗️ System Architecture

```
PRIORA System Architecture
│
├── Input: Motor Characteristics
│   ├── Voltage (V)
│   ├── Current (A)
│   ├── Power (W)
│   ├── Temperature (°C)
│   └── Vibration (mm/s)
│
├── Processing Pipeline
│   ├── 1️⃣ Motor Type Classification (Random Forest)
│   ├── 2️⃣ Feature Scaling (StandardScaler)
│   └── 3️⃣ Fault Detection (XGBoost)
│
└── Output: Maintenance Decision
    ├── Motor Type
    ├── Fault Probability (%)
    ├── Priority Level (🟢🟡🟠🔴)
    └── Recommended Action
```

---

## 📊 Results & Performance

### **Model Performance by Motor Type**

| Motor Type | Model | Accuracy | F1-Score | Recall | Precision |
|-----------|-------|----------|----------|--------|-----------|
| **AC 3-Phase** | XGBoost (26% threshold) | **96.12%** | **90.44%** | **94.02%** | 87.15% |
| **DC Brushed** | XGBoost | **100.00%** | **100.00%** | **100.00%** | 100.00% |
| **AC Single-Phase** | XGBoost | **100.00%** | **100.00%** | **100.00%** | 100.00% |
| **Motor Classifier** | Random Forest | **100.00%** | **100.00%** | **100.00%** | 100.00% |

### **Maintenance Priority Distribution (AC 3-Phase)**

```
🟢 Normal    (0-20%)   [████████████████████] 1,872 motors (78.5%)
🟡 Atención  (20-50%)  [██]                     70 motors (2.9%)
🟠 Riesgo    (50-80%)  [█]                      38 motors (1.6%)
🔴 Crítico   (80-100%) [██████]                420 motors (17.6%)
```

### **Improvements vs v1**

| Metric | v1 | v2 | Improvement |
|--------|----|----|-------------|
| **F1-Score (AC 3-Phase)** | 72.48% | 90.44% | **+17.96%** ⬆️ |
| **Recall (AC 3-Phase)** | 79.41% | 94.02% | **+14.61%** ⬆️ |
| **Model Type** | Combined | Independent | Better per-type accuracy |
| **Threshold** | Auto | 26% (Optimized) | Improved sensitivity |

---

## 📦 Datasets

### **Data Summary**

| Motor Type | Records | Source | Characteristics |
|-----------|---------|--------|-----------------|
| **AC 3-Phase** | 12,000 | 9,661 real + 2,339 synthetic | Industrial datasets |
| **DC Brushed** | 377 | Zenodo (Real data) | Commutator-based motors |
| **AC Single-Phase** | 2,200 | Physics-informed synthetic | Home/small appliances |
| **TOTAL** | **14,577** | Mixed sources | Comprehensive coverage |

### **Features Used**
- Voltage, Current, Power consumption
- Temperature, Vibration levels
- Operating frequency, Load percentage
- Historical fault patterns

---

## 🛠️ Installation

### **Requirements**
- Python 3.8+
- pandas
- scikit-learn
- xgboost
- numpy
- joblib

### **Setup**

```bash
# Clone the repository
git clone https://github.com/yoh6ly/priora-maintenance-system.git
cd priora-maintenance-system

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **Azure ML Setup (Optional)**

```bash
# Install Azure ML SDK
pip install azureml-sdk

# Configure your Azure ML workspace
# Update config.json with your workspace details
```

---

## 🚀 Usage

### **Quick Start - Predict Motor Maintenance Priority**

```python
import joblib
import pandas as pd

# Load pre-trained models
clf_tipo = joblib.load('models/clf_tipo.pkl')
modelo_ac3 = joblib.load('models/modelo_ac3.pkl')
scaler_ac3 = joblib.load('models/scaler_ac3.pkl')

# Motor data (example)
motor_data = {
    'voltage': 380,
    'current': 15.5,
    'power': 8500,
    'temperature': 68,
    'vibration': 4.2
}

# Convert to DataFrame
df = pd.DataFrame([motor_data])

# 1. Classify motor type
motor_type = clf_tipo.predict(df)[0]
print(f"Motor Type: {motor_type}")

# 2. Scale features
df_scaled = scaler_ac3.transform(df)

# 3. Predict fault probability
fault_prob = modelo_ac3.predict_proba(df_scaled)[0][1] * 100

# 4. Assign priority
if fault_prob < 20:
    priority = "🟢 Normal"
elif fault_prob < 50:
    priority = "🟡 Atención"
elif fault_prob < 80:
    priority = "🟠 Riesgo"
else:
    priority = "🔴 Crítico"

print(f"Fault Probability: {fault_prob:.2f}%")
print(f"Priority Level: {priority}")
```

### **Batch Processing**

```python
import pandas as pd
from priora import PrioraSystem

# Load system
priora = PrioraSystem()

# Load motor data
motors_df = pd.read_csv('data/motors.csv')

# Get predictions for all motors
results = priora.predict_batch(motors_df)

# Export results
results.to_csv('output/maintenance_report.csv', index=False)
```

---

## 📁 File Structure

```
priora-maintenance-system/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
│
├── 📁 models/                        # Pre-trained ML models
│   ├── clf_tipo.pkl                 # Motor type classifier (Random Forest)
│   ├── modelo_ac3.pkl               # AC 3-phase fault detection (XGBoost)
│   ├── modelo_ac1.pkl               # AC single-phase fault detection
│   ├── modelo_dc.pkl                # DC brushed fault detection
│   ├── scaler_ac3.pkl               # Feature scaler (AC 3-phase)
│   ├── scaler_ac1.pkl               # Feature scaler (AC single-phase)
│   ├── scaler_dc.pkl                # Feature scaler (DC brushed)
│   └── scaler_tipo.pkl              # Type classifier scaler
│
├── 📁 notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb     # Data analysis & visualization
│   ├── 02_model_training.ipynb       # Model training & tuning
│   ├── 03_model_evaluation.ipynb     # Performance metrics & analysis
│   └── 04_deployment_guide.ipynb     # Production deployment
│
├── 📁 data/                          # Datasets
│   ├── raw/                          # Original datasets
│   │   ├── ac_trifasico.csv
│   │   ├── dc_brushed.csv
│   │   └── ac_monofasico.csv
│   ├── processed/                    # Preprocessed datasets
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── validation.csv
│   └── sample/                       # Sample data for testing
│       └── sample_motors.csv
│
├── 📁 src/                           # Source code
│   ├── __init__.py
│   ├── priora.py                     # Main PRIORA system class
│   ├── preprocessing.py              # Data preprocessing utilities
│   ├── models.py                     # Model loading & inference
│   ├── utils.py                      # Helper functions
│   └── config.py                     # Configuration settings
│
├── 📁 examples/                      # Usage examples
│   ├── simple_prediction.py          # Basic prediction example
│   ├── batch_processing.py           # Batch prediction example
│   ├── api_demo.py                   # API implementation example
│   └── visualization.py              # Results visualization
│
├── 📁 tests/                         # Unit tests
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_predictions.py
│
├── 📁 docs/                          # Additional documentation
│   ├── INSTALLATION.md               # Detailed installation guide
│   ├── API_REFERENCE.md              # API documentation
│   ├── MODEL_TRAINING.md             # Model training guide
│   └── DEPLOYMENT.md                 # Production deployment guide
│
└── 📁 results/                       # Output directory
    ├── predictions/
    ├── reports/
    └── visualizations/
```

---

## 📖 Documentation

### **Quick Links**

- **[Installation Guide](docs/INSTALLATION.md)** — Detailed setup instructions
- **[API Reference](docs/API_REFERENCE.md)** — Complete API documentation
- **[Model Training](docs/MODEL_TRAINING.md)** — How to retrain models
- **[Deployment Guide](docs/DEPLOYMENT.md)** — Production deployment
- **[Examples](examples/)** — Code examples and use cases

### **Key Documentation Files**

| File | Purpose |
|------|---------|
| `INSTALLATION.md` | Setup and configuration |
| `API_REFERENCE.md` | Complete API documentation |
| `MODEL_TRAINING.md` | Training pipeline details |
| `DEPLOYMENT.md` | Production deployment |

---

## 🔍 Model Details

### **Motor Type Classifier**
- **Model:** Random Forest
- **Accuracy:** 100%
- **Input:** 5 motor characteristics
- **Output:** Motor type (AC 3-phase, AC single-phase, DC)

### **Fault Detection Models**

#### AC 3-Phase Motor
- **Model:** XGBoost with 26% probability threshold
- **Accuracy:** 96.12%
- **F1-Score:** 90.44%
- **Use Case:** Industrial 3-phase motors

#### DC Brushed Motor
- **Model:** XGBoost
- **Accuracy:** 100%
- **F1-Score:** 100%
- **Use Case:** DC commutator-based motors

#### AC Single-Phase Motor
- **Model:** XGBoost
- **Accuracy:** 100%
- **F1-Score:** 100%
- **Use Case:** Home appliances and small motors

---

## 🎓 Learning Resources

- 📚 [XGBoost Documentation](https://xgboost.readthedocs.io/)
- 📚 [Scikit-learn Guide](https://scikit-learn.org/)
- 📚 [Predictive Maintenance Best Practices](https://docs.microsoft.com/en-us/azure/machine-learning/)

---

## 📈 Future Improvements

- [ ] Real-time API endpoint
- [ ] Web dashboard for monitoring
- [ ] Additional motor types
- [ ] LSTM/RNN time-series models
- [ ] Automated retraining pipeline
- [ ] Mobile app integration
- [ ] IoT sensor integration

---

## 📝 License

This project is open source and available under the MIT License.

---

## 👤 Author

**Yohaly De La Rosa S.**

- 🎓 **Electronics Engineering Student**
- 🤖 **Machine Learning Enthusiast**
- 📧 **Email:** yohaly982@gmail.com
- 🔗 **GitHub:** [@yoh6ly](https://github.com/yoh6ly)
- 💼 **LinkedIn:** [Yohaly De La Rosa](https://www.linkedin.com/in/yohaly-d-2005ba308)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## ⭐ Show Your Support

If you found this project helpful, please give it a star! It helps others discover the project.

---

<div align="center">

**Made with ❤️ using Python, XGBoost, and Azure ML**

*Last Updated: July 2026*

</div>
