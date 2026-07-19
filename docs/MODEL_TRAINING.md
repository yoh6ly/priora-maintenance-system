# Model Training Guide — PRIORA System

## 📖 Overview

This guide explains how to retrain the PRIORA models with your own data.

---

## 🗂️ Data Structure

### Required Columns

```python
columns = [
    'voltage',      # float: Volts (V)
    'current',      # float: Amps (A)
    'power',        # float: Watts (W)
    'temperature',  # float: Celsius (°C)
    'vibration',    # float: mm/s
    'motor_type',   # int: 0=AC3, 1=AC1, 2=DC
    'fault'         # int: 0=Normal, 1=Fault
]
```

### Sample Data Format

```csv
voltage,current,power,temperature,vibration,motor_type,fault
380,15.5,8500,68,4.2,0,0
380,18.2,9500,75,6.8,0,1
230,8.5,1800,62,3.2,1,0
```

---

## 📊 Data Preparation

### 1. Load Data

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import DataPreprocessor

# Load data
df = pd.read_csv('data/training_data.csv')

# Display info
print(df.info())
print(df.describe())
```

### 2. Data Cleaning

```python
# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
df = df.dropna()

# Remove outliers
from src.preprocessing import remove_outliers
df = remove_outliers(df, columns=['voltage', 'current', 'power', 'temperature', 'vibration'])
```

### 3. Feature Engineering

```python
# Create derived features if needed
df['power_factor'] = df['power'] / (df['voltage'] * df['current'])
df['temp_vibration_ratio'] = df['temperature'] / df['vibration']

# Select features
features = ['voltage', 'current', 'power', 'temperature', 'vibration']
X = df[features]
y = df['fault']
```

### 4. Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
```

---

## 🤖 Model Training

### Option 1: Using Jupyter Notebook

Open `notebooks/02_model_training.ipynb` for complete training pipeline.

### Option 2: Python Script

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

# Load and prepare data
df = pd.read_csv('data/training_data.csv')
X = df[['voltage', 'current', 'power', 'temperature', 'vibration']]
y = df['fault']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=True
)

# Evaluate
from sklearn.metrics import accuracy_score, f1_score, recall_score
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")

# Save model and scaler
joblib.dump(model, 'models/modelo_custom.pkl')
joblib.dump(scaler, 'models/scaler_custom.pkl')
```

---

## 🎯 Hyperparameter Tuning

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.6, 0.8, 1.0]
}

xgb = XGBClassifier(random_state=42)
grid_search = GridSearchCV(
    xgb, param_grid, 
    cv=5, 
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

---

## 📈 Model Evaluation

### Classification Metrics

```python
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    roc_curve,
    auc
)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Classification Report
print(classification_report(y_test, y_pred))

# ROC Curve
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
print(f"ROC AUC: {roc_auc:.4f}")
```

### Cross-Validation

```python
from sklearn.model_selection import cross_validate

cv_results = cross_validate(
    model, X_train_scaled, y_train,
    cv=5,
    scoring=['accuracy', 'f1', 'recall']
)

print(f"CV Accuracy: {cv_results['test_accuracy'].mean():.4f}")
print(f"CV F1: {cv_results['test_f1'].mean():.4f}")
```

---

## 🔄 Training Multiple Models

### Separate Models by Motor Type

```python
for motor_type in [0, 1, 2]:
    # Filter data by motor type
    df_motor = df[df['motor_type'] == motor_type]
    
    if len(df_motor) < 50:  # Skip if insufficient data
        continue
    
    X = df_motor[features]
    y = df_motor['fault']
    
    # Split, scale, train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = XGBClassifier()
    model.fit(X_train_scaled, y_train)
    
    # Save
    motor_names = {0: 'ac3', 1: 'ac1', 2: 'dc'}
    joblib.dump(model, f'models/modelo_{motor_names[motor_type]}_new.pkl')
    joblib.dump(scaler, f'models/scaler_{motor_names[motor_type]}_new.pkl')
```

---

## 💾 Save and Load Models

### Save Model

```python
import joblib

# Save trained model
joblib.dump(model, 'models/my_model.pkl')

# Save scaler
joblib.dump(scaler, 'models/my_scaler.pkl')
```

### Load Model

```python
# Load model
model = joblib.load('models/my_model.pkl')

# Load scaler
scaler = joblib.load('models/my_scaler.pkl')

# Use for prediction
predictions = model.predict(scaler.transform(X_new))
```

---

## 🧪 Testing Models

### Make Predictions

```python
# Single prediction
single_sample = [[380, 15.5, 8500, 68, 4.2]]
scaled = scaler.transform(single_sample)
prediction = model.predict(scaled)
probability = model.predict_proba(scaled)

print(f"Prediction: {prediction[0]}")
print(f"Probability: {probability[0]}")
```

### Batch Predictions

```python
# Batch prediction
X_new = pd.read_csv('data/new_motors.csv')[features]
X_new_scaled = scaler.transform(X_new)
predictions = model.predict(X_new_scaled)
probabilities = model.predict_proba(X_new_scaled)
```

---

## 🚀 Production Deployment

### Update Models in PRIORA

```bash
# After training, update the models in PRIORA
cp models/modelo_ac3_new.pkl models/modelo_ac3.pkl
cp models/scaler_ac3_new.pkl models/scaler_ac3.pkl

# Test with PRIORA
python examples/simple_prediction.py
```

---

## 📚 Best Practices

1. **Data Quality**: Ensure clean, representative data
2. **Balanced Dataset**: Balance fault vs. normal samples
3. **Feature Scaling**: Always scale before training
4. **Cross-Validation**: Use CV to validate generalization
5. **Monitor Performance**: Track metrics over time
6. **Version Control**: Keep track of model versions
7. **Documentation**: Document training parameters

---

## 🔗 Resources

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn Guide](https://scikit-learn.org/)
- [Jupyter Notebooks](../notebooks/)

