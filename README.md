readme = """#  Priora — Predictive Maintenance System

Sistema de detección de fallas y priorización de mantenimiento para motores eléctricos industriales, desarrollado en Azure Machine Learning.

---

##  Descripción

Priora identifica el tipo de motor eléctrico y predice si presentará una falla, asignando automáticamente un nivel de prioridad de mantenimiento. El sistema soporta tres tipos de motores industriales.

---

##  Tipos de Motor Soportados

| Tipo | Dataset | Registros |
|---|---|---|
| AC Trifásico | Figshare DOI 10.6084/m9.figshare.27216219 | 19,982 |
| DC Brushed | Zenodo DOI 10.5281/zenodo.4314249 | 377 |
| AC Monofásico | Generador sintético físico-informado | 2,100 |

---

##  Modelos

| Modelo | Objetivo | Accuracy | F1-Score |
|---|---|---|---|
| Random Forest | Identificación de tipo de motor | 100.00% | 100.00% |
| XGBoost + SMOTE | Detección de fallas | 99.98% | 99.99% |

---

##  Sistema de Priorización

| Probabilidad de Falla | Prioridad | Acción |
|---|---|---|
| 0% — 20% | 🟢 Normal | Motor operando correctamente |
| 20% — 50% | 🟡 Atención | Incrementar frecuencia de monitoreo |
| 50% — 80% | 🟠 Riesgo | Programar mantenimiento a corto plazo |
| 80% — 100% | 🔴 Crítico | Detener motor para inspección inmediata |

---

##  Pipeline
Datos de sensores
↓
Identificación del tipo de motor (Random Forest)
↓
Detección de falla (XGBoost + SMOTE)
↓
Score de probabilidad (0-100%)
↓
Prioridad de mantenimiento

---

##  Dataset Total

- **Total registros:** 22,459
- **Normales:** 4,678 (20.8%)
- **Fallas:** 17,781 (79.2%)

---

##  Tecnologías

- **Plataforma:** Azure Machine Learning
- **Lenguaje:** Python 3.10
- **Librerías:** scikit-learn, XGBoost, imbalanced-learn, pandas, numpy, h5py, joblib

---

##  Estructura del Proyecto
├── priora_maintenance_system.ipynb  # Notebook principal
├── priora_modelo_tipo.pkl           # Modelo identificador de tipo
├── priora_modelo_falla.pkl          # Modelo detector de fallas
├── priora_scaler.pkl                # Normalizador
├── priora_label_encoder.pkl         # Codificador de tipos
└── README.md                        # Este archivo

---

## 👤 Autor

**Yohaly De La Rosa S.**
Ingeniería Electrónica | Azure Machine Learning
github.com/yoh6ly/priora-maintenance-system

---

*Desarrollado con Python en Azure Machine Learning Studio*
"""

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme)
