Priora — Predictive Maintenance System

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
=================================================================
         PRIORA v2 — RESUMEN FINAL
=================================================================

  DATASETS
   AC Trifásico:  12,000 registros (9,661 reales + 2,339 sint.)
   DC Brushed:    377 registros (reales — Zenodo)
   AC Monofásico: 2,200 registros (sintéticos físico-informados)
   Total:         14,577 registros

  MODELOS INDEPENDIENTES POR TIPO
Tipo                 Modelo                   Accuracy       F1   Recall
----------------------------------------------------------------------
AC Trifásico         XGBoost (umbral 26%)       96.12%   90.44%   94.02%
DC Brushed           XGBoost                   100.00%  100.00%  100.00%
AC Monofásico        XGBoost                   100.00%  100.00%  100.00%
Clasificador tipo    Random Forest             100.00%  100.00%  100.00%

 SISTEMA DE PRIORIZACIÓN (AC Trifásico)
   🟢 Normal   (0-20%):  1,872 motores
   🟡 Atención (20-50%):    70 motores
   🟠 Riesgo   (50-80%):    38 motores
   🔴 Crítico  (80-100%):  420 motores

  CORRECCIONES APLICADAS vs v1
   Modelos independientes por tipo (elimina sesgo)
   Fallas sintéticas graduales con superposición
   Umbral ajustado a 26% (mejor recall)
   Comparación de 3 modelos (XGBoost ganador)
   F1 AC Trifásico: 72.48% → 90.44% (+17.96%)
   Recall AC Trifásico: 79.41% → 94.02% (+14.61%)

 MODELOS GUARDADOS
   clf_tipo.pkl — 106.9 KB
   modelo_ac1.pkl — 152.2 KB
   modelo_ac3.pkl — 492.4 KB
   modelo_dc.pkl — 155.9 KB
   scaler_ac1.pkl — 0.7 KB
   scaler_ac3.pkl — 0.7 KB
   scaler_dc.pkl — 0.8 KB
   scaler_tipo.pkl — 0.7 KB

===========================================================
 PRIORA v2
===========================================================                      

# Este archivo

---

## 👤 Autor

**Yohaly De La Rosa S.**
Ingeniería Electrónica | Machine Learning
github.com/yoh6ly/priora-maintenance-system

---

*Desarrollado con Python en Azure Machine Learning Studio y google colab*
