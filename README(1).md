# 🔧 PRIORA — Predictive Maintenance System for Industrial Electric Motors

Sistema de mantenimiento predictivo que integra instrumentación electrónica,
modelos de Machine Learning auditados con rigor estadístico, y una plataforma
de software segura para diagnosticar en tiempo real el estado de motores
eléctricos industriales.

> **Nota:** este README describe la versión final del proyecto (v3), que
> reduce el alcance de 3 a 2 tipos de motor (AC Trifásico y DC Brushed) para
> priorizar la calidad de los datasets reales utilizados, y agrega hardware
> funcional, servidor con ciberseguridad, y documentación nivel tesis.

---

## 📌 Descripción

Priora identifica el tipo de motor bajo diagnóstico y la naturaleza de la
señal física disponible (proceso, vibración o corriente), aplica el modelo
XGBoost correspondiente, y traduce la probabilidad de falla resultante en
una acción de mantenimiento concreta con cuatro niveles de prioridad.

## ⚡ Motores y modelos soportados

| Motor / Señal | Dataset | Registros | F1 (validación cruzada) | Umbral |
|---|---|---|---|---|
| AC Trifásico — Proceso | AI4I 2020 (UCI) | 10,000 | 70.12% (±0.83%) | 0.50 |
| AC Trifásico — Vibración | NASA IMS Bearing | 984 | 99.45% (±0.00%) | 0.30 |
| DC Brushed | Open DC Motor (Zenodo) | 377 | 98.86% (±1.05%) | 0.30 |

**Todos los modelos son XGBoost**, seleccionado tras comparación empírica
contra Árbol de Decisión y Random Forest, y validado con validación cruzada
de 5 folds replicando el pipeline completo (SMOTE + optimización de umbral)
dentro de cada partición — ver sección de metodología más abajo.

---

## 🎯 Sistema de priorización

| Nivel | Probabilidad | Acción |
|---|---|---|
| 🟢 Normal | 0% – 20% | Continuar operación |
| 🟡 Atención | 20% – 50% | Incrementar frecuencia de monitoreo |
| 🟠 Riesgo | 50% – 80% | Programar mantenimiento a corto plazo |
| 🔴 Crítico | 80% – 100% | Detener motor para inspección inmediata |

---

## 🏗️ Arquitectura

El sistema opera en tres capas comunicadas por **red WiFi local, sin
dependencia de internet**:

```
Hardware (ESP32 + sensores)
        ↓ HTTP POST
Servidor Flask (modelos XGBoost + seguridad)
        ↓ JSON
Panel de control web (diagnóstico en tiempo real)
```

![Arquitectura del sistema](priora_arquitectura.png)

### Capa física — Hardware
- **ESP32 DevKit** — microcontrolador y comunicación WiFi
- **INA219** — corriente y voltaje DC vía I2C
- **DHT22** — temperatura y humedad
- **Módulo relé** — control seguro de encendido/apagado
- **Motor DC (5-12V)** — objeto de monitoreo
- **Potenciómetro 10kΩ** — simulación controlada de fallas

![Diagrama de conexiones](priora_conexiones.png)
![Prototipo ensamblado](priora_prototipo_mockup.png)

### Capa de procesamiento — Servidor Flask
Recibe las lecturas, valida la petición, ejecuta el modelo correspondiente
y devuelve el diagnóstico. Ver [`priora_server/README.md`](priora_server/README.md)
para instalación y endpoints.

**Ciberseguridad implementada:**
- Autenticación obligatoria por API Key
- Rate limiting — 30 peticiones / 60s por IP
- Validación de esquema y rango físico plausible
- CORS restringido al origen autorizado
- Registro de auditoría con IP, endpoint y resultado

### Capa de presentación — Panel de control
![Panel en uso](priora_dashboard_uso.png)

---

## 📊 Metodología y rigor científico

### Datasets — 100% reales, sin datos sintéticos
- **AI4I 2020** (UCI Machine Learning Repository)
- **NASA IMS Bearing Dataset** (NASA Ames Research Center)
- **Open DC Motor Dataset** (Zenodo, DOI 10.5281/zenodo.4314249)

### Criterio de etiquetado — NASA IMS
El dataset NASA documenta un experimento de degradación progresiva sin
etiqueta binaria explícita. En lugar de asumir arbitrariamente un punto de
falla, se aplicó el criterio estadístico validado por el paper original
(Qiu, Lee y Lin, 2006): el inicio de falla se definió como el primer punto
donde el RMS de vibración supera la media + 3 desviaciones estándar de la
línea base, de forma sostenida. Resultado: registro 532 de 984 (54.1% del
tiempo del experimento).

### Auditoría de sesgo
Una primera validación cruzada simple mostró una caída inesperada en el
desempeño (de 72.48% a 42.00% F1 en el modelo de proceso). La investigación
reveló que la validación no replicaba el pipeline completo (SMOTE +
optimización de umbral) dentro de cada fold. Al corregirlo, los resultados
fueron estables y consistentes:

| Modelo | F1 sin pipeline por fold | F1 con pipeline completo por fold |
|---|---|---|
| AC Trifásico — Proceso | 42.00% (±14.80%) | 70.12% (±0.83%) |
| AC Trifásico — Vibración | 84.94% (±29.02%) | 99.45% (±0.00%) |
| DC Brushed | 94.04% (±9.07%) | 98.86% (±1.05%) |

---

## 📁 Estructura del repositorio

```
├── priora_v2.ipynb                          # Notebook de entrenamiento (Colab)
├── priora_server/
│   ├── app.py                               # Servidor Flask con seguridad
│   ├── static/dashboard.html                # Panel de control web
│   ├── requirements.txt
│   ├── models/                              # Modelos .pkl entrenados
│   └── README.md                            # Instrucciones del servidor
├── priora_app.html                          # App standalone (offline, sin backend)
├── priora_arquitectura.png / .svg
├── priora_conexiones.png / .svg
├── priora_prototipo_mockup.png / .svg
├── priora_dashboard_uso.png / .svg
├── Priora_Documentacion_Tesis_ES.docx        # Documentación completa (español)
├── Priora_Documentation_Thesis_EN.docx       # Documentación completa (inglés)
├── Priora_Hardware_Etapa1.docx               # Guía de ensamblaje de hardware
├── Calculos_Metodos_Priora_v2.docx           # Cálculos y métodos paso a paso
├── Priora_Presentacion_ES.pptx               # Presentación (español)
└── README.md
```

---

## 🛠️ Tecnologías

- **Lenguaje:** Python 3.10
- **ML:** scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Hardware:** ESP32, INA219, DHT22, Arduino IDE
- **Backend:** Flask, Flask-CORS
- **Frontend:** HTML / CSS / JavaScript (vanilla, sin frameworks)
- **Plataforma de entrenamiento:** Google Colab

---

## 🚀 Cómo ejecutar el sistema completo

1. **Entrenar / obtener los modelos** — ejecutar `priora_v2.ipynb` en Colab
   o usar los `.pkl` ya entrenados.
2. **Levantar el servidor** — seguir `priora_server/README.md`.
3. **Conectar el hardware** — seguir `Priora_Hardware_Etapa1.docx` para el
   ensamblaje y cargar el firmware del ESP32 apuntando a la IP local del
   servidor.
4. **Abrir el panel** — `http://<ip-del-servidor>:5000` desde cualquier
   dispositivo en la misma red local.

---

## 🔬 Limitaciones y trabajo futuro

- El modelo AC Trifásico — Proceso tiene F1 moderado (70.12%) por el fuerte
  desbalance del dataset AI4I 2020 (3.39% de fallas reales).
- El sistema clasifica el estado actual; no estima vida útil restante (RUL).
- Próximos pasos: dataset Figshare de corriente trifásica, modelos LSTM
  para RUL, validación en motores industriales de mayor potencia,
  aprendizaje federado entre plantas.

---

## 👤 Autor

**Yohaly** — Ingeniería Electrónica en Formación
[github.com/yoh6ly/priora-maintenance-system](https://github.com/yoh6ly/priora-maintenance-system)

---

*Proyecto que integra electrónica, ciencia de datos e ingeniería de sistemas
para mantenimiento predictivo en el contexto de la Industria 4.0.*
