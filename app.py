import streamlit as st
import pandas as pd
import joblib

# Cargar modelo y scaler
model  = joblib.load('modelo_motor.pkl')
scaler = joblib.load('scaler_motor.pkl')

# Configuración de la app
st.set_page_config(page_title='Detección de Fallas en Motores', page_icon='🔧')
st.title('🔧 Detección de Fallas en Motores Eléctricos')
st.markdown('Ingresa los valores de los sensores para predecir el estado del motor.')

# Panel de entrada
st.sidebar.header('⚙️ Parámetros del Motor')

tipo = st.sidebar.selectbox('Tipo de motor', options=[0, 1, 2],
                             format_func=lambda x: {0: 'H - Alta calidad',
                                                     1: 'L - Baja calidad',
                                                     2: 'M - Calidad media'}[x])
temp_aire     = st.sidebar.slider('Temperatura del aire [K]',     295.0, 305.0, 298.5, 0.1)
temp_proceso  = st.sidebar.slider('Temperatura del proceso [K]',  305.0, 315.0, 309.0, 0.1)
velocidad     = st.sidebar.slider('Velocidad rotacional [rpm]',   1168,  2886,  1500)
torque        = st.sidebar.slider('Torque [Nm]',                  3.8,   76.6,  42.0, 0.1)
desgaste      = st.sidebar.slider('Desgaste de herramienta [min]',0,     253,   180)

# Predicción
datos = pd.DataFrame([{
    'Type': tipo,
    'Air temperature [K]': temp_aire,
    'Process temperature [K]': temp_proceso,
    'Rotational speed [rpm]': velocidad,
    'Torque [Nm]': torque,
    'Tool wear [min]': desgaste
}])

datos_escalados = scaler.transform(datos)
prediccion      = model.predict(datos_escalados)[0]
probabilidad    = model.predict_proba(datos_escalados)[0]

# Resultado
st.markdown('---')
st.subheader('📊 Resultado del Diagnóstico')

if prediccion == 1:
    st.error(f'🔴 FALLA DETECTADA — Probabilidad: {probabilidad[1]*100:.1f}%')
    st.warning('⚠️ Se recomienda revisión inmediata del motor.')
else:
    st.success(f'🟢 MOTOR NORMAL — Probabilidad de falla: {probabilidad[1]*100:.1f}%')
    st.info('✅ El motor opera dentro de los parámetros normales.')

# Tabla de valores ingresados
st.markdown('---')
st.subheader('📋 Valores Ingresados')
st.dataframe(datos)

# Métricas del modelo
st.markdown('---')
st.subheader('🏆 Rendimiento del Modelo')
col1, col2, col3 = st.columns(3)
col1.metric('Accuracy',  '97.90%')
col2.metric('F1-Score',  '72.00%')
col3.metric('Recall',    '79.41%')


