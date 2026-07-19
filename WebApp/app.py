import streamlit as st
import numpy as np
import joblib
import os

# Configuración básica de la página
@st.cache_resource
def cargar_modelos():
    try:
        # Obtiene la ruta absoluta de la carpeta donde está este archivo (app.py)
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Construye las rutas exactas uniendo la base con la carpeta 'models'
        ruta_modelo = os.path.join(base_dir, 'models', 'rf_triaje_model.pkl')
        ruta_scaler = os.path.join(base_dir, 'models', 'scaler.pkl')
        ruta_mapa = os.path.join(base_dir, 'models', 'mapa_prioridad.pkl')
        ruta_metadata = os.path.join(base_dir, 'models', 'model_metadata.pkl')

        # Carga los archivos usando las rutas absolutas
        modelo = joblib.load(ruta_modelo)
        scaler = joblib.load(ruta_scaler)
        mapa_prioridad = joblib.load(ruta_mapa)
        metadata = joblib.load(ruta_metadata)

        return modelo, scaler, mapa_prioridad, metadata['classes']

    except FileNotFoundError as e:
        st.error(f"⚠️ Error al cargar los archivos. Verifica que la carpeta 'models' exista junto a app.py. Detalle técnico: {e}")
        st.stop()

        # Retornamos tuplas vacías por seguridad en caso de que el entorno no detenga la ejecución
        return None, None, None, None

modelo, scaler, mapa_prioridad, clases = cargar_modelos()

mapa_inverso = {v: k for k, v in mapa_prioridad.items()}

# Crear dos columnas para organizar el formulario
col1, col2 = st.columns(2)

with col1:
    st.subheader("Datos Generales")
    edad = st.number_input("Edad", min_value=0, max_value=120, value=30, step=1)
    dolor = st.slider("Nivel de dolor (0-10)", min_value=0, max_value=10, value=5)

    # Menú desplegable para tiempo de evolución que coincide con tus datos
    tiempo_opcion = st.selectbox(
        "Tiempo de evolución",
        ['< 1 hora', '1–6 horas', '6–24 horas', '> 24 horas']
    )

    # Mapeo del tiempo a los valores numéricos de tu modelo
    mapa_tiempo = {'< 1 hora': 0.5, '1–6 horas': 3.5, '6–24 horas': 15, '> 24 horas': 30}
    tiempo_evolucion_horas = mapa_tiempo[tiempo_opcion]

with col2:
    st.subheader("Síntomas y Condición")
    dificultad_respirar = st.checkbox("Dificultad para respirar")
    dolor_pecho = st.checkbox("Dolor en el pecho")
    sangrado_activo = st.checkbox("Sangrado activo")
    desmayo = st.checkbox("Desmayo")
    convulsiones = st.checkbox("Convulsiones")
    embarazo = st.checkbox("Embarazo")
    enfermedades_cronicas = st.checkbox("Enfermedades crónicas")

st.markdown("---")

# Botón para ejecutar la predicción
if st.button("🚨 Generar Predicción de Triaje", use_container_width=True):

    # 1. Preparar el array de entrada en el orden exacto del entrenamiento
    entrada = np.array([[
        edad,
        dolor,
        int(dificultad_respirar),
        int(dolor_pecho),
        int(sangrado_activo),
        int(desmayo),
        int(convulsiones),
        int(embarazo),
        int(enfermedades_cronicas),
        tiempo_evolucion_horas
    ]])

    # 2. Escalar los datos
    entrada_escalada = scaler.transform(entrada)

    # 3. Realizar predicción
    nivel_num = modelo.predict(entrada_escalada)[0]
    probabilidades = modelo.predict_proba(entrada_escalada)[0]

    # 4. Obtener nombre y probabilidad
    nivel_nombre = mapa_inverso.get(nivel_num, f"Nivel {nivel_num}")
    idx_clase = list(clases).index(nivel_num)
    probabilidad = probabilidades[idx_clase]

    # 5. Mostrar resultados visuales
    st.success("✅ Análisis completado exitosamente")

    # Asignar un color visual dependiendo del nivel de emergencia (Nivel 1 al 5)
    colores_alerta = {
        1: "🔴", 2: "🟠", 3: "🟡", 4: "🔵", 5: "🟢"
    }
    icono = colores_alerta.get(nivel_num, "🏥")

    st.markdown(f"### {icono} Resultado: Nivel {nivel_num} - {nivel_nombre}")
    st.write(f"**Confianza de la predicción:** {probabilidad:.2%}")

    # Barra de progreso visual para la confianza
    st.progress(float(probabilidad))
