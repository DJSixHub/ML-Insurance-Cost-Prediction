import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Cargar modelo RandomForest (debe estar guardado en la carpeta interface)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'random_forest_model.pkl')
if os.path.exists(MODEL_PATH):
    model_bundle = joblib.load(MODEL_PATH)
    if isinstance(model_bundle, dict) and 'model' in model_bundle:
        model = model_bundle['model']
        feature_cols = model_bundle.get('features', None)
    else:
        st.error('El archivo random_forest_model.pkl no tiene el formato esperado.')
        st.stop()
else:
    st.error('No se encontró el modelo RandomForest. Por favor, guárdalo como random_forest_model.pkl en la carpeta interface.')
    st.stop()

# Cargar las descripciones de enfermedades CCSR
ccsr_path = os.path.join(os.path.dirname(__file__), '../data/info/ccsr_reference_2025.csv')
ccsr_df = pd.read_csv(ccsr_path)
ccsr_options = sorted(ccsr_df['CCSR Category Description'].dropna().unique())

# Diccionarios para opciones legibles
salud_dict = {'Excelente': 4, 'Buena': 3, 'Regular': 2, 'Mala': 1}
pobreza_dict = {'Por encima del 400% FPL': 4, '200-399% FPL': 3, '125-199% FPL': 2, 'Pobreza (<125% FPL)': 1}
sexo_dict = {'Masculino': 1, 'Femenino': 0}
estado_civil_dict = {
    'Casado/a': 'estado_civil_Married',
    'Nunca casado/a': 'estado_civil_Never married',
    'Separado/a': 'estado_civil_Separated',
    'Menor de 16': 'estado_civil_Under 16 - not applicable',
    'Viudo/a': 'estado_civil_Widowed'
}
region_dict = {'Midwest': 'region_Midwest', 'Noreste': 'region_Northeast', 'Sur': 'region_South', 'Oeste': 'region_West'}
seguro_dict = {'Solo público': 'seguro_Public only', 'Sin seguro': 'seguro_Uninsured', 'Privado': None}

# Título
st.title('Predicción personalizada de prima médica (RandomForest)')
st.write('Introduce tu información para obtener un rango personalizado de prima médica esperada.')


# Entradas amigables
edad = st.number_input('Edad', min_value=0, max_value=120, value=30)
estado_salud = st.selectbox('Estado de salud percibido', list(salud_dict.keys()))
categoria_pobreza = st.selectbox('Categoría de pobreza', list(pobreza_dict.keys()))
tiene_historial_empleo = st.radio('¿Tienes historial de empleo?', ['Sí', 'No'])
horas_por_semana = st.number_input('Horas trabajadas por semana', min_value=0, max_value=100, value=40)
sexo = st.selectbox('Sexo', list(sexo_dict.keys()))
raza = st.selectbox('Raza/etnicidad', [
    'Asiático no hispano', 'Negro no hispano', 'Otra raza o multirracial no hispano', 'Blanco no hispano'
])
estado_civil = st.selectbox('Estado civil', list(estado_civil_dict.keys()))
region = st.selectbox('Región', list(region_dict.keys()))
seguro = st.selectbox('Tipo de seguro', list(seguro_dict.keys()))

# Enfermedades (multiselect)
enfermedades = st.multiselect('Selecciona tus condiciones médicas principales (CCSR)', ccsr_options)

# Mapeo de enfermedades a las 10 columnas dummy del dataset
ccsr_10 = [
    'Essential hypertension',
    'Disorders of lipid metabolism',
    'Diabetes mellitus without complication',
    'Bacterial infections',
    'Osteoarthritis',
    'Cataract and other lens disorders',
    'Esophageal disorders',
    'Retinal and vitreous conditions',
    'Other general signs and symptoms',
    'Abnormal findings without diagnosis'
]

# Calcular número total de condiciones seleccionadas
ccsr_num_total = len(enfermedades)
# Calcular cuántas de las 10 principales tiene el usuario
ccsr_10_flags = {f'ccsr_{name}': int(name in enfermedades) for name in ccsr_10}
# Calcular cuántas condiciones "otras" (no están en las 10 principales)
ccsr_otra_condicion = max(0, ccsr_num_total - sum(ccsr_10_flags.values()))

# Construir vector de features
features = {
    'edad': edad,
    'estado_salud_percibido': salud_dict[estado_salud],
    'ccsr_num_total': ccsr_num_total,
    'ccsr_otra_condicion': ccsr_otra_condicion,
    'categoria_pobreza': pobreza_dict[categoria_pobreza],
    'tiene_historial_empleo': 1 if tiene_historial_empleo == 'Sí' else 0,
    'horas_por_semana': horas_por_semana,
    'sexo_Male': sexo_dict[sexo],
    'raza_etnicidad_Non-Hispanic Asian only': 1 if raza == 'Asiático no hispano' else 0,
    'raza_etnicidad_Non-Hispanic Black only': 1 if raza == 'Negro no hispano' else 0,
    'raza_etnicidad_Non-Hispanic Other race or multi-race': 1 if raza == 'Otra raza o multirracial no hispano' else 0,
    'raza_etnicidad_Non-Hispanic White only': 1 if raza == 'Blanco no hispano' else 0,
    'estado_civil_Married': 1 if estado_civil == 'Casado/a' else 0,
    'estado_civil_Never married': 1 if estado_civil == 'Nunca casado/a' else 0,
    'estado_civil_Separated': 1 if estado_civil == 'Separado/a' else 0,
    'estado_civil_Under 16 - not applicable': 1 if estado_civil == 'Menor de 16' else 0,
    'estado_civil_Widowed': 1 if estado_civil == 'Viudo/a' else 0,
    'region_Midwest': 1 if region == 'Midwest' else 0,
    'region_Northeast': 1 if region == 'Noreste' else 0,
    'region_South': 1 if region == 'Sur' else 0,
    'region_West': 1 if region == 'Oeste' else 0,
    'seguro_Public only': 1 if seguro == 'Solo público' else 0,
    'seguro_Uninsured': 1 if seguro == 'Sin seguro' else 0
}
# Agregar las 10 enfermedades principales
for k, v in ccsr_10_flags.items():
    features[k] = v

# Botón de predicción
if st.button('Obtener predicción'):
    # Asegurar el orden y presencia de features exactamente como en el entrenamiento
    if feature_cols is not None:
        X = pd.DataFrame([[features.get(col, 0) for col in feature_cols]], columns=feature_cols)
    else:
        X = pd.DataFrame([features])
    pred = model.predict(X)
    lim_exc, lim_bueno, lim_reg = pred[0]
    st.success('Rango personalizado de prima médica:')
    st.markdown(f"""
    - **Excelente:** [\$0, \${lim_exc:.2f}]
    - **Bueno:** [\${lim_exc:.2f}, \${lim_bueno:.2f}]
    - **Regular:** [\${lim_bueno:.2f}, \${lim_reg:.2f}]
    - **Malo:** [> \${lim_reg:.2f}]
    """)
    st.write('Estos rangos se calculan de forma personalizada según personas similares a ti.')
