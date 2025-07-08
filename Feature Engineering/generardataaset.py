import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder

def load_json_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Convert to DataFrame, keeping person_id as a column
    records = []
    for person_id, person_data in data.items():
        row = person_data.copy()
        row['person_id'] = person_id
        records.append(row)
    return pd.DataFrame(records)

def extract_target(row):
    # Calcula la media de todas las primas_out_of_pocket_editada de historial_seguros
    seguros = row.get('historial_seguros', [])
    valores = []
    if seguros and isinstance(seguros, list):
        for s in seguros:
            val = s.get('prima_out_of_pocket_editada', None)
            if val is not None:
                try:
                    valores.append(float(val))
                except:
                    continue
    if valores:
        return np.mean(valores)
    return np.nan

def extract_ccsr_conditions(row):
    # Devuelve lista de condiciones CCSR actuales
    condiciones = row.get('condiciones_medicas_actuales', [])
    return [c.get('descripcion_ccsr', '') for c in condiciones if 'descripcion_ccsr' in c]

def main():
    # Paths
    # Ajustar rutas para ejecución desde la raíz del proyecto
    json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'meps_2022_unified_reduced.json'))
    ccsr_reference_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'info', 'ccsr_reference_2025.csv'))
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Models', 'meps_ml_dataset.csv'))

    # Load data
    df = load_json_data(json_path)

    # Extraer variable objetivo
    df['prima_out_of_pocket_editada'] = df.apply(extract_target, axis=1)


    # Extraer condiciones CCSR
    df['ccsr_conditions'] = df.apply(extract_ccsr_conditions, axis=1)

    # Cargar referencia CCSR
    ccsr_ref = pd.read_csv(ccsr_reference_path)
    known_ccsr = set(ccsr_ref['CCSR'].unique()) if 'CCSR' in ccsr_ref.columns else set(ccsr_ref.iloc[:,0].unique())

    # Obtener todas las condiciones presentes en el dataset
    all_ccsr = set()
    for conds in df['ccsr_conditions']:
        all_ccsr.update(conds)

    # Crear matriz binaria de enfermedades
    ccsr_list = sorted(list(all_ccsr))
    ccsr_matrix = pd.DataFrame(0, index=df.index, columns=ccsr_list)
    for idx, conds in enumerate(df['ccsr_conditions']):
        for c in conds:
            if c in ccsr_matrix.columns:
                ccsr_matrix.at[idx, c] = 1

    # Calcular correlación absoluta con la variable objetivo
    y_corr = df['prima_out_of_pocket_editada']
    corrs = ccsr_matrix.apply(lambda col: abs(col.corr(y_corr, method='spearman')), axis=0)
    top_ccsr = corrs.sort_values(ascending=False).head(10).index.tolist()

    # Crear columnas binarias solo para las top correlacionadas (top 10)
    for ccsr in top_ccsr:
        df[f'ccsr_{ccsr}'] = ccsr_matrix[ccsr]

    # Contar cuántas condiciones "otras" tiene cada persona (no top 10)
    df['ccsr_otra_condicion'] = df['ccsr_conditions'].apply(lambda conds: sum([(c not in top_ccsr) for c in conds]))

    # Feature: número total de condiciones
    df['ccsr_num_total'] = df['ccsr_conditions'].apply(len)

    # One-hot encoding de tipo de seguro (cobertura_seguro)
    df = pd.get_dummies(df, columns=['cobertura_seguro'], prefix='seguro', drop_first=True, dtype=int)

    # Label encoding para categoria_pobreza
    if 'categoria_pobreza' in df.columns:
        le_pobreza = LabelEncoder()
        df['categoria_pobreza'] = le_pobreza.fit_transform(df['categoria_pobreza'].astype(str))
    else:
        df['categoria_pobreza'] = 0

    # One-hot encoding de si tiene historial_empleo (1 si hay al menos una entrada, 0 si no)
    df['tiene_historial_empleo'] = df['historial_empleo'].apply(lambda x: int(isinstance(x, list) and len(x) > 0))

    # Calcular media de horas_por_semana en historial_empleo (si no hay, usar media global)
    def extraer_horas(row):
        empleos = row.get('historial_empleo', [])
        horas = []
        if empleos and isinstance(empleos, list):
            for e in empleos:
                val = e.get('horas_por_semana', None)
                try:
                    valf = float(val)
                    horas.append(valf)
                except:
                    continue
        if horas:
            return np.mean(horas)
        return np.nan
    df['horas_por_semana'] = df.apply(extraer_horas, axis=1)
    global_mean_horas = df['horas_por_semana'].mean()
    df['horas_por_semana'] = df['horas_por_semana'].fillna(global_mean_horas)

    # Variables categóricas one-hot (con 0/1 en vez de True/False)
    one_hot_features = ['sexo', 'raza_etnicidad', 'estado_civil', 'region']
    df = pd.get_dummies(df, columns=one_hot_features, drop_first=True, dtype=int)

    # Variables label encoding
    label_features = ['estado_salud_percibido']
    for col in label_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Selección de features finales
    features = [
        'edad', 'estado_salud_percibido',
        'ccsr_num_total', 'ccsr_otra_condicion',
        'categoria_pobreza', 'tiene_historial_empleo', 'horas_por_semana'
    ]
    # Agregar dummies y top CCSR
    features += [c for c in df.columns if c.startswith('sexo_') or c.startswith('raza_etnicidad_') or c.startswith('estado_civil_') or c.startswith('region_')]
    features += [f'ccsr_{c}' for c in top_ccsr]
    features += [c for c in df.columns if c.startswith('seguro_')]

    # Eliminar person_id y columnas auxiliares
    X = df[features].copy()
    y = df['prima_out_of_pocket_editada']

    # Guardar el dataset completo (sin eliminar outliers)
    dataset = X.copy()
    dataset['prima_out_of_pocket_editada'] = y
    dataset.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path} (con outliers de prima_out_of_pocket_editada)")

if __name__ == "__main__":
    main()
