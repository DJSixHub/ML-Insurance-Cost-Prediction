#!/usr/bin/env python3
"""
Generador de JSON Reducido MEPS 2022
Crea un dataset unificado con personas seleccionadas al azar
para análisis más rápido y prototipado.
"""

import json
import pandas as pd
import numpy as np
import random
import os

def cargar_datos():
    """Cargar todos los datasets procesados"""
    files = {
        'demographics': 'data/meps_fyc_2022_processed.csv',
        'conditions': 'data/meps_cond_2022_processed.csv',
        'jobs': 'data/meps_jobs_2022_processed.csv',
        'insurance': 'data/meps_prpl_2022_processed.csv'
    }
    
    print("Cargando datos MEPS procesados...")
    
    # Cargar datos demográficos (una fila por persona)
    print("Cargando datos demográficos...")
    df_demographics = pd.read_csv(files['demographics'], low_memory=False)
    
    # Cargar datos de condiciones médicas (múltiples filas por persona)
    print("Cargando datos de condiciones médicas...")
    df_conditions = pd.read_csv(files['conditions'], low_memory=False)
    
    # Cargar datos de trabajos (múltiples filas por persona)
    print("Cargando datos de trabajos...")
    df_jobs = pd.read_csv(files['jobs'], low_memory=False)
    
    # Cargar datos de seguros privados (múltiples filas por persona)
    print("Cargando datos de seguros privados...")
    df_insurance = pd.read_csv(files['insurance'], low_memory=False)
    
    # Cargar archivo de referencia CCSR para mapeo de descripciones
    print("Cargando archivo de referencia CCSR...")
    try:
        df_ccsr_ref = pd.read_csv(os.path.join('data', 'info', 'ccsr_reference_2025.csv'))
        print(f"✓ Archivo CCSR cargado con {len(df_ccsr_ref)} registros")
    except FileNotFoundError:
        print("⚠️  No se encontró el archivo CCSR de referencia")
        df_ccsr_ref = None
    
    return df_demographics, df_conditions, df_jobs, df_insurance, df_ccsr_ref

def seleccionar_muestra_aleatoria(df_demographics, df_insurance, sample_size=10000):
    """Seleccionar una muestra aleatoria de personas con historial de seguros válido"""
    print(f"Seleccionando muestra aleatoria de {sample_size} personas con historial de seguros válido...")
    
    # Establecer semilla para reproducibilidad
    random.seed(42)
    np.random.seed(42)
    
    # Filtrar personas que tengan historial de seguros
    personas_con_seguros = set(df_insurance['person_unique_id'].unique())
    print(f"Personas con historial de seguros: {len(personas_con_seguros)}")
    
    # Filtrar personas que NO tengan "Inapplicable" en prima_out_of_pocket
    # También verificar que no sea None o NaN
    df_insurance_valido = df_insurance[
        (df_insurance['out_of_pocket_premium'] != 'Inapplicable') &
        (df_insurance['out_of_pocket_premium'].notna()) &
        (df_insurance['out_of_pocket_premium'] != '')
    ]
    personas_con_prima_valida = set(df_insurance_valido['person_unique_id'].unique())
    print(f"Personas con prima out-of-pocket válida (no 'Inapplicable', no NaN, no vacío): {len(personas_con_prima_valida)}")
    
    # Intersección: personas que tengan seguros Y prima válida
    personas_validas = personas_con_seguros.intersection(personas_con_prima_valida)
    print(f"Personas con seguros y prima válida: {len(personas_validas)}")
    
    # Filtrar el DataFrame demográfico
    df_demo_valido = df_demographics[df_demographics['person_unique_id'].isin(personas_validas)]
    
    # Seleccionar muestra aleatoria
    if len(df_demo_valido) < sample_size:
        print(f"⚠️  Solo hay {len(df_demo_valido)} personas válidas. Usando todas.")
        selected_ids = df_demo_valido['person_unique_id'].tolist()
    else:
        selected_ids = df_demo_valido['person_unique_id'].sample(n=sample_size, random_state=42).tolist()
    
    print(f"✓ Seleccionadas {len(selected_ids)} personas válidas")
    return selected_ids

def filtrar_datos_por_muestra(df_demographics, df_conditions, df_jobs, df_insurance, df_ccsr_ref, selected_ids):
    """Filtrar todos los datasets para incluir solo las personas seleccionadas"""
    print("Filtrando datos para la muestra seleccionada...")
    
    # Filtrar demographics
    df_demo_filtered = df_demographics[df_demographics['person_unique_id'].isin(selected_ids)]
    
    # Filtrar conditions
    df_cond_filtered = df_conditions[df_conditions['person_unique_id'].isin(selected_ids)]
    
    # Aplicar mapeo CCSR si está disponible
    if df_ccsr_ref is not None:
        print("Aplicando mapeo de descripciones ICD10 y CCSR...")
        
        # Crear mapeo de ICD10 a descripción
        icd10_map = {}
        
        for _, row in df_ccsr_ref.iterrows():
            icd10_code = row['ICD-10-CM Code']
            icd10_desc = row['ICD-10-CM Code Description']
            if pd.notna(icd10_code) and pd.notna(icd10_desc):
                if icd10_code not in icd10_map:
                    icd10_map[icd10_code] = icd10_desc
        
        # Crear mapeo de CCSR a descripción (manejar duplicados tomando el primero)
        ccsr_map = {}
        for _, row in df_ccsr_ref.iterrows():
            ccsr_code = row['CCSR Category']
            ccsr_desc = row['CCSR Category Description']
            if pd.notna(ccsr_code) and pd.notna(ccsr_desc) and ccsr_code not in ccsr_map:
                ccsr_map[ccsr_code] = ccsr_desc
        
        print(f"✓ Mapeos creados: {len(icd10_map)} códigos ICD10, {len(ccsr_map)} códigos CCSR")
        
        # Aplicar mapeos a las condiciones filtradas
        df_cond_filtered = df_cond_filtered.copy()
        
        # Función para mapear ICD10 con fallback a CCSR
        def mapear_icd10_con_fallback(row):
            icd10_code = row['icd10_code']
            ccsr_category = row['ccsr_category_1']
            
            # Primero intentar con ICD10
            if pd.notna(icd10_code) and icd10_code in icd10_map:
                return icd10_map[icd10_code]
            
            # Si ICD10 no se encuentra, usar CCSR como fallback
            if pd.notna(ccsr_category) and ccsr_category in ccsr_map:
                return ccsr_map[ccsr_category]
            
            # Si ninguno funciona, usar descripción genérica
            return "Condición médica no especificada"
        
        # Aplicar mapeo ICD10 con fallback a CCSR
        df_cond_filtered['icd10_description'] = df_cond_filtered.apply(mapear_icd10_con_fallback, axis=1)
        
        # Mapear CCSR - usar la descripción real del archivo CCSR reference
        df_cond_filtered['ccsr_description'] = df_cond_filtered['ccsr_category_1'].apply(
            lambda x: ccsr_map.get(x, "Categoría médica no especificada") if pd.notna(x) else "No especificado"
        )
        
        print(f"✓ Mapeo aplicado a {len(df_cond_filtered)} registros de condiciones")
        
        # Estadísticas del mapeo
        icd10_mapeados = sum(1 for x in df_cond_filtered['icd10_code'] if pd.notna(x) and x in icd10_map)
        ccsr_mapeados = sum(1 for x in df_cond_filtered['ccsr_category_1'] if pd.notna(x) and x in ccsr_map)
        
        print(f"✓ ICD10 con descripción encontrada: {icd10_mapeados}")
        print(f"✓ CCSR con descripción encontrada: {ccsr_mapeados}")
        
    else:
        # Si no hay archivo de referencia, usar descripciones genéricas
        print("⚠️  Sin archivo de referencia CCSR - usando descripciones genéricas")
        df_cond_filtered = df_cond_filtered.copy()
        
        # Función para mapear cuando no hay archivo de referencia
        def mapear_sin_referencia(row):
            icd10_code = row['icd10_code']
            ccsr_category = row['ccsr_category_1']
            
            if pd.notna(icd10_code):
                return "Condición médica no especificada"
            
            if pd.notna(ccsr_category):
                return "Categoría médica no especificada"
            
            return "No especificado"
        
        df_cond_filtered['icd10_description'] = df_cond_filtered.apply(mapear_sin_referencia, axis=1)
        df_cond_filtered['ccsr_description'] = df_cond_filtered['ccsr_category_1'].apply(
            lambda x: "Categoría médica no especificada" if pd.notna(x) else "No especificado"
        )
    
    # Filtrar jobs
    df_jobs_filtered = df_jobs[df_jobs['person_unique_id'].isin(selected_ids)]
    
    # Filtrar insurance
    df_insurance_filtered = df_insurance[df_insurance['person_unique_id'].isin(selected_ids)]
    
    print(f"✓ Datos filtrados:")
    print(f"   - Demografía: {len(df_demo_filtered)} personas")
    print(f"   - Condiciones: {len(df_cond_filtered)} registros")
    print(f"   - Empleos: {len(df_jobs_filtered)} registros")
    print(f"   - Seguros: {len(df_insurance_filtered)} registros")
    
    return df_demo_filtered, df_cond_filtered, df_jobs_filtered, df_insurance_filtered

def crear_json_unificado(df_demographics, df_conditions, df_jobs, df_insurance, df_ccsr_ref):
    #
    """Crear el JSON unificado con una entrada por persona, dividiendo condiciones médicas en actuales/pasadas y calculando cantidad_lesiones"""
    print("Creando JSON unificado...")

   
    def filtrar_historial_seguros(unified_data):
        # Filtro final: eliminar registros de historial_seguros con prima_out_of_pocket_editada igual a 0.0
        for person_id, person_data in unified_data.items():
            seguros_filtrados = []
            for seguro in person_data['historial_seguros']:
                try:
                    prima_val = float(seguro.get('prima_out_of_pocket_editada', 0))
                except (TypeError, ValueError):
                    prima_val = 0
                if prima_val != 0.0:
                    seguros_filtrados.append(seguro)
            person_data['historial_seguros'] = seguros_filtrados

        # Eliminar personas que se hayan quedado sin historial_seguros tras el filtro
        personas_sin_seguros = [pid for pid, pdata in unified_data.items() if not pdata['historial_seguros']]
        for pid in personas_sin_seguros:
            del unified_data[pid]
        return unified_data
    CCIR_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'info', 'CCIR_v2025-1.csv'))
    CCSR_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'info', 'ccsr_reference_2025.csv'))
    def cargar_mapeos_cronicos():
        try:
            df_ccir = pd.read_csv(CCIR_FILE, skiprows=2)
        except FileNotFoundError:
            alt_ccir = os.path.join('data', 'info', 'CCIR_v2025-1.csv')
            df_ccir = pd.read_csv(alt_ccir, skiprows=2)
        try:
            df_ccsr = pd.read_csv(CCSR_FILE)
        except FileNotFoundError:
            alt_ccsr = os.path.join('data', 'info', 'ccsr_reference_2025.csv')
            df_ccsr = pd.read_csv(alt_ccsr)
        ccir_chronic_map = {}
        icd_col = None
        chronic_col = None
        for col in df_ccir.columns:
            col_clean = str(col).strip("'\"")
            if 'ICD-10-CM CODE' in col_clean and 'DESCRIPTION' not in col_clean:
                icd_col = col
            elif 'CHRONIC INDICATOR' in col_clean:
                chronic_col = col
        if icd_col is not None and chronic_col is not None:
            for _, row in df_ccir.iterrows():
                icd10_code = str(row.get(icd_col, '')).strip("'\"")
                chronic_indicator = row.get(chronic_col, 0)
                if pd.notna(icd10_code) and icd10_code:
                    ccir_chronic_map[icd10_code] = int(chronic_indicator) == 1
        ccsr_to_icd10_map = {}
        for _, row in df_ccsr.iterrows():
            ccsr_desc = row.get('CCSR Category Description', '')
            icd10_code = row.get('ICD-10-CM Code', '')
            if pd.notna(ccsr_desc) and pd.notna(icd10_code) and ccsr_desc and icd10_code:
                if ccsr_desc not in ccsr_to_icd10_map:
                    ccsr_to_icd10_map[ccsr_desc] = set()
                ccsr_to_icd10_map[ccsr_desc].add(icd10_code)
        return ccir_chronic_map, ccsr_to_icd10_map
    def es_condicion_cronica(descripcion_ccsr, ccir_chronic_map, ccsr_to_icd10_map):
        if not descripcion_ccsr or descripcion_ccsr == "No especificado":
            return False
        icd10_codes = ccsr_to_icd10_map.get(descripcion_ccsr, set())
        if not icd10_codes:
            return False
        for icd10_code in icd10_codes:
            if ccir_chronic_map.get(icd10_code, False):
                return True
        return False
    ccir_chronic_map, ccsr_to_icd10_map = cargar_mapeos_cronicos()

    # Mapeo robusto de estado de salud percibido
    def map_estado_salud(val):
        mapping = {
            'excellent': 'excellent',
            'very good': 'very good',
            'good': 'good',
            'fair': 'fair',
            'poor': 'poor',
            'excelente': 'excellent',
            'muy buena': 'very good',
            'buena': 'good',
            'regular': 'fair',
            'mala': 'poor',
        }
        # Lowercase and strip
        if val is None:
            return 'fair'  # Default fallback
        v = str(val).strip().lower()
        # Map common non-informative values to a default (e.g., 'fair')
        if v in ['inapplicable', 'unknown/not reported', "don't know", 'unknown', 'not ascertained', 'refused', 'na', 'n/a', '', 'nan', 'none']:
            return 'fair'  # Or choose another default if preferred
        # Try mapping
        return mapping.get(v, 'fair')  # Default to 'fair' if not found

    unified_data = {}
    # Inicializar personas
    for _, person_row in df_demographics.iterrows():
        person_id = person_row['person_unique_id']
        person_data = {
            'edad': person_row.get('age_last_birthday', None),
            'sexo': person_row.get('sex', None),
            'raza_etnicidad': person_row.get('race_ethnicity', None),
            'estado_civil': person_row.get('marital_status_2022', None),
            'region': person_row.get('region_2022', None),
            'categoria_pobreza': person_row.get('poverty_category_2022', None),
            'cobertura_seguro': person_row.get('insurance_coverage_2022', None),
            'estado_salud_percibido': map_estado_salud(person_row.get('perceived_health_status', None)),
            'condiciones_medicas_actuales': [],
            'condiciones_medicas_pasadas': [],
            'historial_empleo': [],
            'historial_seguros': [],
            'max_round_seguros': 0
        }
        unified_data[person_id] = person_data

    # Procesar historial de seguros para determinar round máximo
    for _, insurance_row in df_insurance.iterrows():
        person_id = insurance_row['person_unique_id']
        if person_id in unified_data:
            prima_editada = insurance_row.get('out_of_pocket_premium_edited', None)
            # Filtrar registros con prima igual a 0.0
            try:
                prima_val = float(prima_editada)
            except (TypeError, ValueError):
                prima_val = None
            if (prima_editada is not None and prima_editada != '' and prima_editada != 'Inapplicable' and pd.notna(prima_editada) and prima_val != 0.0):
                round_num = insurance_row.get('round_number', None)
                insurance_data = {
                    'cobertura_seguro': insurance_row.get('insurance_coverage', None),
                    'prima_out_of_pocket_editada': prima_editada
                }
                unified_data[person_id]['historial_seguros'].append(insurance_data)
                if round_num is not None and pd.notna(round_num):
                    try:
                        round_num_int = int(round_num)
                        if round_num_int > unified_data[person_id]['max_round_seguros']:
                            unified_data[person_id]['max_round_seguros'] = round_num_int
                    except (ValueError, TypeError):
                        pass

    # Procesar condiciones médicas: dividir en actuales/pasadas y contar lesiones
    for _, condition_row in df_conditions.iterrows():
        person_id = condition_row['person_unique_id']
        if person_id in unified_data:
            max_round_seguros = unified_data[person_id]['max_round_seguros']
            condition_round = condition_row.get('condition_round', None)
            try:
                condition_round_int = int(condition_round) if pd.notna(condition_round) else None
            except (ValueError, TypeError):
                condition_round_int = None
            descripcion_ccsr = condition_row.get('ccsr_description', None)
            injury_flag = condition_row.get('injury_flag', None)
            es_cronica = es_condicion_cronica(descripcion_ccsr, ccir_chronic_map, ccsr_to_icd10_map)
            # Lesiones no se consideran ya que no hay registros en el dataset original
            cond_dict = {
                'descripcion_ccsr': descripcion_ccsr,
                'edad_diagnostico': condition_row.get('age_at_diagnosis', None),
                'icd10_code': condition_row.get('icd10_code', None),
                'ccsr_category_1': condition_row.get('ccsr_category_1', None)
            }
            if (es_cronica or (condition_round_int is not None and condition_round_int == max_round_seguros)):
                unified_data[person_id]['condiciones_medicas_actuales'].append(cond_dict)
            elif (not es_cronica and (condition_round_int is not None and condition_round_int < max_round_seguros)):
                unified_data[person_id]['condiciones_medicas_pasadas'].append(cond_dict)

    # Procesar historial de empleos (igual que antes, pero mapeando valores a 'Not Reported')
    valores_no_reportados = set([
        'Inapplicable', 'Unknown/Not reported', "Don't know", 'Unknown', 'Not ascertained', 'Refused', 'NA', 'N/A', '', None
    ])
    valores_no_reportados_lower = set(x.lower() for x in valores_no_reportados if isinstance(x, str))
    def map_no_reportado(val):
        if val is None:
            return 'Not Reported'
        if isinstance(val, str):
            v = val.strip().lower()
            if v in valores_no_reportados_lower:
                return 'Not Reported'
        if val in valores_no_reportados:
            return 'Not Reported'
        return val
    for _, job_row in df_jobs.iterrows():
        person_id = job_row['person_unique_id']
        if person_id in unified_data:
            job_round = job_row.get('round_number', None)
            max_round_seguros = unified_data[person_id]['max_round_seguros']
            should_include = True
            if job_round is not None and pd.notna(job_round) and max_round_seguros > 0:
                try:
                    job_round_int = int(job_round)
                    if job_round_int > max_round_seguros:
                        should_include = False
                except (ValueError, TypeError):
                    pass
            if should_include:
                job_data = {
                    'seguro_ofrecido': map_no_reportado(job_row.get('insurance_offered', None)),
                    'trabajo_temporal': map_no_reportado(job_row.get('temporary_job', None)),
                    'empleado_asalariado': map_no_reportado(job_row.get('salaried_employee', None)),
                    'salario_por_hora': map_no_reportado(job_row.get('hourly_wage', None)),
                    'horas_por_semana': map_no_reportado(job_row.get('hours_per_week', None))
                }
                unified_data[person_id]['historial_empleo'].append(job_data)

    # Eliminar personas que no tengan al menos un seguro con prima válida (>0, no vacío, no inaplicable, no NaN)
    personas_sin_prima_valida = []
    for person_id, person_data in list(unified_data.items()):
        tiene_prima_valida = False
        for seguro in person_data['historial_seguros']:
            prima_editada = seguro.get('prima_out_of_pocket_editada')
            try:
                prima_val = float(prima_editada)
            except (TypeError, ValueError):
                prima_val = None
            if prima_val is not None and prima_val > 0:
                tiene_prima_valida = True
                break
        if not tiene_prima_valida:
            personas_sin_prima_valida.append(person_id)
    for person_id in personas_sin_prima_valida:
        del unified_data[person_id]
    # No aplicar ningún otro filtro de exclusión
    return unified_data

def generar_estadisticas(unified_data):
    """Generar estadísticas del dataset"""
    print("Generando estadísticas...")
    
    total_personas = len(unified_data)
    personas_con_condiciones = sum(
        1 for person in unified_data.values()
        if person.get('condiciones_medicas_actuales', []) or person.get('condiciones_medicas_pasadas', [])
    )
    personas_con_empleos = sum(1 for person in unified_data.values() if person['historial_empleo'])
    personas_con_seguros = sum(1 for person in unified_data.values() if person['historial_seguros'])

    total_condiciones = sum(
        len(person.get('condiciones_medicas_actuales', [])) + len(person.get('condiciones_medicas_pasadas', []))
        for person in unified_data.values()
    )
    promedio_condiciones = total_condiciones / total_personas if total_personas > 0 else 0

    # Evitar división por cero en estadísticas
    if total_personas == 0:
        stats = {
            'total_personas': 0,
            'personas_con_condiciones': 0,
            'personas_con_empleos': 0,
            'personas_con_seguros': 0,
            'total_condiciones': 0,
            'promedio_condiciones': 0,
            'porcentaje_con_condiciones': 0,
            'porcentaje_con_empleos': 0,
            'porcentaje_con_seguros': 0
        }
        print("❌ No hay personas válidas en el dataset final. Revisa los filtros y la lógica de inclusión de primas.")
        return stats

    stats = {
        'total_personas': total_personas,
        'personas_con_condiciones': personas_con_condiciones,
        'personas_con_empleos': personas_con_empleos,
        'personas_con_seguros': personas_con_seguros,
        'total_condiciones': total_condiciones,
        'promedio_condiciones': promedio_condiciones,
        'porcentaje_con_condiciones': (personas_con_condiciones / total_personas) * 100 if total_personas > 0 else 0,
        'porcentaje_con_empleos': (personas_con_empleos / total_personas) * 100 if total_personas > 0 else 0,
        'porcentaje_con_seguros': (personas_con_seguros / total_personas) * 100 if total_personas > 0 else 0
    }
    return stats

def guardar_json(unified_data, filename='meps_2022_unified_reduced.json'):
    """Guardar el JSON unificado, eliminando campos internos innecesarios de condiciones médicas"""
    print(f"Guardando datos en {filename}...")

    # Convertir numpy types a tipos nativos de Python
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return obj

    # Eliminar campos 'icd10_code' y 'ccsr_category_1' de condiciones médicas
    def clean_condiciones(lista):
        return [
            {k: v for k, v in cond.items() if k not in ('icd10_code', 'ccsr_category_1')}
            for cond in lista
        ]

    # Aplicar conversión recursivamente y limpiar condiciones
    def clean_data(data):
        if isinstance(data, dict):
            # Eliminar el campo 'max_round_seguros' si existe
            d = {k: clean_data(v) for k, v in data.items() if k != 'max_round_seguros'}
            # Limpiar condiciones médicas si existen
            if 'condiciones_medicas_actuales' in d:
                d['condiciones_medicas_actuales'] = clean_condiciones(d['condiciones_medicas_actuales'])
            if 'condiciones_medicas_pasadas' in d:
                d['condiciones_medicas_pasadas'] = clean_condiciones(d['condiciones_medicas_pasadas'])
            return d
        elif isinstance(data, list):
            return [clean_data(item) for item in data]
        else:
            return convert_numpy_types(data)

    clean_unified_data = clean_data(unified_data)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(clean_unified_data, f, ensure_ascii=False, indent=2)
    file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
    print(f"Archivo guardado: {filename} ({file_size:.2f} MB)")
    return filename

def main():
    """Función principal"""
    print("="*60)
    print("GENERADOR DE JSON REDUCIDO MEPS 2022")
    print("="*60)
    
    try:
        # Cargar datos
        df_demographics, df_conditions, df_jobs, df_insurance, df_ccsr_ref = cargar_datos()
        
        # Seleccionar muestra aleatoria con filtro estricto
        selected_ids = seleccionar_muestra_aleatoria(df_demographics, df_insurance, sample_size=21000)
        
        # Filtrar datos
        df_demo_filtered, df_cond_filtered, df_jobs_filtered, df_insurance_filtered = filtrar_datos_por_muestra(
            df_demographics, df_conditions, df_jobs, df_insurance, df_ccsr_ref, selected_ids
        )
        
        # Crear JSON unificado
        unified_data = crear_json_unificado(
            df_demo_filtered, df_cond_filtered, df_jobs_filtered, df_insurance_filtered, df_ccsr_ref
        )
        
        # Generar estadísticas
        stats = generar_estadisticas(unified_data)
        
        print("\nEstadísticas del dataset reducido:")
        print(f"Total de personas: {stats['total_personas']}")
        print(f"Personas con condiciones médicas: {stats['personas_con_condiciones']} ({stats['porcentaje_con_condiciones']:.1f}%)")
        print(f"Personas con historial de empleo: {stats['personas_con_empleos']} ({stats['porcentaje_con_empleos']:.1f}%)")
        print(f"Personas con historial de seguro privado: {stats['personas_con_seguros']} ({stats['porcentaje_con_seguros']:.1f}%)")
        print(f"Promedio de condiciones médicas por persona: {stats['promedio_condiciones']:.2f}")
        
        # Guardar JSON
        filename = guardar_json(unified_data)
        
        print(f"\n✓ Proceso completado exitosamente!")
        print(f"✓ Archivo generado: {filename}")
        
        # Mostrar ejemplo de entrada
        print("="*60)
        print("EJEMPLO DE ENTRADA EN EL JSON:")
        print("="*60)
        if unified_data:
            sample_person_id = list(unified_data.keys())[0]
            sample_person = unified_data[sample_person_id]
            print(f"ID de persona: {sample_person_id}")
            print(f"Edad: {sample_person['edad']}")
            print(f"Sexo: {sample_person['sexo']}")
            total_condiciones = len(sample_person.get('condiciones_medicas_actuales', [])) + len(sample_person.get('condiciones_medicas_pasadas', []))
            print(f"Condiciones médicas (total): {total_condiciones}")
            print(f"Condiciones actuales: {len(sample_person.get('condiciones_medicas_actuales', []))}")
            print(f"Condiciones pasadas: {len(sample_person.get('condiciones_medicas_pasadas', []))}")
            print(f"Cantidad de lesiones: {sample_person.get('cantidad_lesiones', 0)}")
            print(f"Historial de empleo: {len(sample_person['historial_empleo'])}")
            print(f"Historial de seguro: {len(sample_person['historial_seguros'])}")
            # Mostrar primera condición si existe
            if sample_person.get('condiciones_medicas_actuales'):
                print(f"Primera condición actual (descripcion_ccsr): {sample_person['condiciones_medicas_actuales'][0]['descripcion_ccsr']}")
            elif sample_person.get('condiciones_medicas_pasadas'):
                print(f"Primera condición pasada (descripcion_ccsr): {sample_person['condiciones_medicas_pasadas'][0]['descripcion_ccsr']}")
        
    except Exception as e:
        print(f"❌ Error durante el proceso: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
