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
    """Crear el JSON unificado con una entrada por persona"""
    print("Creando JSON unificado...")
    
    unified_data = {}
    
    # Procesar cada persona
    for _, person_row in df_demographics.iterrows():
        person_id = person_row['person_unique_id']
        
        # Información demográfica base
        person_data = {
            'edad': person_row.get('age_last_birthday', None),
            'sexo': person_row.get('sex', None),
            'raza_etnicidad': person_row.get('race_ethnicity', None),
            'estado_civil': person_row.get('marital_status_2022', None),
            'region': person_row.get('region_2022', None),
            'gastos_medicos_totales': person_row.get('total_healthcare_exp_2022', None),
            'gastos_out_of_pocket': person_row.get('total_out_of_pocket_exp_2022', None),
            'categoria_pobreza': person_row.get('poverty_category_2022', None),
            'cobertura_seguro': person_row.get('insurance_coverage_2022', None),
            'estado_salud_percibido': person_row.get('perceived_health_status', None),
            'peso_persona': person_row.get('person_weight_2022', None),
            'condiciones_medicas': [],
            'historial_empleo': [],
            'historial_seguros': [],
            'max_round_seguros': 0  # Para tracking del round máximo
        }
        
        unified_data[person_id] = person_data
    
    # PASO 1: Procesar historial de seguros privados primero para determinar round máximo
    print("Procesando historial de seguros privados...")
    insurance_counts = {'total_processed': 0, 'with_valid_premium': 0, 'inapplicable_skipped': 0}
    
    for _, insurance_row in df_insurance.iterrows():
        person_id = insurance_row['person_unique_id']
        if person_id in unified_data:
            prima_value = insurance_row.get('out_of_pocket_premium', None)
            
            # Solo agregar registros con prima válida (no "Inapplicable", no NaN, no vacío)
            if (prima_value is not None and 
                prima_value != 'Inapplicable' and 
                prima_value != '' and 
                pd.notna(prima_value)):
                
                round_num = insurance_row.get('round_number', None)
                
                insurance_data = {
                    'cobertura_seguro': insurance_row.get('insurance_coverage', None),
                    'prima_out_of_pocket': prima_value,
                    'prima_out_of_pocket_editada': insurance_row.get('out_of_pocket_premium_edited', None),
                    'round_reportado': round_num
                }
                
                unified_data[person_id]['historial_seguros'].append(insurance_data)
                insurance_counts['with_valid_premium'] += 1
                
                # Actualizar el round máximo para esta persona
                if round_num is not None and pd.notna(round_num):
                    try:
                        round_num_int = int(round_num)
                        if round_num_int > unified_data[person_id]['max_round_seguros']:
                            unified_data[person_id]['max_round_seguros'] = round_num_int
                    except (ValueError, TypeError):
                        pass
            else:
                insurance_counts['inapplicable_skipped'] += 1
            
            insurance_counts['total_processed'] += 1
    
    print(f"✓ Seguros procesados: {insurance_counts['total_processed']}")
    print(f"✓ Registros con prima válida agregados: {insurance_counts['with_valid_premium']}")
    print(f"✓ Registros con prima 'Inapplicable' omitidos: {insurance_counts['inapplicable_skipped']}")
    print(f"✓ Solo se agregaron registros de seguros con prima válida (sin 'Inapplicable')")
    
    # PASO 2: Procesar condiciones médicas filtrando por round máximo de seguros
    print("Procesando condiciones médicas...")
    condition_counts = {'total_processed': 0, 'filtered_by_round': 0}
    
    for _, condition_row in df_conditions.iterrows():
        person_id = condition_row['person_unique_id']
        if person_id in unified_data:
            condition_round = condition_row.get('condition_round', None)
            max_round_seguros = unified_data[person_id]['max_round_seguros']
            
            # Solo agregar si el round de la condición <= round máximo de seguros
            should_include = True
            if condition_round is not None and pd.notna(condition_round) and max_round_seguros > 0:
                try:
                    condition_round_int = int(condition_round)
                    if condition_round_int > max_round_seguros:
                        should_include = False
                        condition_counts['filtered_by_round'] += 1
                except (ValueError, TypeError):
                    pass
            
            if should_include:
                condition_data = {
                    'descripcion_ccsr': condition_row.get('ccsr_description', None),
                    'edad_diagnostico': condition_row.get('age_at_diagnosis', None),
                    'es_lesion': condition_row.get('injury_flag', None),
                    'round_reportado': condition_row.get('condition_round', None)
                }
                unified_data[person_id]['condiciones_medicas'].append(condition_data)
            
            condition_counts['total_processed'] += 1
    
    print(f"✓ Condiciones procesadas: {condition_counts['total_processed']}")
    print(f"✓ Condiciones filtradas por round máximo: {condition_counts['filtered_by_round']}")
    print(f"✓ Condiciones incluidas: {condition_counts['total_processed'] - condition_counts['filtered_by_round']}")
    
    # PASO 3: Procesar historial de empleos filtrando por round máximo de seguros
    print("Procesando historial de empleos...")
    job_counts = {'total_processed': 0, 'filtered_by_round': 0}
    
    for _, job_row in df_jobs.iterrows():
        person_id = job_row['person_unique_id']
        if person_id in unified_data:
            job_round = job_row.get('round_number', None)
            max_round_seguros = unified_data[person_id]['max_round_seguros']
            
            # Solo agregar si el round del empleo <= round máximo de seguros
            should_include = True
            if job_round is not None and pd.notna(job_round) and max_round_seguros > 0:
                try:
                    job_round_int = int(job_round)
                    if job_round_int > max_round_seguros:
                        should_include = False
                        job_counts['filtered_by_round'] += 1
                except (ValueError, TypeError):
                    pass
            
            if should_include:
                job_data = {
                    'seguro_ofrecido': job_row.get('insurance_offered', None),
                    'trabajo_temporal': job_row.get('temporary_job', None),
                    'empleado_asalariado': job_row.get('salaried_employee', None),
                    'salario_por_hora': job_row.get('hourly_wage', None),
                    'horas_por_semana': job_row.get('hours_per_week', None),
                    'round_reportado': job_row.get('round_number', None)
                }
                
                # Añadir directamente
                unified_data[person_id]['historial_empleo'].append(job_data)
            
            job_counts['total_processed'] += 1
    
    print(f"✓ Empleos procesados: {job_counts['total_processed']}")
    print(f"✓ Empleos filtrados por round máximo: {job_counts['filtered_by_round']}")
    print(f"✓ Empleos incluidos: {job_counts['total_processed'] - job_counts['filtered_by_round']}")
    
    # Estadísticas finales del filtrado por round máximo
    print("\n--- Resumen del filtrado por round máximo ---")
    personas_con_max_round = sum(1 for person in unified_data.values() if person['max_round_seguros'] > 0)
    print(f"✓ Personas con round máximo de seguros determinado: {personas_con_max_round}")
    print(f"✓ Total de registros filtrados por round máximo:")
    print(f"   - Condiciones médicas: {condition_counts['filtered_by_round']}")
    print(f"   - Empleos: {job_counts['filtered_by_round']}")
    print(f"   - Total filtrado: {condition_counts['filtered_by_round'] + job_counts['filtered_by_round']}")
    print("✓ Solo se incluyen condiciones médicas y empleos con round <= round máximo de seguros")
    
    # Verificación final: asegurar que todas las personas tienen al menos un registro de seguro válido
    print("\nVerificación final de datos de seguros...")
    personas_sin_prima_valida = []
    for person_id, person_data in unified_data.items():
        if not person_data['historial_seguros']:
            personas_sin_prima_valida.append(person_id)
            continue
        
        # Verificar que al menos un registro tenga prima válida
        tiene_prima_valida = False
        for seguro in person_data['historial_seguros']:
            prima = seguro.get('prima_out_of_pocket')
            if prima is not None and prima != 'Inapplicable' and prima != '':
                tiene_prima_valida = True
                break
        
        if not tiene_prima_valida:
            personas_sin_prima_valida.append(person_id)
    
    if personas_sin_prima_valida:
        print(f"⚠️  Encontradas {len(personas_sin_prima_valida)} personas sin prima válida")
        # Eliminar personas sin prima válida del dataset final
        for person_id in personas_sin_prima_valida:
            del unified_data[person_id]
        print(f"✓ Eliminadas {len(personas_sin_prima_valida)} personas sin prima válida")
        print(f"✓ Total final de personas: {len(unified_data)}")
    else:
        print("✓ Todas las personas tienen al menos un registro de prima válida")
    
    return unified_data

def generar_estadisticas(unified_data):
    """Generar estadísticas del dataset"""
    print("Generando estadísticas...")
    
    total_personas = len(unified_data)
    personas_con_condiciones = sum(1 for person in unified_data.values() if person['condiciones_medicas'])
    personas_con_empleos = sum(1 for person in unified_data.values() if person['historial_empleo'])
    personas_con_seguros = sum(1 for person in unified_data.values() if person['historial_seguros'])
    
    total_condiciones = sum(len(person['condiciones_medicas']) for person in unified_data.values())
    promedio_condiciones = total_condiciones / total_personas if total_personas > 0 else 0
    
    stats = {
        'total_personas': total_personas,
        'personas_con_condiciones': personas_con_condiciones,
        'personas_con_empleos': personas_con_empleos,
        'personas_con_seguros': personas_con_seguros,
        'total_condiciones': total_condiciones,
        'promedio_condiciones': promedio_condiciones,
        'porcentaje_con_condiciones': (personas_con_condiciones / total_personas) * 100,
        'porcentaje_con_empleos': (personas_con_empleos / total_personas) * 100,
        'porcentaje_con_seguros': (personas_con_seguros / total_personas) * 100
    }
    
    return stats

def guardar_json(unified_data, filename='meps_2022_unified_reduced.json'):
    """Guardar el JSON unificado"""
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
    
    # Aplicar conversión recursivamente
    def clean_data(data):
        if isinstance(data, dict):
            # Eliminar el campo 'max_round_seguros' si existe
            return {k: clean_data(v) for k, v in data.items() if k != 'max_round_seguros'}
        elif isinstance(data, list):
            return [clean_data(item) for item in data]
        else:
            return convert_numpy_types(data)

    clean_unified_data = clean_data(unified_data)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(clean_unified_data, f, ensure_ascii=False, indent=2)
    
    # Obtener tamaño del archivo
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
            print(f"Condiciones médicas: {len(sample_person['condiciones_medicas'])}")
            print(f"Historial de empleo: {len(sample_person['historial_empleo'])}")
            print(f"Historial de seguro: {len(sample_person['historial_seguros'])}")
            if sample_person['condiciones_medicas']:
                print(f"Primera condición (descripcion_ccsr): {sample_person['condiciones_medicas'][0]['descripcion_ccsr']}")
        
    except Exception as e:
        print(f"❌ Error durante el proceso: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
