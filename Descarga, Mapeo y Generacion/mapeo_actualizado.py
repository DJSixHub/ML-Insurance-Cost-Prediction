"""
Mapeo de Datos MEPS 2022 - Versión Actualizada
Convierte códigos MEPS a valores legibles para humanos y mejora nombres de columnas
"""

import pandas as pd
import numpy as np
import os

def load_ccsr_reference():
    """Cargar el archivo de referencia CCSR para mapeo de condiciones médicas"""
    try:
        ccsr_df = pd.read_csv(os.path.join('data', 'info', 'ccsr_reference_2025.csv'))
        return ccsr_df
    except FileNotFoundError:
        print("Warning: No se encontró el archivo ccsr_reference_2025.csv")
        return None

def get_reserved_codes():
    """Códigos reservados de MEPS y sus significados"""
    return {
        -1: "Inapplicable",
        -2: "Determined in previous round",
        -7: "Refused",
        -8: "Don't know",
        -9: "Not ascertained",
        -10: "Hourly wage >= $103.36",
        -13: "Initial wage imputed",
        -15: "Cannot be computed"
    }

def get_column_mappings():
    """Mapeo de nombres de columnas técnicos a nombres legibles"""
    return {
        # Full Year Consolidated (FYC)
        'DUID': 'dwelling_unit_id',
        'PID': 'person_id',
        'DUPERSID': 'person_unique_id',
        'PANEL': 'panel_number',
        'AGELAST': 'age_last_birthday',
        'SEX': 'sex',
        'RACETHX': 'race_ethnicity',
        'MARRY22X': 'marital_status_2022',
        'REGION22': 'region_2022',
        'TOTEXP22': 'total_healthcare_exp_2022',
        'TOTSLF22': 'total_out_of_pocket_exp_2022',
        'POVCAT22': 'poverty_category_2022',
        'INSCOV22': 'insurance_coverage_2022',
        'RTHLTH53': 'perceived_health_status',
        'PERWT22F': 'person_weight_2022',
        
        # Medical Conditions (COND)
        'CONDIDX': 'condition_id',
        'CONDRN': 'condition_round',
        'AGEDIAG': 'age_at_diagnosis',
        'INJURY': 'injury_flag',
        'ICD10CDX': 'icd10_code',
        'CCSR1X': 'ccsr_category_1',
        
        # Private Insurance (PRPL)
        'RN': 'round_number',
        'INSCOV': 'insurance_coverage',
        'OOPPREM': 'out_of_pocket_premium',
        'OOPPREMX': 'out_of_pocket_premium_edited',
        
        # Jobs (JOBS)
        'JOBSIDX': 'job_id',
        'OFFRDINS': 'insurance_offered',
        'TEMPJOB': 'temporary_job',
        'SALARIED': 'salaried_employee',
        'HRLYWAGE': 'hourly_wage',
        'HRSPRWK': 'hours_per_week'
    }

def get_value_mappings():
    """Mapeo de códigos categóricos a valores legibles"""
    return {
        'sex': {
            1: 'Male',
            2: 'Female'
        },
        'race_ethnicity': {
            1: 'Hispanic',
            2: 'Non-Hispanic White only',
            3: 'Non-Hispanic Black only',
            4: 'Non-Hispanic Asian only',
            5: 'Non-Hispanic Other race or multi-race'
        },
        'marital_status_2022': {
            1: 'Married',
            2: 'Widowed',
            3: 'Divorced',
            4: 'Separated',
            5: 'Never married',
            6: 'Under 16 - not applicable'
        },
        'region_2022': {
            1: 'Northeast',
            2: 'Midwest',
            3: 'South',
            4: 'West',
            -1: 'Inapplicable',
            -7: 'Refused',
            -8: "Don't know"
        },
        'poverty_category_2022': {
            1: 'Poor/negative',
            2: 'Near poor',
            3: 'Low income',
            4: 'Middle income',
            5: 'High income'
        },
        'insurance_coverage_2022': {
            1: 'Any private',
            2: 'Public only',
            3: 'Uninsured'
        },
        'perceived_health_status': {
            1: 'Excellent',
            2: 'Very good',
            3: 'Good',
            4: 'Fair',
            5: 'Poor',
            -1: 'Inapplicable',
            -7: 'Refused',
            -8: "Don't know"
        },
        'insurance_coverage': {
            0: 'No',
            1: 'Yes'
        },
        'injury_flag': {
            0: 'No',
            1: 'Yes'
        },
        'insurance_offered': {
            1: 'Yes',
            2: 'No',
            -1: 'Inapplicable',
            -7: 'Refused',
            -8: "Don't know",
            -9: 'Not ascertained'
        },
        'temporary_job': {
            1: 'Yes',
            2: 'No',
            -1: 'Inapplicable',
            -7: 'Refused',
            -8: "Don't know",
            -9: 'Not ascertained'
        },
        'salaried_employee': {
            1: 'Yes',
            2: 'No',
            3: 'Unknown/Not reported',
            -1: 'Inapplicable',
            -7: 'Refused',
            -8: "Don't know",
            -9: 'Not ascertained'
        }
    }

def clean_numeric_field(value):
    """Limpiar campos numéricos y manejar códigos especiales"""
    if pd.isna(value) or value == '' or value == ' ':
        return np.nan
    
    # Convertir a string para procesamiento
    str_value = str(value).strip()
    
    # Manejar códigos reservados
    reserved_codes = get_reserved_codes()
    try:
        numeric_value = float(str_value)
        if numeric_value in reserved_codes:
            return reserved_codes[numeric_value]
        return numeric_value
    except:
        return str_value

def apply_categorical_mapping(df, column, mapping_key):
    """Aplicar mapeo categórico a una columna"""
    mappings = get_value_mappings()
    if mapping_key in mappings:
        mapping = mappings[mapping_key]
        df[column] = df[column].apply(lambda x: mapping.get(x, x) if pd.notna(x) else x)
    return df

def process_fyc_data(df):
    """Procesar datos del archivo Full Year Consolidated"""
    print("Procesando datos FYC (Full Year Consolidated)...")
    
    # Renombrar columnas
    column_mappings = get_column_mappings()
    df = df.rename(columns=column_mappings)
    
    # Limpiar campos numéricos
    numeric_fields = ['age_last_birthday', 'total_healthcare_exp_2022', 'total_out_of_pocket_exp_2022', 'person_weight_2022']
    for field in numeric_fields:
        if field in df.columns:
            df[field] = df[field].apply(clean_numeric_field)
    
    # Aplicar mapeos categóricos
    categorical_mappings = [
        ('sex', 'sex'),
        ('race_ethnicity', 'race_ethnicity'),
        ('marital_status_2022', 'marital_status_2022'),
        ('region_2022', 'region_2022'),
        ('poverty_category_2022', 'poverty_category_2022'),
        ('insurance_coverage_2022', 'insurance_coverage_2022'),
        ('perceived_health_status', 'perceived_health_status')
    ]
    
    for column, mapping_key in categorical_mappings:
        if column in df.columns:
            # Convertir a numérico primero
            df[column] = pd.to_numeric(df[column], errors='coerce')
            df = apply_categorical_mapping(df, column, mapping_key)
    
    return df

def process_cond_data(df):
    """Procesar datos del archivo Medical Conditions"""
    print("Procesando datos COND (Medical Conditions)...")
    
    # Renombrar columnas
    column_mappings = get_column_mappings()
    df = df.rename(columns=column_mappings)
    
    # Limpiar campos numéricos
    numeric_fields = ['age_at_diagnosis']
    for field in numeric_fields:
        if field in df.columns:
            df[field] = df[field].apply(clean_numeric_field)
    
    # Aplicar mapeos categóricos
    if 'injury_flag' in df.columns:
        df['injury_flag'] = pd.to_numeric(df['injury_flag'], errors='coerce')
        df = apply_categorical_mapping(df, 'injury_flag', 'injury_flag')
    
    # Ya no se debe mapear los nombres reales de medicamentos/condiciones aquí
    # Solo limpiar y mapear el resto de campos
    return df

def process_prpl_data(df):
    """Procesar datos del archivo Private Insurance"""
    print("Procesando datos PRPL (Private Insurance)...")
    
    # Renombrar columnas
    column_mappings = get_column_mappings()
    df = df.rename(columns=column_mappings)
    
    # Limpiar campos numéricos
    numeric_fields = ['out_of_pocket_premium', 'out_of_pocket_premium_edited']
    for field in numeric_fields:
        if field in df.columns:
            df[field] = df[field].apply(clean_numeric_field)
    
    # Aplicar mapeos categóricos
    if 'insurance_coverage' in df.columns:
        df['insurance_coverage'] = pd.to_numeric(df['insurance_coverage'], errors='coerce')
        df = apply_categorical_mapping(df, 'insurance_coverage', 'insurance_coverage')
    
    return df

def process_jobs_data(df):
    """Procesar datos del archivo Jobs"""
    print("Procesando datos JOBS (Jobs)...")
    
    # Renombrar columnas
    column_mappings = get_column_mappings()
    df = df.rename(columns=column_mappings)
    
    # Limpiar campos numéricos
    numeric_fields = ['hourly_wage', 'hours_per_week']
    for field in numeric_fields:
        if field in df.columns:
            df[field] = df[field].apply(clean_numeric_field)
    
    # Aplicar mapeos categóricos
    categorical_mappings = [
        ('insurance_offered', 'insurance_offered'),
        ('temporary_job', 'temporary_job'),
        ('salaried_employee', 'salaried_employee')
    ]
    
    for column, mapping_key in categorical_mappings:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
            df = apply_categorical_mapping(df, column, mapping_key)
    
    return df

def process_all_meps_data(data_dir='data'):
    """Procesar todos los archivos MEPS y guardar versiones legibles"""
    print("="*60)
    print("Procesando y Mapeando Datos MEPS 2022")
    print("="*60)
    
    processed_data = {}
    
    # Procesar FYC
    fyc_path = os.path.join(data_dir, 'meps_fyc_2022.csv')
    if os.path.exists(fyc_path):
        fyc_df = pd.read_csv(fyc_path)
        fyc_processed = process_fyc_data(fyc_df)
        processed_data['fyc'] = fyc_processed
        
        # Guardar versión procesada
        output_path = os.path.join(data_dir, 'meps_fyc_2022_processed.csv')
        fyc_processed.to_csv(output_path, index=False)
        print(f"✓ FYC procesado y guardado en: {output_path}")
    
    # Procesar COND
    cond_path = os.path.join(data_dir, 'meps_cond_2022.csv')
    if os.path.exists(cond_path):
        cond_df = pd.read_csv(cond_path)
        cond_processed = process_cond_data(cond_df)
        processed_data['cond'] = cond_processed
        
        # Guardar versión procesada
        output_path = os.path.join(data_dir, 'meps_cond_2022_processed.csv')
        cond_processed.to_csv(output_path, index=False)
        print(f"✓ COND procesado y guardado en: {output_path}")
    
    # Procesar PRPL
    prpl_path = os.path.join(data_dir, 'meps_prpl_2022.csv')
    if os.path.exists(prpl_path):
        prpl_df = pd.read_csv(prpl_path)
        prpl_processed = process_prpl_data(prpl_df)
        processed_data['prpl'] = prpl_processed
        
        # Guardar versión procesada
        output_path = os.path.join(data_dir, 'meps_prpl_2022_processed.csv')
        prpl_processed.to_csv(output_path, index=False)
        print(f"✓ PRPL procesado y guardado en: {output_path}")
    
    # Procesar JOBS
    jobs_path = os.path.join(data_dir, 'meps_jobs_2022.csv')
    if os.path.exists(jobs_path):
        jobs_df = pd.read_csv(jobs_path)
        jobs_processed = process_jobs_data(jobs_df)
        processed_data['jobs'] = jobs_processed
        
        # Guardar versión procesada
        output_path = os.path.join(data_dir, 'meps_jobs_2022_processed.csv')
        jobs_processed.to_csv(output_path, index=False)
        print(f"✓ JOBS procesado y guardado en: {output_path}")
    
    print("\n" + "="*60)
    print("Procesamiento completado exitosamente!")
    print("="*60)
    
    return processed_data

def generate_data_summary(processed_data):
    """Generar resumen de los datos procesados"""
    print("\n" + "="*60)
    print("RESUMEN DE DATOS PROCESADOS")
    print("="*60)
    
    for dataset_name, df in processed_data.items():
        print(f"\n{dataset_name.upper()} Dataset:")
        print(f"  - Filas: {len(df):,}")
        print(f"  - Columnas: {len(df.columns)}")
        print(f"  - Columnas: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
        
        # Mostrar algunos estadísticos básicos
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"  - Columnas numéricas: {len(numeric_cols)}")
            print(f"  - Valores nulos: {df[numeric_cols].isnull().sum().sum():,}")

if __name__ == "__main__":
    # Procesar todos los datos
    processed_data = process_all_meps_data()
    
    # Generar resumen
    generate_data_summary(processed_data)
