"""
Descargador de Datos MEPS
Script automatizado para descargar y extraer datos MEPS 2022 en bruto (RAW).
"""

import requests
import os
import zipfile
import time
import pandas as pd

def download_file(url, filename, directory):
    """Descargar un archivo desde URL"""
    try:
        print(f"Descargando {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✓ Descargado {filename}")
        return filepath
    except Exception as e:
        print(f"✗ Error descargando {filename}: {e}")
        return None

def extract_zip(zip_path, extract_to):
    """Extraer archivo ZIP"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✓ Extraído {os.path.basename(zip_path)}")
        return True
    except Exception as e:
        print(f"✗ Error extrayendo {zip_path}: {e}")
        return False

def load_h243_file(filepath):
    """Cargar archivo H243 (Full Year Consolidated) RAW"""
    column_specs = [
        ('DUID', 1, 7),           # @1, 7.0
        ('PID', 8, 10),           # @8, 3.0  
        ('DUPERSID', 11, 20),     # @11, $10.0
        ('PANEL', 21, 22),        # @21, 2.0
        ('AGELAST', 194, 195),    # @194, 2.0 (edad última cumpleaños)
        ('SEX', 202, 202),        # @202, 1.0 (sexo)
        ('RACETHX', 209, 209),    # @209, 1.0 (raza/etnicidad)
        ('MARRY22X', 218, 219),   # @218, 2.0 (estado civil)
        ('REGION22', 88, 89),     # @88, 2.0 (región)
        ('TOTEXP22', 2616, 2622), # @2616, 7.0 (gastos totales)
        ('TOTSLF22', 2623, 2628), # @2623, 6.0 (gastos propios)
        ('POVCAT22', 1466, 1466), # @1466, 1.0 (categoría pobreza)
        ('INSCOV22', 2235, 2235), # @2235, 1.0 (cobertura seguro)
        ('RTHLTH53', 425, 426),   # @425, 2.0 (estado salud)
        ('PERWT22F', 4001, 4013)  # @4001, 13.6 (peso persona)
    ]
    colspecs = [(start-1, end) for _, start, end in column_specs]
    names = [name for name, _, _ in column_specs]
    df = pd.read_fwf(filepath, colspecs=colspecs, names=names, dtype=str)
    return df

def load_h242_file(filepath):
    """Cargar archivo H242 (Person Round Plan) RAW"""
    column_specs = [
        ('DUPERSID', 36, 45),     # @36, $10.0
        ('PANEL', 106, 107),      # @106, 2.0
        ('RN', 108, 108),         # @108, 1.0
        ('INSCOV', 134, 134),     # Usando PHOLDER @134, 1.0 (policy holder)
        ('OOPPREM', 192, 199),    # @192, 8.2 (Monthly out-of-pocket premium)
        ('OOPPREMX', 200, 206),   # @200, 7.2 (Monthly out-of-pocket premium edited/imputed)
    ]
    colspecs = [(start-1, end) for _, start, end in column_specs]
    names = [name for name, _, _ in column_specs]
    df = pd.read_fwf(filepath, colspecs=colspecs, names=names, dtype=str)
    return df

def load_h241_file(filepath):
    """Cargar archivo H241 (Medical Conditions) RAW"""
    column_specs = [
        ('DUPERSID', 11, 20),     # @11, $10.0
        ('CONDIDX', 23, 35),      # @23, $13.0
        ('PANEL', 36, 37),        # @36, 2.0
        ('CONDRN', 38, 38),       # @38, 1.0
        ('AGEDIAG', 39, 41),      # @39, 3.0
        ('INJURY', 58, 58),       # Estimando posición para injury
        ('ICD10CDX', 64, 71),     # @64, $8.0
        ('CCSR1X', 72, 77),       # @72, $6.0
    ]
    colspecs = [(start-1, end) for _, start, end in column_specs]
    names = [name for name, _, _ in column_specs]
    df = pd.read_fwf(filepath, colspecs=colspecs, names=names, dtype=str)
    return df

def load_h237_file(filepath):
    """Cargar archivo H237 (Jobs) RAW"""
    column_specs = [
        ('DUPERSID', 42, 51),     # @42, $10.0
        ('JOBSIDX', 1, 14),       # @1, $14.0
        ('PANEL', 64, 65),        # @64, 2.0
        ('RN', 62, 62),           # @62, 1.0
        ('OFFRDINS', 164, 165),   # @164, 2.0 (Offered insurance but chose not to take)
        ('TEMPJOB', 141, 142),    # @141, 2.0 (Temporary job)
        ('SALARIED', 188, 189),   # @188, 2.0 (Salaried worker)
        ('HRLYWAGE', 211, 216),   # @211, 6.2 (How much person makes per hour)
        ('HRSPRWK', 136, 138),    # @136, 3.0 (Number of hours worked per week)
    ]
    colspecs = [(start-1, end) for _, start, end in column_specs]
    names = [name for name, _, _ in column_specs]
    df = pd.read_fwf(filepath, colspecs=colspecs, names=names, dtype=str)
    return df

def download_and_process_meps_data():
    """Función principal para descargar datos MEPS en bruto"""
    print("="*60)
    print("Descargador de Datos MEPS 2022 (RAW)")
    print("="*60)
    
    processed_files = [
        os.path.join('data', 'meps_fyc_2022.csv'),
        os.path.join('data', 'meps_prpl_2022.csv'),
        os.path.join('data', 'meps_cond_2022.csv'),
        os.path.join('data', 'meps_jobs_2022.csv')
    ]
    
    if all(os.path.exists(f) for f in processed_files):
        print("✓ ¡Archivos de datos MEPS RAW existentes encontrados!")
        for f in processed_files:
            if os.path.exists(f):
                df = pd.read_csv(f)
                print(f"  - {f}: {len(df)} registros")
        return True
    
    files_to_download = {
        'HC-243': {
            'name': 'Archivo Consolidado de Año Completo',
            'url': 'https://meps.ahrq.gov/mepsweb/data_files/pufs/h243/h243dat.zip',
            'sas_url': 'https://meps.ahrq.gov/data_stats/download_data/pufs/h243/h243su.txt'
        },
        'HC-242': {
            'name': 'Archivo de Plan de Ronda por Persona',
            'url': 'https://meps.ahrq.gov/mepsweb/data_files/pufs/h242/h242dat.zip',
            'sas_url': 'https://meps.ahrq.gov/data_stats/download_data/pufs/h242/h242su.txt'
        },
        'HC-241': {
            'name': 'Archivo de Condiciones Médicas',
            'url': 'https://meps.ahrq.gov/mepsweb/data_files/pufs/h241/h241dat.zip',
            'sas_url': 'https://meps.ahrq.gov/data_stats/download_data/pufs/h241/h241su.txt'
        },
        'HC-237': {
            'name': 'Archivo de Empleos',
            'url': 'https://meps.ahrq.gov/mepsweb/data_files/pufs/h237/h237dat.zip',
            'sas_url': 'https://meps.ahrq.gov/data_stats/download_data/pufs/h237/h237su.txt'
        }
    }
    
    raw_dir = 'data/raw'
    os.makedirs(raw_dir, exist_ok=True)
    
    # Descargar y extraer archivos
    for file_id, info in files_to_download.items():
        print(f"\nProcesando {file_id}: {info['name']}")
        zip_filename = f"{file_id.lower()}.zip"
        zip_path = download_file(info['url'], zip_filename, raw_dir)
        
        if zip_path:
            extract_dir = os.path.join(raw_dir, file_id)
            extract_zip(zip_path, extract_dir)
            os.remove(zip_path)
            sas_filename = f"{file_id.lower()}_sas.txt"
            download_file(info['sas_url'], sas_filename, raw_dir)
        
        time.sleep(1)
    
    # Guardar archivos RAW como CSV
    print(f"\n{'='*60}")
    print("Extrayendo y guardando archivos RAW como CSV...")
    print("="*60)
    
    # 1. Full Year Consolidated
    fyc_path = os.path.join(raw_dir, 'HC-243', 'h243.dat')
    if os.path.exists(fyc_path):
        fyc_df = load_h243_file(fyc_path)
        fyc_df.to_csv('data/meps_fyc_2022.csv', index=False)
        print("✓ Guardado: data/meps_fyc_2022.csv (RAW)")
    
    # 2. Person Round Plan
    prpl_path = os.path.join(raw_dir, 'HC-242', 'h242.dat')
    if os.path.exists(prpl_path):
        prpl_df = load_h242_file(prpl_path)
        prpl_df.to_csv('data/meps_prpl_2022.csv', index=False)
        print("✓ Guardado: data/meps_prpl_2022.csv (RAW)")
    
    # 3. Medical Conditions
    cond_path = os.path.join(raw_dir, 'HC-241', 'h241.dat')
    if os.path.exists(cond_path):
        cond_df = load_h241_file(cond_path)
        cond_df.to_csv('data/meps_cond_2022.csv', index=False)
        print("✓ Guardado: data/meps_cond_2022.csv (RAW)")
    
    # 4. Jobs
    jobs_path = os.path.join(raw_dir, 'HC-237', 'h237.dat')
    if os.path.exists(jobs_path):
        jobs_df = load_h237_file(jobs_path)
        jobs_df.to_csv('data/meps_jobs_2022.csv', index=False)
        print("✓ Guardado: data/meps_jobs_2022.csv (RAW)")
    
    print(f"\n{'='*60}")
    print("¡DESCARGA Y EXTRACCIÓN DE ARCHIVOS RAW COMPLETA!")
    print("="*60)
    print("\nArchivos RAW creados:")
    print("- data/meps_fyc_2022.csv   (RAW)")
    print("- data/meps_prpl_2022.csv  (RAW)")
    print("- data/meps_cond_2022.csv  (RAW)")
    print("- data/meps_jobs_2022.csv  (RAW)")
    print("\n✅ Listos para procesamiento posterior en mapeo.py")
    return True

if __name__ == "__main__":
    success = download_and_process_meps_data()
    if success:
        print("\n✓ ¡Todos los archivos RAW están listos para el pipeline de procesamiento!")
    else:
        print("\n✗ Ocurrieron algunos problemas durante la descarga. Revisa los logs arriba.")
