import json
import os
import pandas as pd
from collections import defaultdict


# Ruta de entrada y salida (ajustada para que siempre apunte al archivo correcto)
INPUT_JSON = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'meps_2022_unified_reduced.json'))
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), 'snapshots.json')

# Rutas de archivos de referencia
CCIR_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'info', 'CCIR_v2025-1.csv'))
CCSR_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'info', 'ccsr_reference_2025.csv'))

def cargar_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def cargar_mapeos_cronicos():
    """Cargar mapeos para identificar condiciones crónicas"""
    print("Cargando mapeos de condiciones crónicas...")
    
    # Cargar CCIR para mapeo de ICD10 a indicador crónico
    try:
        df_ccir = pd.read_csv(CCIR_FILE, skiprows=2)  # Saltar las 2 primeras filas de header
        print(f"✓ CCIR cargado con {len(df_ccir)} registros")
    except FileNotFoundError:
        # Intentar ruta alternativa
        alt_ccir = os.path.join('data', 'info', 'CCIR_v2025-1.csv')
        try:
            df_ccir = pd.read_csv(alt_ccir, skiprows=2)
            print(f"✓ CCIR cargado con {len(df_ccir)} registros")
        except FileNotFoundError:
            print(f"⚠️  No se encontró el archivo CCIR en ninguna ubicación")
            return None
    
    # Cargar CCSR reference para mapeo de descripción a códigos ICD10
    try:
        df_ccsr = pd.read_csv(CCSR_FILE)
        print(f"✓ CCSR reference cargado con {len(df_ccsr)} registros")
    except FileNotFoundError:
        # Intentar ruta alternativa
        alt_ccsr = os.path.join('data', 'info', 'ccsr_reference_2025.csv')
        try:
            df_ccsr = pd.read_csv(alt_ccsr)
            print(f"✓ CCSR reference cargado con {len(df_ccsr)} registros")
        except FileNotFoundError:
            print(f"⚠️  No se encontró el archivo CCSR en ninguna ubicación")
            return None
    
    # Crear mapeo de ICD10 a indicador crónico desde CCIR
    ccir_chronic_map = {}
    
    # Buscar las columnas correctas (pueden tener comillas)
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
    
    print(f"✓ Mapeo CCIR creado con {len(ccir_chronic_map)} códigos ICD10")
    
    if len(ccir_chronic_map) > 0:
        cronicos_count = sum(1 for is_chronic in ccir_chronic_map.values() if is_chronic)
        print(f"   Códigos crónicos encontrados: {cronicos_count}")
    
    # Crear mapeo de descripción CCSR a códigos ICD10
    ccsr_to_icd10_map = {}
    for _, row in df_ccsr.iterrows():
        ccsr_desc = row.get('CCSR Category Description', '')
        icd10_code = row.get('ICD-10-CM Code', '')
        if pd.notna(ccsr_desc) and pd.notna(icd10_code) and ccsr_desc and icd10_code:
            if ccsr_desc not in ccsr_to_icd10_map:
                ccsr_to_icd10_map[ccsr_desc] = set()
            ccsr_to_icd10_map[ccsr_desc].add(icd10_code)
    
    print(f"✓ Mapeo CCSR a ICD10 creado con {len(ccsr_to_icd10_map)} descripciones")
    
    return ccir_chronic_map, ccsr_to_icd10_map

def es_condicion_cronica(descripcion_ccsr, ccir_chronic_map, ccsr_to_icd10_map):
    """Verificar si una condición es crónica basándose en su descripción CCSR"""
    if not descripcion_ccsr or descripcion_ccsr == "No especificado":
        return False
    
    # Buscar códigos ICD10 asociados a esta descripción CCSR
    icd10_codes = ccsr_to_icd10_map.get(descripcion_ccsr, set())
    
    if not icd10_codes:
        return False
    
    # Verificar si algún código ICD10 tiene CHRONIC INDICATOR = 1
    for icd10_code in icd10_codes:
        if ccir_chronic_map.get(icd10_code, False):
            return True
    
    return False

def generar_snapshots(unified_data, ccir_chronic_map, ccsr_to_icd10_map):
    snapshots = []
    
    condiciones_cronicas_cache = {}  # Cache para evitar recalcular
    
    for person_id, person in unified_data.items():
        # Obtener todos los rounds de seguro con prima válida
        rounds_seguros = [
            int(seg['round_reportado'])
            for seg in person['historial_seguros']
            if seg.get('round_reportado') is not None and str(seg.get('round_reportado')).isdigit()
        ]
        if not rounds_seguros:
            continue
        rounds_seguros = sorted(set(rounds_seguros))

        # Preprocesar condiciones médicas: lista de (desc, round, es_cronica, resto)
        condiciones = []
        for cond in person['condiciones_medicas']:
            desc = cond.get('descripcion_ccsr')
            round_diag = cond.get('round_reportado')
            if round_diag is None or not str(round_diag).isdigit():
                continue
            round_diag = int(round_diag)
            
            # Verificar si es crónica usando cache
            if desc not in condiciones_cronicas_cache:
                condiciones_cronicas_cache[desc] = es_condicion_cronica(desc, ccir_chronic_map, ccsr_to_icd10_map)
            
            es_cronica = condiciones_cronicas_cache[desc]
            condiciones.append((desc, round_diag, es_cronica, cond))

        # Acumular condiciones crónicas hasta cada round
        for round_actual in rounds_seguros:
            snapshot = {
                'person_unique_id': person_id,
                'round': round_actual,
                'edad': person['edad'],
                'sexo': person['sexo'],
                'raza_etnicidad': person['raza_etnicidad'],
                'estado_civil': person['estado_civil'],
                'region': person['region'],
                'gastos_medicos_totales': person['gastos_medicos_totales'],
                'gastos_out_of_pocket': person['gastos_out_of_pocket'],
                'categoria_pobreza': person['categoria_pobreza'],
                'cobertura_seguro': person['cobertura_seguro'],
                'estado_salud_percibido': person['estado_salud_percibido'],
                'peso_persona': person['peso_persona'],
                'condiciones_medicas': [],
                'historial_empleo': [emp for emp in person['historial_empleo'] if emp.get('round_reportado') is not None and str(emp.get('round_reportado')).isdigit() and int(emp['round_reportado']) <= round_actual],
                'seguro': next((seg for seg in person['historial_seguros'] if int(seg['round_reportado']) == round_actual), None),
            }
            # Acumular condiciones hasta el round actual
            cronicas_presentes = {}
            for desc, round_diag, es_cronica, cond in condiciones:
                if round_diag > round_actual:
                    continue
                if es_cronica:
                    # Si es crónica, arrastrar desde el diagnóstico
                    if desc not in cronicas_presentes:
                        copia = cond.copy()
                        copia['round_reportado'] = round_actual
                        cronicas_presentes[desc] = copia
                else:
                    # No crónica: solo si fue diagnosticada en este round
                    if round_diag == round_actual:
                        snapshot['condiciones_medicas'].append(cond)
            # Agregar crónicas presentes
            snapshot['condiciones_medicas'].extend(list(cronicas_presentes.values()))
            # Solo agregar si hay seguro válido en ese round
            if snapshot['seguro'] is not None:
                snapshots.append(snapshot)
    return snapshots

def main():
    print('Cargando datos...')
    unified_data = cargar_json(INPUT_JSON)
    
    print('Cargando mapeos de condiciones crónicas...')
    mapeos_result = cargar_mapeos_cronicos()
    if mapeos_result is None:
        print('❌ No se pudieron cargar los mapeos de condiciones crónicas')
        return
    
    ccir_chronic_map, ccsr_to_icd10_map = mapeos_result
    
    print('Generando snapshots...')
    snapshots = generar_snapshots(unified_data, ccir_chronic_map, ccsr_to_icd10_map)
    
    print(f'Se generaron {len(snapshots)} snapshots.')
    
    # Estadísticas de condiciones crónicas
    total_condiciones = sum(len(snapshot['condiciones_medicas']) for snapshot in snapshots)
    condiciones_cronicas = sum(
        1 for snapshot in snapshots 
        for cond in snapshot['condiciones_medicas']
        if es_condicion_cronica(cond.get('descripcion_ccsr'), ccir_chronic_map, ccsr_to_icd10_map)
    )
    
    print(f'Total de condiciones médicas en snapshots: {total_condiciones}')
    print(f'Condiciones crónicas encontradas: {condiciones_cronicas}')
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(snapshots, f, ensure_ascii=False, indent=2)
    print(f'Snapshots guardados en {OUTPUT_JSON}')

if __name__ == '__main__':
    main()
