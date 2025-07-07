import json
import os
from collections import defaultdict

# Ruta de entrada y salida
INPUT_JSON = os.path.join('..', 'meps_2022_unified_reduced.json')
OUTPUT_JSON = os.path.join('snapshots.json')

# Lista de condiciones crónicas por descripción CCSR (puedes expandirla)
CRONICAS_CCSR = set([
    'Diabetes mellitus',
    'Hypertension',
    'Asthma',
    'COPD',
    'Chronic kidney disease',
    'Heart failure',
    'Ischemic heart disease',
    'HIV',
    'Cancer',
    # Agrega más según tu referencia CCSR
])

def cargar_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generar_snapshots(unified_data):
    snapshots = []
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
            es_cronica = desc in CRONICAS_CCSR
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
    print('Generando snapshots...')
    snapshots = generar_snapshots(unified_data)
    print(f'Se generaron {len(snapshots)} snapshots.')
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(snapshots, f, ensure_ascii=False, indent=2)
    print(f'Snapshots guardados en {OUTPUT_JSON}')

if __name__ == '__main__':
    main()
