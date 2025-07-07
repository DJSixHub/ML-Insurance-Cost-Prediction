import json
import pandas as pd
import numpy as np

# Lista de enfermedades importantes
IMPORTANT_DISEASES = [
    "Coronary atherosclerosis and other heart disease",
    "Diabetes mellitus without complication",
    "Disorders of lipid metabolism",
    "Esophageal disorders",
    "Essential hypertension",
    "Exposure, encounters, screening or contact with infectious disease",
    "Musculoskeletal pain, not low back pain",
    "Neurodevelopmental disorders",
    "Osteoarthritis",
    "Thyroid disorders"
]

def flatten_snapshot(snapshot):
    flat = {
        'person_unique_id': snapshot.get('person_unique_id'),
        # 'round': snapshot.get('round'),  # Eliminado del CSV final
        'edad': snapshot.get('edad'),
        'sexo': snapshot.get('sexo'),
        'raza_etnicidad': snapshot.get('raza_etnicidad'),
        'estado_civil': snapshot.get('estado_civil'),
        'region': snapshot.get('region'),
        'gastos_medicos_totales': snapshot.get('gastos_medicos_totales'),
        'gastos_out_of_pocket': snapshot.get('gastos_out_of_pocket'),
        'categoria_pobreza': snapshot.get('categoria_pobreza'),
        'cobertura_seguro': snapshot.get('cobertura_seguro'),
        'estado_salud_percibido': snapshot.get('estado_salud_percibido'),
        'peso_persona': snapshot.get('peso_persona'),
    }
    # Enfermedades importantes como features binarias
    enfermedades = [c.get('descripcion_ccsr') for c in snapshot.get('condiciones_medicas', [])]
    for disease in IMPORTANT_DISEASES:
        flat[f'disease_{disease[:30].replace(" ", "_").replace(",", "").replace("-", "").lower()}'] = int(disease in enfermedades)
    # Número total de condiciones
    flat['num_condiciones_medicas'] = len(enfermedades)
    # Feature: ¿tiene alguna enfermedad crónica?
    flat['tiene_enfermedad_cronica'] = int(len(enfermedades) > 0)
    # Historial empleo: número de empleos y promedio de horas
    empleos = snapshot.get('historial_empleo', [])
    flat['num_empleos'] = len(empleos)
    horas = [float(e['horas_por_semana']) for e in empleos if 'horas_por_semana' in e and str(e['horas_por_semana']).replace('.','',1).isdigit()]
    flat['horas_trabajadas_promedio'] = np.mean(horas) if horas else 0
    # Seguro
    seguro = snapshot.get('seguro', {})
    flat['cobertura_seguro_detallada'] = seguro.get('cobertura_seguro')
    # Eliminado: no incluir prima_out_of_pocket en el CSV final
    try:
        flat['prima_out_of_pocket_editada'] = float(seguro.get('prima_out_of_pocket_editada', 0))
    except:
        flat['prima_out_of_pocket_editada'] = np.nan
    return flat

def main():
    # Cargar snapshots
    with open('Feature Engineering/snapshots.json', 'r', encoding='utf-8') as f:
        snapshots = json.load(f)
    # Aplanar
    flat_records = [flatten_snapshot(s) for s in snapshots]
    df = pd.DataFrame(flat_records)
    # Guardar CSV
    df.to_csv('snapshots_flat.csv', index=False)
    print(f"✅ Datos aplanados y guardados en snapshots_flat.csv. Total de registros: {len(df)}")

if __name__ == "__main__":
    main()
