import os
import json
import re
import unicodedata
import pandas as pd
from dotenv import load_dotenv

# 1. Cargar configuración desde el .env
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
GOLDEN_PATH = os.getenv("GOLDEN_STANDARD_PATH", "data/golden_standard.json")
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")


def normalizar_texto(texto: str) -> str:
    """
    Limpia cadenas para comparar sin penalizar errores humanos:
    - Pasa a minúsculas.
    - Quita espacios iniciales y finales.
    - Reemplaza dobles espacios internos o tabuladores por un único espacio.
    - Elimina tildes y diacríticos (ej: 'Núria' -> 'nuria').
    """
    if not isinstance(texto, str):
        return ""
    
    # Minúsculas y quitar espacios en extremos
    texto = texto.strip().lower()
    
    # Sustituir dos o más espacios consecutivos por uno solo
    texto = re.sub(r"\s+", " ", texto)
    
    # Eliminar acentos y diacríticos
    texto = "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )
    
    return texto


def calcular_metricas_presencia(lista_predicha, lista_real):
    """Calcula TP, FP, FN, Precision, Recall y F1 para nombres identificados."""
    pred_set = set(normalizar_texto(n) for n in lista_predicha if n)
    real_set = set(normalizar_texto(n) for n in lista_real if n)

    # Eliminar posibles strings vacíos generados tras normalizar
    pred_set.discard("")
    real_set.discard("")

    tp = len(pred_set.intersection(real_set))
    fp = len(pred_set - real_set)
    fn = len(real_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return tp, fp, fn, precision, recall, f1


def ejecutar_calculo():
    if not MODEL_NAME:
        raise ValueError("La variable MODEL_NAME no está configurada en el archivo .env.")

    nombre_carpeta_modelo = MODEL_NAME.replace("/", "_")
    carpeta_modelo = os.path.join(RESULTS_DIR, nombre_carpeta_modelo)
    ruta_predicciones = os.path.join(carpeta_modelo, "predicciones.json")

    # Validar existencia de ficheros
    if not os.path.exists(GOLDEN_PATH):
        raise FileNotFoundError(f"No se encuentra el Golden Standard en: {GOLDEN_PATH}")
    if not os.path.exists(ruta_predicciones):
        raise FileNotFoundError(f"No se encuentran las predicciones en: {ruta_predicciones}. Ejecuta primero evaluacion_modelos.py")

    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    with open(ruta_predicciones, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    # Mapear predicciones por ID de artículo
    pred_dict = {
        item["id"]: item.get("prediccio", item.get("prediccion", {})) 
        for item in pred_data
    }

    total_tp = 0
    total_fp = 0
    total_fn = 0
    coincidencias_titular = 0
    total_articulos_evaluados = 0

    detalles_articulos = []

    for item in golden_data:
        art_id = item.get("id")
        if art_id not in pred_dict:
            continue

        total_articulos_evaluados += 1
        anotacion_real = item.get("anotacion_humana", {})
        pred = pred_dict[art_id]

        mujeres_reales = anotacion_real.get("mujeres_mencionadas", [])
        mujeres_pred = pred.get("mujeres_mencionadas", [])

        # 1. Comparar entidades nombradas (presencia de mujeres)
        tp, fp, fn, prec, rec, f1 = calcular_metricas_presencia(mujeres_pred, mujeres_reales)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # 2. Comparar presencia en titular o apertura
        titular_real = anotacion_real.get("mujer_en_titular_o_apertura", False)
        titular_pred = pred.get("mujer_en_titular_o_apertura", False)
        acierto_titular = (titular_real == titular_pred)
        if acierto_titular:
            coincidencias_titular += 1

        detalles_articulos.append({
            "id": art_id,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision_nombres": round(prec, 3),
            "recall_nombres": round(rec, 3),
            "f1_nombres": round(f1, 3),
            "acierto_titular": acierto_titular
        })

    # Métricas agregadas globales
    precision_global = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall_global = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_global = (2 * precision_global * recall_global) / (precision_global + recall_global) if (precision_global + recall_global) > 0 else 0.0
    exactitud_titular = (coincidencias_titular / total_articulos_evaluados) if total_articulos_evaluados > 0 else 0.0

    resumen_modelo = {
        "modelo": MODEL_NAME,
        "articulos_evaluados": total_articulos_evaluados,
        "precision_mujeres": round(precision_global, 4),
        "recall_mujeres": round(recall_global, 4),
        "f1_mujeres": round(f1_global, 4),
        "exactitud_titular_apertura": round(exactitud_titular, 4),
        "verdaderos_positivos": total_tp,
        "falsos_positivos": total_fp,
        "falsos_negativos": total_fn
    }

    # 1. Guardar resumen individual en la subcarpeta del modelo
    ruta_metricas_json = os.path.join(carpeta_modelo, "metricas.json")
    with open(ruta_metricas_json, "w", encoding="utf-8") as f:
        json.dump(resumen_modelo, f, ensure_ascii=False, indent=2)

    # 2. Guardar desglose por artículo en CSV para auditoría
    df_detalles = pd.DataFrame(detalles_articulos)
    ruta_detalles_csv = os.path.join(carpeta_modelo, "desglose_por_articulo.csv")
    df_detalles.to_csv(ruta_detalles_csv, index=False, encoding="utf-8")

    # 3. Actualizar la tabla general comparativa de todos los modelos
    ruta_comparativa_csv = os.path.join(RESULTS_DIR, "comparativa_modelos.csv")
    if os.path.exists(ruta_comparativa_csv):
        df_comp = pd.read_csv(ruta_comparativa_csv)
        # Si el modelo ya estaba registrado, sustituye su fila por los datos nuevos
        df_comp = df_comp[df_comp["modelo"] != MODEL_NAME]
        df_nuevo = pd.DataFrame([resumen_modelo])
        df_comp = pd.concat([df_comp, df_nuevo], ignore_index=True)
    else:
        df_comp = pd.DataFrame([resumen_modelo])

    df_comp.to_csv(ruta_comparativa_csv, index=False, encoding="utf-8")

    # Mostrar reporte en consola
    print("\n" + "=" * 55)
    print(f" RESULTADOS DEL BENCHMARK: {MODEL_NAME}")
    print("=" * 55)
    print(f" Artículos evaluados:        {total_articulos_evaluados}")
    print(f" Aciertos (TP):              {total_tp}")
    print(f" Falsas alarmas (FP):        {total_fp}")
    print(f" Omitidas / No vistas (FN):  {total_fn}")
    print("-" * 55)
    print(f" Precisión (Mujeres):        {precision_global:.2%}")
    print(f" Recall (Mujeres):           {recall_global:.2%}")
    print(f" F1-Score (Mujeres):         {f1_global:.2%}")
    print(f" Exactitud Titular/Apertura: {exactitud_titular:.2%}")
    print("=" * 55)
    print(f"-> Fichero individual:   {ruta_metricas_json}")
    print(f"-> Tabla comparativa:    {ruta_comparativa_csv}\n")


if __name__ == "__main__":
    ejecutar_calculo()