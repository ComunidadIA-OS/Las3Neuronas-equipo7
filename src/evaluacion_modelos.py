import os
import json
import re
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# 1. Cargar las variables de entorno desde el .env
load_dotenv()

API_KEY = os.getenv("OVH_API_KEY")
API_BASE = os.getenv("OVH_API_BASE", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Meta-Llama-3_3-70B-Instruct")
INPUT_PATH = os.getenv("INPUT_DATA_PATH", "data/dataset_15.json")
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")

# 2. Inicializar cliente compatible con OpenAI (OVHcloud)
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE
)

# 3. Prompt de extracción con reglas de perspectiva de género
SYSTEM_PROMPT = """Eres un analizador de igualdad de género en textos de prensa.
Analiza el titular y el texto proporcionados y responde ÚNICAMENTE con un objeto JSON válido con esta estructura:
{
  "total_personas_mencionadas": int,
  "mujeres_mencionadas": [str],
  "hombres_mencionados": [str],
  "mujer_en_titular_o_apertura": bool,
  "roles_mujeres": [
    {
      "nombre": str,
      "tipo_rol": "lider_o_experta" | "secundario" | "acompanante_o_victima",
      "tiene_cita_directa": bool
    }
  ]
}

REGLAS OBLIGATORIAS PARA CONTAR Y EXTRAER PERSONAS:
1. Solo cuenta e identifica INDIVIDUOS CONCRETOS (personas específicas nombradas o identificables individualmente por su cargo/rol, ej: "María Pérez", "la ministra", "el alcalde").
2. NO cuentes NUNCA ni incluyas cifras colectivas, estadísticas ni grupos indeterminados de gente (ejemplos a IGNORAR: "10 millones de personas", "miles de asistentes", "la población", "los ciudadanos", "un grupo de manifestantes").
3. Si el texto habla de cifras globales sin nombrar personas individuales, "total_personas_mencionadas" debe ser 0.

No añadas explicaciones ni texto fuera del JSON."""


def limpiar_respuesta_json(texto: str) -> dict:
    """Extrae y parsea el objeto JSON incluso si el modelo incluye bloques markdown."""
    texto = texto.strip()
    # Eliminar bloques markdown tipo ```json ... ```
    if texto.startswith("```"):
        lineas = texto.splitlines()
        if lineas[0].startswith("```"):
            lineas = lineas[1:]
        if lineas and lineas[-1].startswith("```"):
            lineas = lineas[:-1]
        texto = "\n".join(lineas).strip()

    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        # Intento de extracción por expresión regular si viene texto añadido
        coincidencia = re.search(r"\{.*\}", texto, re.DOTALL)
        if coincidencia:
            return json.loads(coincidencia.group(0))
        raise


def consultar_modelo(titular: str, texto_cuerpo: str) -> dict:
    """Llama a la API de OVHcloud con el modelo configurado."""
    prompt_usuario = f"TITULAR: {titular}\n\nTEXTO: {texto_cuerpo}"

    try:
        respuesta = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_usuario}
            ],
            temperature=0.0  # Temperatura 0 para asegurar respuestas deterministas y reproducibles
        )
        contenido = respuesta.choices[0].message.content
        return limpiar_respuesta_json(contenido)

    except Exception as e:
        print(f"  [ERROR] Fallo al procesar: {e}")
        return {
            "total_personas_mencionadas": 0,
            "mujeres_mencionadas": [],
            "hombres_mencionados": [],
            "mujer_en_titular_o_apertura": False,
            "roles_mujeres": [],
            "error": str(e)
        }


def ejecutar_evaluacion():
    # Validar que exista el archivo de entrada
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"No se encuentra el archivo de datos: {INPUT_PATH}")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        articulos = json.load(f)

    # Si estamos procesando los 250 artículos, se guarda en su carpeta final dedicada
    if "250" in INPUT_PATH:
        carpeta_destino = os.path.join(RESULTS_DIR, "analisis_final_250")
    else:
        # Si es el benchmark de 15, se guarda en la carpeta del modelo correspondiente
        nombre_carpeta_modelo = MODEL_NAME.replace("/", "_")
        carpeta_destino = os.path.join(RESULTS_DIR, nombre_carpeta_modelo)

    os.makedirs(carpeta_destino, exist_ok=True)

    print(f"=== INICIANDO EXTRACCIÓN CON MODELO: {MODEL_NAME} ===")
    print(f"Dataset de entrada: {INPUT_PATH} ({len(articulos)} artículos)")
    print(f"Guardando resultados en: {carpeta_destino}/\n")

    resultados_completos = []
    filas_csv = []

    for i, art in enumerate(articulos, start=1):
        art_id = art.get("id")
        titular = art.get("titular", art.get("title", ""))
        texto = art.get("text", art.get("content", art.get("cos", "")))

        print(f"[{i}/{len(articulos)}] Analizando artículo ID: {art_id}...")
        prediccion = consultar_modelo(titular, texto)

        # 1. Guardar estructura completa para el JSON
        registro = {
            "id": art_id,
            "titular": titular,
            "prediccion": prediccion
        }
        resultados_completos.append(registro)

        # 2. Aplanar datos para el archivo CSV
        filas_csv.append({
            "id": art_id,
            "titular": titular,
            "total_personas": prediccion.get("total_personas_mencionadas", 0),
            "num_mujeres": len(prediccion.get("mujeres_mencionadas", [])),
            "num_hombres": len(prediccion.get("hombres_mencionados", [])),
            "mujeres_mencionadas": "; ".join(prediccion.get("mujeres_mencionadas", [])),
            "hombres_mencionados": "; ".join(prediccion.get("hombres_mencionados", [])),
            "mujer_en_titular_o_apertura": prediccion.get("mujer_en_titular_o_apertura", False),
            "num_roles_mujeres": len(prediccion.get("roles_mujeres", [])),
            "roles_detalle": json.dumps(prediccion.get("roles_mujeres", []), ensure_ascii=False)
        })

    # Guardar predicciones en JSON
    ruta_json = os.path.join(carpeta_destino, "predicciones.json")
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resultados_completos, f, ensure_ascii=False, indent=2)

    # Guardar predicciones en CSV
    ruta_csv = os.path.join(carpeta_destino, "predicciones.csv")
    df = pd.DataFrame(filas_csv)
    df.to_csv(ruta_csv, index=False, encoding="utf-8")

    print("\n Extracción completada.")
    print(f"  -> Archivo JSON generado: {ruta_json}")
    print(f"  -> Archivo CSV generado:  {ruta_csv}")


if __name__ == "__main__":
    ejecutar_evaluacion()