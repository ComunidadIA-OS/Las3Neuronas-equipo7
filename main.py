import os
import sys
from dotenv import load_dotenv

# 1. Cargar y comprobar variables de configuración desde el .env
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
API_KEY = os.getenv("OVH_API_KEY")
INPUT_DATA_PATH = os.getenv("INPUT_DATA_PATH", "data/dataset_15.json")

# 2. Importar los módulos que están dentro de la carpeta src/
try:
    from src.evaluacion_modelos import ejecutar_evaluacion
    from src.calcular_metricas import ejecutar_calculo
    from src.visualizacion import ejecutar_visualizacion
except ImportError as error_importacion:
    print(f"\n[ERROR DE IMPORTACIÓN]: {error_importacion}")
    print("Asegúrate de ejecutar el comando desde la raíz del proyecto (reto-ia-mujeres-equipo7).\n")
    sys.exit(1)


def main():
    print("\n" + "=" * 65)
    print("   RETO 'MUJERES POR LA IA ABIERTA' - PIPELINE AUTOMATIZADO")
    print("=" * 65)

    # Validaciones obligatorias antes de ejecutar
    if not API_KEY:
        print("[ERROR] No se ha detectado la variable OVH_API_KEY en el archivo .env.")
        print("Revisa tu archivo .env e introduce tu token de OVHcloud.\n")
        sys.exit(1)

    if not MODEL_NAME:
        print("[ERROR] La variable MODEL_NAME no está configurada en el archivo .env.\n")
        sys.exit(1)

    print(f"• Modelo activo:       {MODEL_NAME}")
    print(f"• Archivo de entrada:  {INPUT_DATA_PATH}")
    print("=" * 65)

    # PASO 1: Inferencia con el modelo LLM configurado
    print("\n>>> [PASO 1/3] Ejecutando extracción e inferencia con el modelo...")
    try:
        ejecutar_evaluacion()
    except Exception as error_paso1:
        print(f"[ERROR en evaluacion_modelos]: {error_paso1}")
        sys.exit(1)

    # PASO 2: Evaluación frente al Golden Standard
    print("\n>>> [PASO 2/3] Comparando con el Golden Standard y calculando métricas...")
    try:
        ejecutar_calculo()
    except Exception as error_paso2:
        print(f"[ERROR en calcular_metricas]: {error_paso2}")
        sys.exit(1)

    # PASO 3: Generación de visualizaciones y gráficas
    print("\n>>> [PASO 3/3] Generando gráficas de presencia, roles y benchmark global...")
    try:
        ejecutar_visualizacion()
    except Exception as error_paso3:
        print(f"[ERROR en visualizacion]: {error_paso3}")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("   PROCESO COMPLETADO CON ÉXITO")
    print(f"   Revisa la carpeta 'results/{MODEL_NAME.replace('/', '_')}' para ver los datos.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()