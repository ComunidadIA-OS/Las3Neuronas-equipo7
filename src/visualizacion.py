import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# 1. Cargar variables de entorno
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")

# Configuración estética general
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def generar_grafico_benchmark():
    """Genera la gráfica comparativa global de los 6 modelos a partir de comparativa_modelos.csv."""
    ruta_comparativa = os.path.join(RESULTS_DIR, "comparativa_modelos.csv")
    if not os.path.exists(ruta_comparativa):
        print(f"  [AVISO] No se encontró el archivo comparativo: {ruta_comparativa}")
        print("         Ejecuta primero 'calcular_metricas.py' para registrar las evaluaciones.")
        return

    df = pd.read_csv(ruta_comparativa)
    if df.empty:
        return

    columnas_metricas = ["precision_mujeres", "recall_mujeres", "f1_mujeres"]
    df_melted = df.melt(
        id_vars=["modelo"],
        value_vars=columnas_metricas,
        var_name="Métrica",
        value_name="Puntuación"
    )

    df_melted["Métrica"] = df_melted["Métrica"].map({
        "precision_mujeres": "Precisión",
        "recall_mujeres": "Recall",
        "f1_mujeres": "F1-Score"
    })

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=df_melted,
        x="modelo",
        y="Puntuación",
        hue="Métrica",
        palette="viridis"
    )
    plt.title("Comparativa de Modelos Abiertos (Benchmark vs Golden Standard)", fontsize=14, weight="bold")
    plt.xlabel("Modelo Evaluado", fontsize=12)
    plt.ylabel("Puntuación (0.0 - 1.0)", fontsize=12)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15, ha="right")
    plt.legend(title="Métrica", loc="lower right")

    # Anotar valores numéricos encima de cada barra
    for p in ax.patches:
        altura = p.get_height()
        if altura > 0:
            ax.annotate(f"{altura:.2f}",
                        (p.get_x() + p.get_width() / 2., altura),
                        ha='center', va='bottom', fontsize=8, xytext=(0, 2),
                        textcoords='offset points')

    # Se guarda en la raíz de results para no sobreescribir la carpeta de ningún modelo
    ruta_salida = os.path.join(RESULTS_DIR, "benchmark_global_modelos.png")
    plt.savefig(ruta_salida, dpi=300)
    plt.close()
    print(f"  -> [OK] Gráfica comparativa global guardada en: {ruta_salida}")


def generar_graficos_modelo_actual():
    """Genera las métricas de presencia y protagonismo dentro de la subcarpeta del modelo actual."""
    if not MODEL_NAME:
        print("  [ERROR] La variable MODEL_NAME no está definida en el archivo .env.")
        return

    nombre_carpeta_modelo = MODEL_NAME.replace("/", "_")
    carpeta_modelo = os.path.join(RESULTS_DIR, nombre_carpeta_modelo)
    ruta_predicciones = os.path.join(carpeta_modelo, "predicciones.json")

    if not os.path.exists(ruta_predicciones):
        print(f"  [AVISO] No se encontró el archivo de predicciones en: {ruta_predicciones}")
        print("         Ejecuta primero 'evaluacion_modelos.py' para este modelo.")
        return

    with open(ruta_predicciones, "r", encoding="utf-8") as f:
        datos = json.load(f)

    total_mujeres = 0
    total_hombres = 0
    roles_conteo = {}
    citas_directas_conteo = {"Con cita directa": 0, "Sin cita directa": 0}

    for item in datos:
        pred = item.get("prediccio", item.get("prediccion", {}))
        m = len(pred.get("mujeres_mencionadas", []))
        h = len(pred.get("hombres_mencionados", []))
        total_mujeres += m
        total_hombres += h

        for rol in pred.get("roles_mujeres", []):
            tipo = rol.get("tipo_rol", "no_definido")
            roles_conteo[tipo] = roles_conteo.get(tipo, 0) + 1

            if rol.get("tiene_cita_directa", False):
                citas_directas_conteo["Con cita directa"] += 1
            else:
                citas_directas_conteo["Sin cita directa"] += 1

    # 1. Gráfica de Presencia Global (Mujeres vs Hombres)
    plt.figure(figsize=(6, 5))
    categorias_genero = ["Mujeres", "Hombres"]
    valores_genero = [total_mujeres, total_hombres]
    colores = ["#e7298a", "#1b9e77"]

    bars = plt.bar(categorias_genero, valores_genero, color=colores, width=0.45)
    plt.title(f"Presencia Total por Género\n({MODEL_NAME})", fontsize=13, weight="bold")
    plt.ylabel("Número total de menciones individuales", fontsize=11)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f"{int(yval)}", ha="center", va="bottom", weight="bold")

    ruta_salida_presencia = os.path.join(carpeta_modelo, "presencia_genero.png")
    plt.savefig(ruta_salida_presencia, dpi=300)
    plt.close()
    print(f"  -> [OK] Gráfica de presencia guardada en: {ruta_salida_presencia}")

    # 2. Gráfica de Tipología de Roles y Citas Directas (Protagonismo)
    if roles_conteo:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Subgráfico A: Distribución por tipo de rol
        df_roles = pd.DataFrame(list(roles_conteo.items()), columns=["Tipo de Rol", "Conteo"])
        sns.barplot(data=df_roles, x="Tipo de Rol", y="Conteo", ax=axes[0], palette="mako")
        axes[0].set_title("Distribución de Roles Femeninos", weight="bold")
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=15, ha="right")

        for p in axes[0].patches:
            val = p.get_height()
            if val > 0:
                axes[0].annotate(f"{int(val)}", (p.get_x() + p.get_width() / 2., val),
                                 ha='center', va='bottom', xytext=(0, 2), textcoords='offset points')

        # Subgráfico B: Citas directas (Voz propia)
        labels_citas = list(citas_directas_conteo.keys())
        sizes_citas = list(citas_directas_conteo.values())
        if sum(sizes_citas) > 0:
            axes[1].pie(sizes_citas, labels=labels_citas, autopct='%1.1f%%', colors=["#4daf4a", "#e41a1c"], startangle=90)
            axes[1].set_title("Mujeres con Citas Directas (Voz Propia)", weight="bold")

        ruta_salida_roles = os.path.join(carpeta_modelo, "roles_mujeres.png")
        plt.savefig(ruta_salida_roles, dpi=300)
        plt.close()
        print(f"  -> [OK] Gráfica de roles y citas guardada en: {ruta_salida_roles}")


def ejecutar_visualizacion():
    print(f"\n=== GENERANDO VISUALIZACIONES PARA: {MODEL_NAME} ===")
    generar_graficos_modelo_actual()
    generar_grafico_benchmark()
    print("=== PROCESO DE VISUALIZACIÓN COMPLETADO ===\n")


if __name__ == "__main__":
    ejecutar_visualizacion()