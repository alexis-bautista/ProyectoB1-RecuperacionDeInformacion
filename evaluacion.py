"""Utilidades de evaluación para los modelos de recuperación de información."""

import ast
import re
import pandas as pd

# --- 1. UTILIDADES DE DATASET Y QRELS ---


def _parsear_temas(valor):
    """Convierte el contenido de la columna topics en una lista de temas limpios."""
    if pd.isna(valor):
        return []
    texto = str(valor).strip()
    if not texto:
        return []

    try:
        parsed = ast.literal_eval(texto)
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass

    temas = re.findall(r"'([^']+)'", texto)
    if temas:
        return [tema.strip() for tema in temas if tema.strip()]
    return [texto]


def construir_qrels_desde_dataframe(df_corpus, consultas):
    """
    Construye el diccionario de qrels (documentos relevantes)
    buscando la consulta dentro de los temas (topics) del DataFrame.
    """
    qrels = {}
    temas_por_doc = df_corpus["topics"].apply(_parsear_temas)

    for consulta in consultas:
        consulta_normalizada = str(consulta).strip().lower()
        relevantes = set()

        # doc_id aquí debe coincidir exactamente con el índice del corpus_procesado
        for doc_id, temas in enumerate(temas_por_doc):
            temas_normalizados = {tema.lower() for tema in temas}
            if consulta_normalizada in temas_normalizados:
                relevantes.add(doc_id)

        qrels[consulta] = relevantes
    return qrels


# --- 2. MÉTRICAS MATEMÁTICAS ---


def precision_at_k(ranking, relevantes, k=10):
    if k <= 0 or not ranking[:k]:
        return 0.0
    recuperados = ranking[:k]
    aciertos = sum(1 for doc_id, _ in recuperados if doc_id in relevantes)
    return aciertos / len(recuperados)


def recall_at_k(ranking, relevantes, k=10):
    if not relevantes:
        return 0.0
    recuperados = ranking[:k]
    aciertos = sum(1 for doc_id, _ in recuperados if doc_id in relevantes)
    return aciertos / len(relevantes)


def average_precision(ranking, relevantes, k=None):
    if not relevantes:
        return 0.0
    if k is None:
        k = len(ranking)

    hits = 0
    suma_precisiones = 0.0
    for i, (doc_id, _) in enumerate(ranking[:k], start=1):
        if doc_id in relevantes:
            hits += 1
            suma_precisiones += hits / i
    return suma_precisiones / len(relevantes)


# --- 3. EVALUACIÓN Y COMPARACIÓN ---


def evaluar_modelo(resultados_por_consulta, qrels, k=10):
    filas = []
    ap_por_consulta = []

    for consulta, ranking in resultados_por_consulta.items():
        relevantes = qrels.get(consulta, set())
        ap = average_precision(ranking, relevantes, k=k)
        ap_por_consulta.append(ap)

        filas.append(
            {
                "consulta": consulta,
                f"precision@{k}": precision_at_k(ranking, relevantes, k=k),
                f"recall@{k}": recall_at_k(ranking, relevantes, k=k),
                f"ap@{k}": ap,
                "relevantes_totales": len(relevantes),
            }
        )

    map_score = sum(ap_por_consulta) / len(ap_por_consulta) if ap_por_consulta else 0.0
    return pd.DataFrame(filas), map_score


def comparar_modelos(resultados_por_modelo, qrels, k=10):
    filas = []
    for nombre_modelo, resultados_por_consulta in resultados_por_modelo.items():
        resumen_df, map_score = evaluar_modelo(resultados_por_consulta, qrels, k=k)
        filas.append(
            {
                "Modelo": nombre_modelo.upper(),
                f"MAP@{k}": map_score,
                f"Precision Media@{k}": (
                    resumen_df[f"precision@{k}"].mean() if not resumen_df.empty else 0.0
                ),
                f"Recall Medio@{k}": (
                    resumen_df[f"recall@{k}"].mean() if not resumen_df.empty else 0.0
                ),
            }
        )
    return (
        pd.DataFrame(filas)
        .sort_values(by=f"MAP@{k}", ascending=False)
        .reset_index(drop=True)
    )
