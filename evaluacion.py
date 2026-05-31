"""Utilidades de evaluacion para los modelos de recuperacion de informacion."""

from __future__ import annotations

import ast
import re
from typing import Callable, Iterable

import pandas as pd


def _parsear_temas(valor):
    """Convierte el contenido de la columna topics en una lista de temas."""
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


def cargar_corpus_dataframe(path="corpus", archivos=None):
    """Carga los CSV del corpus en un unico DataFrame."""
    if archivos is None:
        from os import listdir

        archivos = [f for f in listdir(path) if f.endswith(".csv")]
    else:
        archivos = [f for f in archivos if f.endswith(".csv")]

    frames = []
    for archivo in sorted(archivos):
        ruta = f"{path}/{archivo}"
        df = pd.read_csv(ruta, encoding="utf-8")
        df = df.copy()
        df["source_file"] = archivo
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def construir_qrels_desde_temas(df, consultas):
    """Construye qrels usando la columna topics como verdad de relevancia."""
    qrels = {}
    temas_por_doc = df["topics"].apply(_parsear_temas)

    for consulta in consultas:
        consulta_normalizada = str(consulta).strip().lower()
        relevantes = set()

        for doc_id, temas in enumerate(temas_por_doc):
            temas_normalizados = {tema.lower() for tema in temas}
            if consulta_normalizada in temas_normalizados:
                relevantes.add(doc_id)

        qrels[consulta] = relevantes

    return qrels


def precision_at_k(ranking, relevantes, k=10):
    """Calcula precision@k para una lista de resultados rankeados."""
    if k <= 0:
        return 0.0

    recuperados = ranking[:k]
    if not recuperados:
        return 0.0

    aciertos = sum(1 for doc_id, _ in recuperados if doc_id in relevantes)
    return aciertos / len(recuperados)


def recall_at_k(ranking, relevantes, k=10):
    """Calcula recall@k para una lista de resultados rankeados."""
    if not relevantes:
        return 0.0

    recuperados = ranking[:k]
    aciertos = sum(1 for doc_id, _ in recuperados if doc_id in relevantes)
    return aciertos / len(relevantes)


def average_precision(ranking, relevantes, k=None):
    """Calcula Average Precision para una consulta."""
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


def evaluar_modelo(resultados_por_consulta, qrels, k=10):
    """Evalua un modelo a nivel de consulta y devuelve resumen y MAP."""
    filas = []
    ap_por_consulta = []

    for consulta, ranking in resultados_por_consulta.items():
        relevantes = qrels.get(consulta, set())
        precision = precision_at_k(ranking, relevantes, k=k)
        recall = recall_at_k(ranking, relevantes, k=k)
        ap = average_precision(ranking, relevantes, k=k)

        filas.append(
            {
                "consulta": consulta,
                f"precision@{k}": precision,
                f"recall@{k}": recall,
                f"ap@{k}": ap,
                "relevantes": len(relevantes),
            }
        )
        ap_por_consulta.append(ap)

    resumen = pd.DataFrame(filas)
    map_score = sum(ap_por_consulta) / len(ap_por_consulta) if ap_por_consulta else 0.0
    return resumen, map_score


def comparar_modelos(resultados_por_modelo, qrels, k=10):
    """Compara varios modelos y devuelve una tabla con MAP por modelo."""
    filas = []

    for nombre_modelo, resultados_por_consulta in resultados_por_modelo.items():
        resumen, map_score = evaluar_modelo(resultados_por_consulta, qrels, k=k)
        filas.append(
            {
                "modelo": nombre_modelo,
                f"map@{k}": map_score,
                f"precision_promedio@{k}": resumen[f"precision@{k}"].mean() if not resumen.empty else 0.0,
                f"recall_promedio@{k}": resumen[f"recall@{k}"].mean() if not resumen.empty else 0.0,
            }
        )

    return pd.DataFrame(filas).sort_values(by=f"map@{k}", ascending=False).reset_index(drop=True)


def ejecutar_evaluacion_completa(
    modelo_semantico_existente,
    corpus_path="corpus",
    archivo_test="ModApte_test.csv",
    consultas_eval=None,
    k=10,
    top_k_semantico=20,
):
    """Ejecuta la evaluacion completa de los cuatro modelos sobre un archivo de prueba.

    Returns:
        tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
            comparacion_modelos, resultados_por_modelo, qrels_eval, resumen_por_modelo
    """
    import indice_invertido
    import modelos
    import preprocesamiento
    import importlib
    import modelo_semantico as modelo_semantico_mod

    modelo_semantico_mod = importlib.reload(modelo_semantico_mod)

    if consultas_eval is None:
        consultas_eval = ["earn", "acq", "crude", "trade", "money-fx", "grain", "cocoa"]

    corpus_eval_df = cargar_corpus_dataframe(path=corpus_path, archivos=[archivo_test])
    corpus_eval = corpus_eval_df["text"].dropna().tolist()
    corpus_eval_procesado = [preprocesamiento.preprocesar(doc) for doc in corpus_eval]
    indice_eval = indice_invertido.crear_indice_invertido(corpus_eval_procesado)
    qrels_eval = construir_qrels_desde_temas(corpus_eval_df, consultas_eval)

    modelo_eval_semantico, base_eval_faiss = modelo_semantico_mod.construir_indice_faiss(
        corpus_eval,
        modelo_existente=modelo_semantico_existente,
    )

    resultados_por_modelo = {"jaccard": {}, "tfidf": {}, "bm25": {}, "semantico": {}}

    for consulta in consultas_eval:
        resultados_por_modelo["jaccard"][consulta] = modelos.recuperar_jaccard(consulta, corpus_eval_procesado)
        resultados_por_modelo["tfidf"][consulta] = modelos.recuperar_tfidf(consulta, corpus_eval_procesado)
        resultados_por_modelo["bm25"][consulta] = modelos.recuperar_bm25(consulta, corpus_eval_procesado, indice_eval)
        resultados_por_modelo["semantico"][consulta] = modelo_semantico_mod.recuperar_semantico(
            consulta,
            modelo_eval_semantico,
            base_eval_faiss,
            top_k=top_k_semantico,
        )

    resumen_por_modelo = {}
    for nombre_modelo, resultados_por_consulta in resultados_por_modelo.items():
        resumen, _ = evaluar_modelo(resultados_por_consulta, qrels_eval, k=k)
        resumen_por_modelo[nombre_modelo] = resumen

    comparacion_modelos = comparar_modelos(resultados_por_modelo, qrels_eval, k=k)
    return comparacion_modelos, resultados_por_modelo, qrels_eval, resumen_por_modelo