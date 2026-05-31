# modelo_semantico.py
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

# def construir_indice_faiss(corpus_original, nombre_modelo="all-MiniLM-L6-v2", modelo_existente=None):
#     """
#     Genera embeddings para el corpus y construye la base de datos vectorial con FAISS.
#     """
#     if modelo_existente is None:
#         print(f"Cargando el modelo preentrenado '{nombre_modelo}'...")
#         modelo = SentenceTransformer(nombre_modelo)
#     else:
#         modelo = modelo_existente
#         print("Reutilizando el modelo preentrenado ya cargado...")

#     print(f"Generando embeddings para {len(corpus_original)} documentos...")
#     print("(Esto puede tomar unos minutos dependiendo del procesador)")

#     # 1. Generar embeddings: encode() devuelve tensores con la representación semántica
#     embeddings = modelo.encode(corpus_original, show_progress_bar=True)

#     # FAISS exige que la matriz de vectores sea estrictamente de tipo float32
#     embeddings = np.array(embeddings).astype("float32")
#     dimension_vector = embeddings.shape[1]  # Para este modelo será de 384 dimensiones

#     print(f"Construyendo el índice FAISS (Dimensión: {dimension_vector})...")
#     # 2. Crear el índice basado en la distancia L2 (Euclidiana)
#     indice_faiss = faiss.IndexFlatL2(dimension_vector)

#     # 3. Almacenar los embeddings en la base de datos vectorial
#     indice_faiss.add(embeddings)

#     print(f"Índice FAISS completado con {indice_faiss.ntotal} vectores.")

#     return modelo, indice_faiss


def construir_indice_faiss(
    corpus_original, nombre_modelo="all-MiniLM-L6-v2", ruta_indice="indice_corpus.bin"
):
    """
    Genera embeddings para el corpus o los carga desde el disco si ya existen.
    """
    print(f"Cargando el modelo preentrenado '{nombre_modelo}'...")
    modelo = SentenceTransformer(nombre_modelo)

    # 1. Verificar si ya hicimos este trabajo antes
    if os.path.exists(ruta_indice):
        print(f"¡Índice encontrado en el disco! Cargando desde '{ruta_indice}'...")
        indice_faiss = faiss.read_index(ruta_indice)
        print(f"Índice FAISS cargado al instante con {indice_faiss.ntotal} vectores.")
        return modelo, indice_faiss

    # 2. Si no existe
    print(
        f"No se encontró índice guardado. Generando embeddings para {len(corpus_original)} documentos..."
    )
    print("(Esto tomará varios minutos.)")

    embeddings = modelo.encode(corpus_original, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    dimension_vector = embeddings.shape[1]

    print("Construyendo el índice FAISS...")
    indice_faiss = faiss.IndexFlatL2(dimension_vector)
    indice_faiss.add(embeddings)

    # 3. Guardar el trabajo para el futuro
    print(f"Guardando el índice en el disco como '{ruta_indice}'...")
    faiss.write_index(indice_faiss, ruta_indice)

    print(f"Proceso completado. Índice listo con {indice_faiss.ntotal} vectores.")

    return modelo, indice_faiss


def recuperar_semantico(query_texto, modelo, indice_faiss, top_k=5):
    """
    Recibe una consulta de texto libre, la vectoriza y recupera los
    documentos más similares usando búsqueda vectorial FAISS.
    """
    # 1. Generar el embedding para la consulta (SIN preprocesar)
    # Lo pasamos como lista [query_texto] porque encode espera un listado
    query_vector = modelo.encode([query_texto])
    query_vector = np.array(query_vector).astype("float32")

    # 2. Búsqueda Vectorial en FAISS
    # Retorna D (Distancias) e I (Índices / doc_ids)
    distancias, indices = indice_faiss.search(query_vector, top_k)

    # 3. Formatear resultados para que coincida con la estructura de modelos anteriores
    resultados = []
    for i in range(top_k):
        doc_id = int(indices[0][i])
        distancia_l2 = float(distancias[0][i])

        # IMPORTANTE: En la distancia L2, un valor más cercano a 0 es MEJOR.
        # Para que el ranking sea coherente con BM25/TF-IDF (donde mayor es mejor),
        # invertimos el score matemáticamente.
        score_invertido = 1 / (1 + distancia_l2)

        resultados.append((doc_id, score_invertido))

    return resultados
