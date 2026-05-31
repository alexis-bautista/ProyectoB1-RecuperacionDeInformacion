# """Módulo de modelos de recuperación de información.
#
# Incluye dos estrategias de recuperación:
# - Jaccard sobre conjuntos de tokens (baseline simple, presencia/ausencia).
# - TF-IDF + similitud de coseno (representación ponderada por término).
#
# Este módulo asume que existe `preprocesamiento.preprocesar(text)` que devuelve
# una lista de tokens para un texto dado.
# """

# Importa el módulo local que contiene la función `preprocesar`.
import preprocesamiento

# Contador de frecuencias para BM25.
from collections import Counter

# Funciones matemáticas para logaritmos y otras operaciones.
import math

# Importa el vectorizador TF-IDF de scikit-learn.
from sklearn.feature_extraction.text import TfidfVectorizer

# Importa la función para calcular similitud coseno entre vectores.
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Recuperación basada en similitud Jaccard (vectores binarios)
# ---------------------------------------------------------------------------
def calcular_jaccard(query_tokens, doc_tokens):
    """Calcula la similitud de Jaccard entre dos listas de tokens.

    La similitud de Jaccard se define como |A ∩ B| / |A ∪ B|.
    Aquí convertimos las listas a conjuntos para considerar solo presencia/ausencia
    (equivalente a vectores binarios).

    Args:
        query_tokens (list[str]): tokens de la consulta.
        doc_tokens (list[str]): tokens del documento.

    Returns:
        float: valor entre 0.0 y 1.0 con la similitud Jaccard.
    """
    # Crear un conjunto con los tokens de la consulta (quita duplicados)
    set_q = set(query_tokens)
    # Crear un conjunto con los tokens del documento (quita duplicados)
    set_d = set(doc_tokens)

    # Calcular la intersección: elementos que aparecen en ambos conjuntos
    interseccion = len(set_q & set_d)  # elementos comunes
    # Calcular la unión: elementos que aparecen en al menos uno
    union = len(set_q | set_d)  # elementos totales distintos

    # Si la unión es 0 (ambos conjuntos vacíos) evitamos división por cero
    if union == 0:
        # No hay términos para comparar → similitud 0.0
        return 0.0

    # Devolver la razón entre intersección y unión (valor entre 0 y 1)
    return interseccion / union


def recuperar_jaccard(query, corpus_procesado):
    """Recupera documentos ordenados por similitud Jaccard.

    Args:
        query (str): texto de la consulta.
        corpus_procesado (list[list[str]]): lista de documentos, cada uno es
            una lista de tokens (preprocesados).

    Returns:
        list[tuple[int, float]]: lista de tuplas (doc_id, score) ordenada
            por score descendente. Solo se incluyen documentos con score>0.
    """
    # Llamar a la función de preprocesamiento para obtener tokens de la consulta
    query_tokens = preprocesamiento.preprocesar(query)

    # Inicializar la lista donde guardaremos (doc_id, puntuación)
    resultados = []

    # Iterar sobre cada documento preprocesado junto con su índice
    for doc_id, doc_tokens in enumerate(corpus_procesado):
        # Calcular la similitud Jaccard entre consulta y documento
        score = calcular_jaccard(query_tokens, doc_tokens)

        # Si hay al menos una coincidencia, añadimos el resultado
        if score > 0:
            resultados.append((doc_id, score))

    # Ordenar los resultados por puntuación (de mayor a menor)
    resultados.sort(key=lambda x: x[1], reverse=True)

    # Devolver la lista ordenada de tuplas (índice_documento, puntuación)
    return resultados


# ---------------------------------------------------------------------------
# Recuperación basada en TF-IDF + similitud de coseno
# ---------------------------------------------------------------------------
def recuperar_tfidf(query, corpus_procesado):
    """Recupera documentos usando TF-IDF y similitud de coseno.

    Flujo :
    1. Convertir cada documento tokenizado a un string (" ".join(tokens)).
    2. Crear un `TfidfVectorizer` y hacer `fit_transform` sobre el corpus
       para obtener la matriz TF-IDF de los documentos.
    3. Preprocesar la consulta y vectorizarla con `transform` (no re-ajustar
       el vectorizador, porque queremos usar el mismo vocabulario).
    4. Calcular la similitud coseno entre la consulta y todos los documentos.
    5. Devolver el ranking de `(doc_id, score)` ordenado.

    Args:
        query (str): texto de la consulta.
        corpus_procesado (list[list[str]]): lista de documentos tokenizados.

    Returns:
        list[tuple[int, float]]: lista de tuplas (doc_id, score) ordenada
            por score descendente. Solo se incluyen documentos con score>0.
    """
    # 1. Adaptar nuestro corpus al formato que exige sklearn (lista de strings)
    # Unimos los tokens de cada documento con un espaci
    corpus_strings = [" ".join(doc) for doc in corpus_procesado]

    # 2. Inicializar el vectorizador TF-IDF
    vectorizador = TfidfVectorizer()

    # 3. Ajustar el modelo (aprender el vocabulario e IDF) y transformar el corpus en una matriz
    tfidf_matriz = vectorizador.fit_transform(corpus_strings)

    # 4. Preprocesar la consulta y convertirla a string para vectorizarla
    query_tokens = preprocesamiento.preprocesar(query)
    query_string = " ".join(query_tokens)

    # 5. Vectorizar la consulta
    # Usamos transform() en lugar de fit_transform() para usar el vocabulario ya aprendido
    query_vector = vectorizador.transform([query_string])

    # 6. Calcular la similitud del coseno entre la consulta y todos los documentos
    # flatten() convierte la matriz de resultados en un arreglo unidimensional simple
    similitudes = cosine_similarity(query_vector, tfidf_matriz).flatten()

    # 7. Crear el ranking final estructurado como (doc_id, score)
    resultados = []
    for doc_id, score in enumerate(similitudes):
        # Filtramos para no mostrar documentos con 0% de similitud
        if score > 0:
            resultados.append((doc_id, float(score)))

    # Ordenar y devolver los resultados por puntuación descendente
    resultados.sort(key=lambda x: x[1], reverse=True)

    return resultados


# ---------------------------------------------------------------------------
# Recuperación basada en BM25
# ---------------------------------------------------------------------------
def calcular_idf_bm25(df, n_documentos):
    """Calcula el IDF con la suavización estándar de BM25."""
    valor = math.log((n_documentos - df + 0.5) / (df + 0.5))
    return max(0.0, valor)


def recuperar_bm25(query_texto, corpus_procesado, indice_invertido, k1=1.5, b=0.75):
    """
    Recupera documentos usando el modelo BM25.
    """
    # 1. Preprocesar la consulta
    query_tokens = preprocesamiento.preprocesar(query_texto)

    # Si la consulta está vacía tras el preprocesamiento, no hay resultados
    if not query_tokens:
        return []

    # 2. Precalcular estadísticas globales del corpus
    n_documentos = len(corpus_procesado)
    avgdl = sum(len(doc) for doc in corpus_procesado) / n_documentos

    resultados = []

    # 3. Iterar sobre los documentos para calcular su score
    for doc_id, doc_tokens in enumerate(corpus_procesado):
        score = 0.0
        dl = len(doc_tokens)  # Longitud del documento actual

        # Evaluamos solo los términos únicos de la consulta
        for termino in set(query_tokens):

            # Si el término existe en nuestro índice y aparece en este documento
            if termino in indice_invertido and doc_id in indice_invertido[termino]:
                # Obtenemos TF y DF del indice invertido
                tf = indice_invertido[termino][doc_id]
                df = len(indice_invertido[termino])

                # Calculamos IDF
                idf = calcular_idf_bm25(df, n_documentos)

                # Calculamos el peso BM25 para este término
                numerador = tf * (k1 + 1)
                denominador = tf + k1 * (1 - b + b * (dl / avgdl))

                # Sumamos al score total del documento
                score += idf * (numerador / denominador)

        # Si el documento tiene alguna coincidencia, lo guardamos
        if score > 0:
            resultados.append((doc_id, float(score)))

    # 4. Ordenar el ranking de mayor a menor relevancia
    resultados.sort(key=lambda x: x[1], reverse=True)

    return resultados
