# """Módulo de modelos de recuperación de información.
#
# Incluye dos estrategias de recuperación:
# - Jaccard sobre conjuntos de tokens (baseline simple, presencia/ausencia).
# - TF-IDF + similitud de coseno (representación ponderada por término).
#
# Este módulo asume que existe `preprocesamiento.preprocesar(text)` que devuelve
# una lista de tokens para un texto dado.
#"""

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
    union = len(set_q | set_d)         # elementos totales distintos

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

    Flujo resumido:
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
    # Convertir cada documento (lista de tokens) a un string separado por espacios
    corpus_strings = [" ".join(doc) for doc in corpus_procesado]

    # Crear el vectorizador TF-IDF con parámetros por defecto
    vectorizador = TfidfVectorizer()

    # Ajustar el vectorizador al corpus y transformar el corpus en matriz TF-IDF
    tfidf_matriz = vectorizador.fit_transform(corpus_strings)

    # Preprocesar la consulta y convertirla a string para vectorizarla
    query_tokens = preprocesamiento.preprocesar(query)
    query_string = " ".join(query_tokens)

    # Transformar la consulta a vector TF-IDF usando el vocabulario aprendido
    query_vector = vectorizador.transform([query_string])

    # Calcular la similitud coseno entre la consulta y cada documento
    similitudes = cosine_similarity(query_vector, tfidf_matriz).flatten()

    # Construir la lista de resultados (doc_id, score) filtrando ceros
    resultados = []
    for doc_id, score in enumerate(similitudes):
        if score > 0:
            resultados.append((doc_id, float(score)))

    # Ordenar y devolver los resultados por puntuación descendente
    resultados.sort(key=lambda x: x[1], reverse=True)

    return resultados


# ---------------------------------------------------------------------------
# Recuperación basada en BM25
# ---------------------------------------------------------------------------
def _calcular_idf_bm25(df, n_documentos):
    """Calcula el IDF con corrección de 0.5 usado en BM25."""
    valor = math.log((n_documentos - df + 0.5) / (df + 0.5))
    return max(0.0, valor)


def recuperar_bm25(query, corpus_procesado, k1=1.5, b=0.75, k3=None):
    """Recupera documentos usando el modelo Okapi BM25.

    Args:
        query (str): texto de la consulta.
        corpus_procesado (list[list[str]]): corpus tokenizado.
        k1 (float): controla la saturación de la frecuencia del término.
        b (float): controla la normalización por longitud del documento.
        k3 (float | None): si se indica, añade el factor de frecuencia en la consulta.

    Returns:
        list[tuple[int, float]]: lista de tuplas (doc_id, score) ordenada de mayor a menor.
    """
    query_tokens = preprocesamiento.preprocesar(query)

    if not corpus_procesado:
        return []

    n_documentos = len(corpus_procesado)
    longitudes = [len(doc) for doc in corpus_procesado]
    avgdl = sum(longitudes) / n_documentos

    # Frecuencia documental: cuántos documentos contienen cada término.
    df_por_termino = Counter()
    for doc_tokens in corpus_procesado:
        df_por_termino.update(set(doc_tokens))

    tf_query = Counter(query_tokens)
    terminos_query = tf_query.keys() if k3 is not None else set(query_tokens)

    resultados = []

    for doc_id, doc_tokens in enumerate(corpus_procesado):
        tf_documento = Counter(doc_tokens)
        dl = len(doc_tokens)
        normalizador = k1 * ((1 - b) + b * (dl / avgdl))
        score = 0.0

        for termino in terminos_query:
            tf_td = tf_documento.get(termino, 0)
            if tf_td == 0:
                continue

            df = df_por_termino.get(termino, 0)
            if df == 0:
                continue

            idf = _calcular_idf_bm25(df, n_documentos)
            tf_documento_factor = ((k1 + 1) * tf_td) / (normalizador + tf_td)

            if k3 is not None:
                tf_tq = tf_query[termino]
                tf_query_factor = ((k3 + 1) * tf_tq) / (k3 + tf_tq)
            else:
                tf_query_factor = 1.0

            score += idf * tf_documento_factor * tf_query_factor

        if score > 0:
            resultados.append((doc_id, float(score)))

    resultados.sort(key=lambda x: x[1], reverse=True)

    return resultados


# Interpretación de los scores BM25:
# - Los scores son valores reales >= 0; valores más altos indican mayor relevancia
#   relativa entre documentos para la misma consulta.
# - No son probabilidades: no suman 1 ni tienen un umbral universal.
# - Use los scores para ordenar y comparar (ranking). Por ejemplo, los
#   documentos con los 5-10 scores más altos suelen ser los más relevantes.
# - La magnitud absoluta depende del corpus (tamaño, vocabulario, longitudes):
#   comparar scores entre distintas colecciones no es directo.
# - Si necesita un valor interpretativo estable, normalice los scores o conviértalos
#   en rangos (e.g., min-max o escala logarítmica) antes de aplicar umbrales.
# - Para evaluación manual, inspeccione siempre los `top-k` (p. ej. top-10)
#   en lugar de basarse en un único umbral fijo.
