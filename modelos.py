import preprocesamiento
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# recuperación basada en similitud Jaccard utilizando vectores binarios
def calcular_jaccard(query_tokens, doc_tokens):
    """
    Calcula la similitud de Jaccard entre dos listas de tokens.
    Al usar set(), simulamos vectores binarios.
    """
    # Convertimos las listas a conjuntos
    set_q = set(query_tokens)
    set_d = set(doc_tokens)

    # Interseccion y union
    interseccion = len(set_q & set_d)
    union = len(set_q | set_d)

    # Evitamos la división por cero
    if union == 0:
        return 0.0

    # Retornamos la similitud jaccard entre 0 y 1
    return interseccion / union


def recuperar_jaccard(query, corpus_procesado):
    """
    Recibe una consulta de texto libre, la preprocesa y devuelve
    un ranking de los documentos más relevantes.
    """
    # 1. Preprocesar la consulta
    query_tokens = preprocesamiento.preprocesar(query)

    resultados = []

    # 2. Iterar sobre todos los documentos del corpus
    for doc_id, doc_tokens in enumerate(corpus_procesado):
        score = calcular_jaccard(query_tokens, doc_tokens)

        # Solo guardamos el documento si tiene al menos una palabra en común (score > 0)
        if score > 0:
            resultados.append((doc_id, score))

    # 3. Ordenar el ranking de mayor a menor similitud
    resultados.sort(key=lambda x: x[1], reverse=True)

    return resultados


# Recuperación basada en similitud de coseno utilizando TF-IDF
def recuperar_tfidf(query, corpus_procesado):
    """
    Recibe una consulta de texto libre, la preprocesa y devuelve
    un ranking de documentos ordenados por similitud coseno usando TF-IDF.
    """
    # 1. Adaptar nuestro corpus al formato que exige sklearn (lista de strings)
    # Unimos los tokens de cada documento con un espacio
    corpus_strings = [" ".join(doc) for doc in corpus_procesado]

    # 2. Inicializar el vectorizador TF-IDF
    vectorizador = TfidfVectorizer()

    # 3. Ajustar el modelo (aprender el vocabulario e IDF) y transformar el corpus en una matriz
    tfidf_matriz = vectorizador.fit_transform(corpus_strings)

    # 4. Preprocesar la consulta
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

    # 8. Ordenar de mayor a menor relevancia
    resultados.sort(key=lambda x: x[1], reverse=True)

    return resultados
