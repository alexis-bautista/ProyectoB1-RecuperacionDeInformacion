# indice_invertido.py


def crear_indice_invertido(corpus_procesado):
    """
    Construye un índice invertido a partir de un corpus preprocesado.

    Args:
        corpus_procesado: Lista donde cada elemento es un documento representado como una lista de tokens.

    Returns:
        dict: Índice invertido con la estructura {termino: {doc_id: frecuencia}}
    """
    indice = {}

    # enumerate nos da tanto el ID del documento (su posición en la lista) como sus tokens
    for doc_id, tokens in enumerate(corpus_procesado):
        for token in tokens:
            # Si el término es completamente nuevo, creamos su entrada en el índice principal
            if token not in indice:
                indice[token] = {}

            # Si el término existe pero es la primera vez que lo vemos en este documento, inicializamos su contador
            if doc_id not in indice[token]:
                indice[token][doc_id] = 0

            # Incrementamos la frecuencia del término para este documento en particular
            indice[token][doc_id] += 1

    print("El indice a sido creado correctamente")

    return indice
