# Biblioteca para preprocesamiento.
# `os` se usa para listar archivos y construir rutas en el sistema de ficheros
import os

# NLTK para recursos de procesamiento del lenguaje (stopwords, stemming)
import nltk

# pandas para leer los archivos CSV que contienen el corpus
import pandas as pd

# re para expresiones regulares (limpieza de caracteres)
import re

# stopwords y PorterStemmer para eliminar palabras vacías y aplicar stemming
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Funcion para cargar el corpus
# Inicio de la sección modificada (2026-05-27 19:52:46): funciones de carga y preprocesamiento
def cargar_corpus(path="corpus", archivos=None):
    """Carga todos los textos del directorio `path` desde archivos CSV.

    Args:
        path (str): carpeta donde están los CSV del corpus (por defecto 'corpus').
        archivos (list[str] | None): lista opcional de nombres de archivos CSV a cargar.

    Returns:
        list[str]: lista con todos los textos extraídos de la columna 'text' de
            cada CSV.
    """
    # Buscar solo los archivos solicitados; si no se especifican, usar todos los CSV.
    if archivos is None:
        archivos = sorted([f for f in os.listdir(path) if f.endswith(".csv")])
    else:
        archivos = sorted([f for f in os.listdir(path) if f.endswith(".csv")])

    archivos = sorted(archivos)
    # Mensaje informativo con el número de archivos encontrados
    print(f"Se encontraron {len(archivos)} archivos CSV en el directorio.")

    documentos = []  # lista donde acumularemos todos los textos del corpus

    # Iterar por cada archivo CSV encontrado
    for archivo in archivos:
        # Construir la ruta completa al archivo
        ruta_completa = os.path.join(path, archivo)

        # Leer el CSV con pandas (asumimos codificación utf-8)
        df = pd.read_csv(ruta_completa, encoding="utf-8")
        # Extraer la columna 'text', eliminar filas vacías y convertir a lista
        textos_archivo = df["text"].dropna().tolist()
        # Añadir los textos de este archivo a la lista global
        documentos.extend(textos_archivo)

    # Imprimir cuántos documentos totales se han cargado
    print(
        f"Carga completa. El corpus tiene un total de {len(documentos)} documentos individuales."
    )
    # Devolver la lista con todos los textos
    return documentos


# Función para preprocesamiento.
# Descargar stopwords de NLTK si no están presentes
nltk.download("stopwords")  # Necesario la primera vez (descarga BD de NLTK)


def preprocesar(texto):
    """Preprocesa un texto: normaliza, limpia, tokeniza y aplica stemming.

    Pasos aplicados:
      1. Normalización: pasar todo a minúsculas.
      2. Limpieza: eliminar caracteres no alfanuméricos (puntuación, símbolos).
      3. Tokenización: dividir en palabras por espacios.
      4. Remover stopwords (inglés por defecto) y aplicar stemming (Porter).

    Args:
        texto (str): texto original.

    Returns:
        list[str]: lista de tokens preprocesados.
    """

    # 1. Normalización → convertir a minúsculas para uniformizar
    texto = texto.lower()

    # 2. Limpiar caracteres especiales: dejar solo letras, números y espacios
    texto_limpio = re.sub(r"[^\w\s]", "", texto)

    # 3. Tokenización simple separando por espacios
    tokens = texto_limpio.split()

    # 4. Preparar el conjunto de stopwords y el stemmer
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()

    # 5. Remover stopwords y aplicar stemming a cada token que no sea stopword
    resultado = [ps.stem(word) for word in tokens if word not in stop_words]

    # Devolver la lista de tokens ya preprocesados
    return resultado


# ---------------------------------- Test de Funciones --------------------------
# Bloque de prueba desactivado por defecto. Si lo deseas, descomenta y
# ejecuta este archivo como script para probar la carga y preprocesamiento.
# if __name__ == "__main__":
#     corpus = cargar_corpus()
#
#     corpus_procesado = []
#     for doc in corpus:
#         tokens_doc = preprocesar(doc)
#         corpus_procesado.append(tokens_doc)
#
#     print(f"Se han preprocesado {len(corpus_procesado)} documentos.")
