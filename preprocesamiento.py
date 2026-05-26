# Biblioteca para preprocesamiento.
import os

# import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Funcion para cargar el corpus
def cargar_corpus(path="corpus"):
    # Buscar unicamente archivos .csv
    archivos = [f for f in os.listdir(path) if f.endswith(".csv")]
    print(f"Se encontraron {len(archivos)} archivos CSV en el directorio.")

    documentos = []

    for archivo in archivos:
        ruta_completa = os.path.join(path, archivo)

        # Leer el CSV.
        df = pd.read_csv(ruta_completa, encoding="utf-8")
        # Extraemos solo los textos, eliminamos posibles nulos y los convertimos a lista
        textos_archivo = df["text"].dropna().tolist()
        documentos.extend(textos_archivo)

    print(
        f"Carga completa. El corpus tiene un total de {len(documentos)} documentos individuales."
    )
    return documentos


# Funcion para preprocesamiento.
# nltk.download("stopwords")  # Necesario la primera vez (descarga BD de NLTK)


def preprocesar(texto):

    # 1. Normalizacion -> conversion a minusculas
    texto = texto.lower()

    # 2. limpiar caracteres especiales
    texto_limpio = re.sub(r"[^\w\s]", "", texto)

    # 3. Tokenizacion
    tokens = texto_limpio.split()

    # 4. Stop Words y Stemmer
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()

    # 5. Remocion de stop words y Stemming
    # Solo procesamos la palabra si no es una stop word
    resultado = [ps.stem(word) for word in tokens if word not in stop_words]

    return resultado


# ---------------------------------- Test de Funciones --------------------------
# if __name__ == "__main__":
#     corpus = cargar_corpus()

#     corpus_procesado = []
#     for doc in corpus:
#         tokens_doc = preprocesar(doc)
#         corpus_procesado.append(tokens_doc)

#     print(f"Se han preprocesado {len(corpus_procesado)} documentos.")
