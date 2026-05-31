import os
import pandas as pd
import preprocesamiento
import indice_invertido
import modelos
import modelo_semantico


def cargar_datos_iniciales():
    """Carga y prepara todos los datos en memoria al arrancar el programa."""
    print("=" * 60)
    print("INICIALIZANDO MOTOR DE BÚSQUEDA...")
    print("=" * 60)

    # 1. Cargar Corpus
    print("[1/4] Cargando documentos CSV...")
    archivos = sorted([f for f in os.listdir("corpus") if f.endswith(".csv")])
    df_corpus = pd.concat(
        [pd.read_csv(f"corpus/{f}", encoding="utf-8") for f in archivos],
        ignore_index=True,
    )
    df_corpus = df_corpus.dropna(subset=["text"]).reset_index(drop=True)
    corpus_original = df_corpus["text"].tolist()

    # 2. Preprocesamiento
    print("[2/4] Preprocesando textos (Limpieza y Stemming)...")
    corpus_procesado = [preprocesamiento.preprocesar(doc) for doc in corpus_original]

    # 3. Índice Invertido
    print("[3/4] Construyendo Índice Invertido...")
    indice = indice_invertido.crear_indice_invertido(corpus_procesado)

    # 4. Modelo Semántico (Carga instantánea desde el disco gracias al .bin)
    print("[4/4] Cargando Base de Datos Vectorial (FAISS)...")
    modelo_transformer, base_vectorial_faiss = modelo_semantico.construir_indice_faiss(
        corpus_original
    )

    print("\n¡Sistema cargado exitosamente!\n")
    return df_corpus, corpus_procesado, indice, modelo_transformer, base_vectorial_faiss


def imprimir_resultados(ranking, df_corpus, etiqueta_score, top_k=5):
    """Imprime el ranking de manera legible con fragmentos de texto."""
    if not ranking:
        print("No se encontraron documentos relevantes.")
        return

    for i, (doc_id, score) in enumerate(ranking[:top_k], 1):
        # Extraemos un pequeño snippet de 150 caracteres para mostrar en consola
        texto_real = str(df_corpus["text"].iloc[doc_id])[:150].replace("\n", " ")
        print(f"{i}. [Doc ID: {doc_id}] | {etiqueta_score}: {score:.4f}")
        print(f"   Fragmento: {texto_real}...\n")


def main():
    # Arrancamos los motores antes de mostrar el menú
    df_corpus, corpus_procesado, indice, modelo_transformer, base_vectorial_faiss = (
        cargar_datos_iniciales()
    )

    while True:
        print("=" * 60)
        print("MENÚ PRINCIPAL - SISTEMA DE RECUPERACIÓN")
        print("=" * 60)
        print("1. Búsqueda Binaria (Jaccard)")
        print("2. Búsqueda Vectorial (TF-IDF)")
        print("3. Búsqueda Probabilística (BM25)")
        print("4. Búsqueda Semántica (Transformers + FAISS)")
        print("5. Salir del sistema")
        print("=" * 60)

        opcion = input("Seleccione un modelo (1-5): ")

        if opcion == "5":
            print("\nApagando el motor de búsqueda. ¡Hasta luego!")
            break

        if opcion not in ["1", "2", "3", "4"]:
            print("\n❌ Opción inválida. Intente de nuevo.")
            continue

        query = input("\nIngrese su consulta de texto libre: ")
        print(f"\nGenerando ranking para: '{query}'...\n")

        # Redireccionamiento al módulo correspondiente
        if opcion == "1":
            ranking = modelos.recuperar_jaccard(query, corpus_procesado)
            imprimir_resultados(ranking, df_corpus, "Similitud")

        elif opcion == "2":
            ranking = modelos.recuperar_tfidf(query, corpus_procesado)
            imprimir_resultados(ranking, df_corpus, "Similitud Coseno")

        elif opcion == "3":
            ranking = modelos.recuperar_bm25(query, corpus_procesado, indice)
            imprimir_resultados(ranking, df_corpus, "Score BM25")

        elif opcion == "4":
            ranking = modelo_semantico.recuperar_semantico(
                query, modelo_transformer, base_vectorial_faiss
            )
            imprimir_resultados(ranking, df_corpus, "Similitud Semántica")

        input("Presione ENTER para realizar otra búsqueda...")


# Esta condición asegura que el menú solo corra si ejecutas el archivo directamente
if __name__ == "__main__":
    main()
