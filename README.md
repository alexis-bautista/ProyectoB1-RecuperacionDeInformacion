# ProyectoB1-Recuperacion-de-Informacion

# Sistema de Recuperación de Información

## Descripción del Proyecto

Este proyecto implementa y evalúa un motor de búsqueda modular que compara modelos de recuperación de información clásicos y modernos. El sistema indexa un corpus de documentos textuales (Reuters ModApte) y permite ejecutar consultas de texto libre, devolviendo un ranking ordenado por relevancia.

Los modelos implementados incluyen:

- **Modelos Léxicos:** Similitud Jaccard (binario), TF-IDF (similitud coseno) y BM25.
- **Modelo Semántico:** Representación de vectores densos utilizando Sentence Transformers (`all-MiniLM-L6-v2`) y búsqueda de similitud vectorial con FAISS.

## Autores

- Alexis Bautista
- Francisco Correa

## Requisitos Previos

El proyecto está desarrollado en Python. Se recomienda utilizar un entorno virtual. Para instalar todas las dependencias necesarias, ejecute el siguiente comando en la raíz del proyecto:

```bash
pip install -r requirements.txt
```

**Nota sobre NLTK:** La primera vez que se ejecute el sistema, el módulo de preprocesamiento descargará automáticamente la base de datos de _stopwords_ de NLTK necesaria para la tokenización.

## Estructura del Proyecto

- `/corpus/`: Directorio que debe contener los documentos fuente en formato `.csv` (ej. `ModApte_train.csv`, `ModApte_test.csv`).
- `preprocesamiento.py`: Módulo para la carga del corpus, limpieza de texto, tokenización, normalización y remoción de stopwords.
- `indice_invertido.py`: Estructura de datos que almacena el vocabulario y las frecuencias de los términos (TF) por documento.
- `modelos.py`: Lógica matemática y algoritmos de recuperación para Jaccard, TF-IDF y BM25.
- `modelo_semantico.py`: Construcción del índice vectorial y lógica de recuperación mediante embeddings.
- `evaluacion.py`: Utilidades para la generación de la verdad terrestre (QRELS) y cálculo de métricas (Precision@k, Recall@k, MAP).
- `main.py`: Interfaz de Línea de Comandos (CLI) interactiva.
- `Proyecto.ipynb`: Informe técnico, pruebas de concepto y evaluación detallada de los modelos.

## Instrucciones de Ejecución

### 1. Preparación de los datos

Asegúrese de que todos los archivos `.csv` del corpus estén ubicados dentro de la carpeta `/corpus/` en el directorio raíz del proyecto.

### 2. Ejecución de la Interfaz Interactiva (CLI)

Para interactuar con el motor de búsqueda y probar consultas de texto libre, ejecute el archivo principal desde su terminal:

```bash
python main.py

```

**Consideración importante sobre el rendimiento:**
Durante la primera ejecución, el sistema procesará los embeddings para todo el corpus y construirá la base de datos de FAISS. Este proceso puede tardar varios minutos dependiendo de los recursos del sistema. Al finalizar, se generará un archivo local llamado `indice_corpus.bin`. En ejecuciones posteriores, el sistema detectará este archivo y la carga del motor será casi instantánea.

### 3. Evaluación y Revisión del Informe Técnico

Para revisar el análisis comparativo, las métricas de evaluación (Precision, Recall y MAP) y las decisiones de diseño arquitectónico, inicie el entorno de Jupyter y abra el informe:

```bash
jupyter notebook Proyecto.ipynb

```

Ejecute las celdas en orden secuencial. El notebook importará automáticamente los módulos del sistema y presentará las tablas comparativas finales.
