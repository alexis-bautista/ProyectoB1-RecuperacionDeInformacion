# ProyectoB1-Recuperacion-de-Informacion

Sistema de recuperacion de informacion implementado en Python para comparar modelos clasicos y semanticos sobre un corpus textual tipo Reuters-21578.

## Requisitos

- Python 3.10 o superior
- Jupyter Notebook o VS Code con soporte para notebooks
- Paquetes Python:
	- numpy
	- pandas
	- nltk
	- scikit-learn
	- sentence-transformers
	- faiss-cpu

Instalacion sugerida:

```bash
pip install numpy pandas nltk scikit-learn sentence-transformers faiss-cpu notebook
```

## Instrucciones de ejecucion

1. Abrir el archivo [Proyecto.ipynb](Proyecto.ipynb).
2. Ejecutar las celdas desde el inicio hasta el final, en orden.
3. El notebook realiza las siguientes etapas:
	 - carga del corpus CSV desde la carpeta `corpus/`
	 - preprocesamiento de texto
	 - construccion del indice invertido
	 - ejecucion de Jaccard, TF-IDF y BM25
	 - construccion del indice vectorial FAISS con embeddings
	 - evaluacion con Precision, Recall y MAP

## Estructura del proyecto

- `preprocesamiento.py`: carga y limpieza del corpus
- `indice_invertido.py`: construccion del indice invertido
- `modelos.py`: Jaccard, TF-IDF y BM25
- `modelo_semantico.py`: embeddings y busqueda con FAISS
- `evaluacion.py`: metrica de evaluacion y comparacion de modelos
- `Proyecto.ipynb`: ejecucion principal y demostracion de resultados

## Resultados esperados

La salida final del notebook incluye una tabla comparativa de modelos con:

- `MAP`
- `Precision promedio@10`
- `Recall promedio@10`

En la prueba realizada, TF-IDF obtuvo el mejor desempeno general, seguido por BM25.

## Notas

- El corpus proviene de archivos CSV ya preparados dentro de `corpus/`.
- La recuperacion semantica usa FAISS, que cumple con el requisito de base vectorial.
- Los qrels de evaluacion se construyen a partir de la columna `topics` del corpus de prueba.