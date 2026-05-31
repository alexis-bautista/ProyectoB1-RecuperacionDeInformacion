# Informe tecnico

## 1. Descripcion del corpus utilizado

El proyecto utiliza un corpus de noticias en formato CSV derivado de Reuters-21578, organizado en archivos como `ModApte_train.csv`, `ModApte_test.csv`, `ModLewis_train.csv` y `ModHayes_train.csv`. Cada registro contiene el texto del documento y etiquetas tematicas en la columna `topics`.

Para la evaluacion se trabajo con `ModApte_test.csv`, con el fin de separar el conjunto de prueba del corpus principal. Las etiquetas tematicas permitieron construir qrels de manera reproducible: un documento se considera relevante para una consulta si el tema de la consulta aparece en su columna `topics`.

## 2. Decisiones de diseno

Se eligio Python por ser el lenguaje recomendado para este tipo de tareas y por la disponibilidad de bibliotecas para preprocesamiento, modelado y evaluacion.

El sistema se dividio en modulos para mantener el notebook limpio y facil de leer:

- `preprocesamiento.py` para carga y normalizacion.
- `indice_invertido.py` para construir el indice invertido.
- `modelos.py` para Jaccard, TF-IDF y BM25.
- `modelo_semantico.py` para embeddings y FAISS.
- `evaluacion.py` para Precision, Recall, AP y MAP.

Para los modelos clasicos se aplico tokenizacion simple, minusculas, eliminacion de stopwords y stemming. Para la recuperacion semantica no se uso preprocesamiento agresivo, porque afecta negativamente a los embeddings.

Se eligio FAISS como base vectorial porque cumple con el requisito del proyecto y permite busqueda eficiente por similitud vectorial.

## 3. Ejemplos de consultas y resultados

Se probaron consultas como:

- `earn`
- `acq`
- `crude`
- `trade`
- `money-fx`
- `grain`
- `cocoa`

Tambien se hizo una prueba libre con una consulta de ejemplo en la parte semantica. Cada modelo devuelve un ranking de documentos ordenado por relevancia.

En la visualizacion final del notebook se obtienen listas de resultados para Jaccard, TF-IDF, BM25 y recuperacion semantica. Adicionalmente, la evaluacion compara los modelos en una misma tabla.

## 4. Analisis de metricas de evaluacion

Se calcularon las metricas pedidas por el enunciado:

- Precision por consulta
- Recall por consulta
- MAP para todo el sistema

La comparacion realizada sobre el conjunto de prueba mostro el siguiente orden general de desempeno:

1. TF-IDF
2. BM25
3. Recuperacion semantica
4. Jaccard

Interpretacion:

- TF-IDF obtuvo el mejor resultado porque las consultas de prueba eran tematicas y coincidian bien con terminos concretos del corpus.
- BM25 quedo muy cerca, pero por debajo de TF-IDF en esta prueba.
- La recuperacion semantica no supero a los modelos clasicos en este conjunto, aunque sigue siendo util para consultas mas naturales o con sinonimos.
- Jaccard fue el mas basico y el menos efectivo, porque solo considera presencia o ausencia de terminos.

## 5. Conclusion

El sistema cumple con los requerimientos principales del proyecto: indice invertido, recuperacion clasica, recuperacion semantica con embeddings, evaluacion con qrels y comparacion de modelos. La mejor estrategia en este corpus especifico fue TF-IDF, mientras que la recuperacion semantica queda como una alternativa complementaria para consultas mas abiertas.