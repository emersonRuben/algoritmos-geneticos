# Algoritmos Genéticos para Optimización de Hiperparámetros

## Descripción

Este proyecto presenta la aplicación de un Algoritmo Genético (AG) para optimizar hiperparámetros de un modelo de clasificación en un problema de aprendizaje automático: predicción de abandono de clientes (churn bancario).

La optimización se realiza sobre un `RandomForestClassifier`, buscando de manera automática la combinación de parámetros que maximiza el desempeño del modelo en la métrica F1.

## Objetivo

Diseñar e implementar un AG que permita encontrar configuraciones de hiperparámetros con mejor rendimiento que una selección manual o aleatoria simple, utilizando validación cruzada para una evaluación más robusta.

## Metodología

El flujo implementado en el notebook sigue estas etapas:

1. Carga y preparación de datos.
2. Definición del espacio de búsqueda de hiperparámetros.
3. Evaluación de aptitud por validación cruzada.
4. Evolución de población mediante selección, cruce y mutación.
5. Conservación del mejor individuo (elitismo).
6. Reporte de la mejor solución encontrada.

## Componentes del Algoritmo Genético

### Representación

Cada individuo se modela como un diccionario de hiperparámetros de Random Forest:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`
- `bootstrap`

### Inicialización

La población inicial se crea aleatoriamente dentro de rangos definidos para cada hiperparámetro.

### Función de Aptitud

La aptitud de cada individuo se calcula con validación cruzada (`cv=3`) usando la métrica `F1`.

### Selección

Se utiliza selección por torneo (`k=3`), eligiendo como padre al individuo con mejor aptitud entre los participantes sorteados.

### Cruzamiento

Se aplica cruce uniforme: cada gen del hijo se hereda aleatoriamente de uno de los dos padres.

### Mutación

Cada gen puede mutar con baja probabilidad (`prob=0.1`) para introducir diversidad sin perder estabilidad evolutiva.

### Terminación

La ejecución finaliza tras un número fijo de generaciones. Al terminar, se muestra el mejor F1 global y su configuración de hiperparámetros asociada.

## Estructura del Proyecto

- `main.ipynb`: notebook principal con la implementación completa del AG y ejecución.

## Requisitos

- Python 3.9 o superior.
- Librerías:
  - `pandas`
  - `scikit-learn`
  - `kagglehub`

Instalación recomendada:

```bash
pip install pandas scikit-learn kagglehub
```

## Ejecución

1. Abrir `main.ipynb` en Jupyter o Visual Studio Code.
2. Ejecutar las celdas en orden.
3. Revisar en consola:
   - evolución por generación,
   - torneos de selección,
   - cruces y mutaciones,
   - mejor resultado final.

## Alcance Académico

Este repositorio corresponde a una tarea de presentación sobre algoritmos genéticos aplicados a aprendizaje automático, con énfasis en la optimización de hiperparámetros y el análisis del proceso evolutivo.
