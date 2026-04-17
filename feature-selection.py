import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# -------------------- TRADUCCIÓN DE CARACTERÍSTICAS (Iris) --------------------
traduccion = {
    'sepal length (cm)': 'Longitud del sépalo (cm)',
    'sepal width (cm)': 'Ancho del sépalo (cm)',
    'petal length (cm)': 'Longitud del pétalo (cm)',
    'petal width (cm)': 'Ancho del pétalo (cm)'
}

# -------------------- CONFIGURACIÓN DEL PROBLEMA --------------------
print("=== PROBLEMA: Selección de mejores características (Iris) ===")
datos = load_iris()
X, y = datos.data, datos.target
nombres_originales = datos.feature_names
nombres_espanol = [traduccion[n] for n in nombres_originales]

print(f"Total de características disponibles: {len(nombres_espanol)}")
print("Listado completo de características:")
for i, nombre in enumerate(nombres_espanol, 1):
    print(f"  {i:2d}. {nombre}")
print()

# -------------------- PARÁMETROS DEL AG --------------------
NUM_GENERACIONES = 20      # Suficiente para 4 características
TAM_POBLACION = 50       # Población pequeña
NUM_GENES = len(nombres_espanol)   # = 4
PROB_MUTACION = 0.1
ALPHA_PENALIZACION = 0.005

random.seed(42)
np.random.seed(42)

# -------------------- FUNCIÓN DE APTITUD --------------------evalua 
def evaluar_aptitud(cromosoma, X, y):
    caracteristicas = [i for i, bit in enumerate(cromosoma) if bit == 1]  #1,0,1,0
    if len(caracteristicas) == 0:
        return 0.0
    modelo = KNeighborsClassifier(n_neighbors=3, n_jobs=1)
    modelo.fit(X[:, caracteristicas], y)
    precision = modelo.score(X[:, caracteristicas], y)
    return precision - ALPHA_PENALIZACION * len(caracteristicas)

# -------------------- INICIALIZACIÓN --------------------
def crear_poblacion(tam_poblacion, num_genes):
    return [np.random.randint(0, 2, num_genes).tolist() for _ in range(tam_poblacion)]

# -------------------- SELECCIÓN (TORNEO) --------------------
def seleccion_torneo(poblacion, aptitudes, k=2):
    participantes = random.sample(list(zip(poblacion, aptitudes)), k)   #[([1,0,1,0], 0.94), ([0,1,1,0], 0.87), ([1,1,0,0], 0.91), ...]
    mejor = max(participantes, key=lambda x: x[1])
    return mejor[0]

# -------------------- CRUZAMIENTO --------------------
def cruce_un_punto(padre1, padre2):
    punto = random.randint(1, NUM_GENES - 1)
    hijo1 = padre1[:punto] + padre2[punto:]   #[1, 0, 1, 0]
    hijo2 = padre2[:punto] + padre1[punto:]   #[0, 1, 1, 0]
    return hijo1, hijo2

# -------------------- MUTACIÓN --------------------
def mutacion(cromosoma, prob_mutacion):     #[1,0,1,0]
    for i in range(len(cromosoma)):            #Recorre cada posición del cromosoma
        if random.random() < prob_mutacion:
            cromosoma[i] = 1 - cromosoma[i]
    return cromosoma

# -------------------- CICLO PRINCIPAL --------------------
print("=== INICIO DEL ALGORITMO GENÉTICO ===")
print(f"{'Generación':>10} | {'Mejor aptitud':>15} | {'Caract. sel.':>12}")
print("-" * 45)

poblacion = crear_poblacion(TAM_POBLACION, NUM_GENES)
mejor_aptitud_global = -float('inf')
mejor_cromosoma_global = None
historial_aptitud = []

for generacion in range(NUM_GENERACIONES):
    aptitudes = [evaluar_aptitud(ind, X, y) for ind in poblacion]
    mejor_aptitud_gen = max(aptitudes)
    idx_mejor = aptitudes.index(mejor_aptitud_gen)
    mejor_cromosoma_gen = poblacion[idx_mejor]
    
    if mejor_aptitud_gen > mejor_aptitud_global:
        mejor_aptitud_global = mejor_aptitud_gen
        mejor_cromosoma_global = mejor_cromosoma_gen.copy()
    
    historial_aptitud.append(mejor_aptitud_global)
    num_caract = sum(mejor_cromosoma_global)
    print(f"{generacion+1:10d} | {mejor_aptitud_global:15.4f} | {int(num_caract):12d}")
    
    nueva_poblacion = [mejor_cromosoma_gen.copy()]
    while len(nueva_poblacion) < TAM_POBLACION:
        padre1 = seleccion_torneo(poblacion, aptitudes)
        padre2 = seleccion_torneo(poblacion, aptitudes)
        hijo1, hijo2 = cruce_un_punto(padre1, padre2)
        hijo1 = mutacion(hijo1, PROB_MUTACION)
        hijo2 = mutacion(hijo2, PROB_MUTACION)
        nueva_poblacion.extend([hijo1, hijo2])
    poblacion = nueva_poblacion[:TAM_POBLACION]

# -------------------- RESULTADOS FINALES --------------------
print("\n=== RESULTADO FINAL ===")
print(f"Cromosoma óptimo (lista): {mejor_cromosoma_global}")

caract_finales = [i for i, bit in enumerate(mejor_cromosoma_global) if bit == 1]
caract_no_seleccionadas = NUM_GENES - len(caract_finales)
modelo_final = KNeighborsClassifier(n_neighbors=3, n_jobs=1)
modelo_final.fit(X[:, caract_finales], y)
precision_real = modelo_final.score(X[:, caract_finales], y)

print(f"Precisión real (sin penalización): {precision_real * 100:.2f}%")
print(f"Aptitud final (con penalización): {mejor_aptitud_global:.4f}")
print(f"Número de características seleccionadas: {len(caract_finales)}")
print(f"Número de características no seleccionadas: {caract_no_seleccionadas}")

print("\n--- Características completas (seleccionadas / descartadas) ---")
print(f"{'#':<3} {'Característica':<30} {'Estado':<12}")
print("-" * 48)
for i, bit in enumerate(mejor_cromosoma_global):
    estado = "ELEGIDA" if bit == 1 else "DESCARTADA"
    print(f"{i+1:<3} {nombres_espanol[i]:<30} {estado:<12}")

# -------------------- GRÁFICA --------------------
plt.figure(figsize=(10, 5))
plt.plot(range(1, NUM_GENERACIONES + 1), historial_aptitud, marker='o')
plt.xlabel('Generación')
plt.ylabel('Mejor aptitud (precisión - penalización)')
plt.title('Evolución del AG para selección de características (Iris)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()