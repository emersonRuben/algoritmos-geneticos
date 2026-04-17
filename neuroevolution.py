import numpy as np
import random
import matplotlib.pyplot as plt
import warnings
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Silenciar advertencias para que la consola esté limpia
warnings.filterwarnings("ignore")

# -------------------- CONFIGURACIÓN DEL PROBLEMA --------------------
print("=== PROBLEMA: Neuroevolution - Optimización de Neuronas (Iris) ===")
datos = load_iris()
X, y = datos.data, datos.target

# Escalado de características (CRÍTICO para Redes Neuronales)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------- PARÁMETROS DEL AG --------------------
NUM_GENERACIONES = 15      
TAM_POBLACION = 20       
NUM_GENES = 1              # Un solo gen representa el número de neuronas
RANGO_NEURONAS = (2, 100)  # De 2 a 100 neuronas en la capa oculta
PROB_MUTACION = 0.2

random.seed(42)
np.random.seed(42)

# -------------------- FUNCIÓN DE APTITUD --------------------
def evaluar_aptitud(cromosoma, X, y):
    # REPRESENTACIÓN: El cromosoma es una lista con un valor entero
    n_neuronas = int(cromosoma[0])
    if n_neuronas < 1: n_neuronas = 1
    
    # Creación del modelo con la arquitectura del cromosoma
    modelo = MLPClassifier(hidden_layer_sizes=(n_neuronas,), max_iter=300, random_state=42)
    
    try:
        modelo.fit(X, y)
        precision = modelo.score(X, y)
        return precision
    except:
        return 0.0

# -------------------- INICIALIZACIÓN --------------------
def crear_poblacion(tam_poblacion, num_genes):
    # Cada individuo es una lista con un número aleatorio de neuronas
    return [[random.randint(RANGO_NEURONAS[0], RANGO_NEURONAS[1])] for _ in range(tam_poblacion)]

# -------------------- SELECCIÓN (TORNEO) --------------------
def seleccion_torneo(poblacion, aptitudes, k=3):
    participantes = random.sample(list(zip(poblacion, aptitudes)), k)
    mejor = max(participantes, key=lambda x: x[1])
    return mejor[0]

# -------------------- CRUZAMIENTO --------------------
def cruce_promedio(padre1, padre2):
    # Para neuroevolution con un gen, el cruce puede ser el promedio
    hijo1 = [int((padre1[0] + padre2[0]) / 2)]
    hijo2 = [random.randint(RANGO_NEURONAS[0], RANGO_NEURONAS[1])] # Exploración aleatoria
    return hijo1, hijo2

# -------------------- MUTACIÓN --------------------
def mutacion(cromosoma, prob_mutacion):
    if random.random() < prob_mutacion:
        # Sumamos o restamos un número pequeño de neuronas
        cambio = random.randint(-5, 5)
        cromosoma[0] = max(RANGO_NEURONAS[0], min(RANGO_NEURONAS[1], cromosoma[0] + cambio))
    return cromosoma

# -------------------- CICLO PRINCIPAL --------------------
print("=== INICIO DEL ALGORITMO GENÉTICO ===")
print(f"{'Generación':>10} | {'Mejor Precisión':>15} | {'Neuronas':>10}")
print("-" * 45)

poblacion = crear_poblacion(TAM_POBLACION, NUM_GENES)
mejor_aptitud_global = -float('inf')
mejor_cromosoma_global = None
historial_aptitud = []

for generacion in range(NUM_GENERACIONES):
    aptitudes = [evaluar_aptitud(ind, X_scaled, y) for ind in poblacion]
    
    mejor_aptitud_gen = max(aptitudes)
    idx_mejor = aptitudes.index(mejor_aptitud_gen)
    mejor_cromosoma_gen = poblacion[idx_mejor]
    
    if mejor_aptitud_gen > mejor_aptitud_global:
        mejor_aptitud_global = mejor_aptitud_gen
        mejor_cromosoma_global = mejor_cromosoma_gen.copy()
    
    historial_aptitud.append(mejor_aptitud_global)
    print(f"{generacion+1:10d} | {mejor_aptitud_global:15.4f} | {int(mejor_cromosoma_gen[0]):10d}")
    
    nueva_poblacion = [mejor_cromosoma_gen.copy()] # Elitismo
    while len(nueva_poblacion) < TAM_POBLACION:
        padre1 = seleccion_torneo(poblacion, aptitudes)
        padre2 = seleccion_torneo(poblacion, aptitudes)
        hijo1, hijo2 = cruce_promedio(padre1, padre2)
        hijo1 = mutacion(hijo1, PROB_MUTACION)
        hijo2 = mutacion(hijo2, PROB_MUTACION)
        nueva_poblacion.extend([hijo1, hijo2])
    poblacion = nueva_poblacion[:TAM_POBLACION]

# -------------------- RESULTADOS FINALES --------------------
print("\n=== RESULTADO FINAL ===")
print(f"Número de neuronas óptimo: {int(mejor_cromosoma_global[0])}")
print(f"Precisión máxima lograda: {mejor_aptitud_global * 100:.2f}%")

# -------------------- GRÁFICA --------------------
plt.figure(figsize=(10, 5))
plt.plot(range(1, NUM_GENERACIONES + 1), historial_aptitud, marker='s', color='green')
plt.xlabel('Generación')
plt.ylabel('Precisión (Aptitud)')
plt.title('Neuroevolution: Evolución del número de neuronas (Iris)')
plt.grid(True, alpha=0.3)
plt.show()