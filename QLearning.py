import numpy as np 
import random

# Q-Learning en Machine Learning

dimensiones = (5, 5)
estado_incial = (0, 0)
estado_objetivo = (4, 4)
obstaculos = [(1, 1), (1, 3), (2, 3), (3, 0)]
acciones = [(-1, 0), (1, 0), (0, -1), (0, 1)]

num_estados = dimensiones[0] * dimensiones[1]
print(num_estados)

num_acciones = len(acciones)
print(num_acciones)

Q = np.zeros((num_estados, num_acciones))
print(Q)

def estado_a_indice(estado):
    return estado[0] * dimensiones[1] + estado[1]
ejemplo = estado_a_indice((0, 0))
print(ejemplo)

alpha = 0.1
gamma = 0.99 
epsilon = 0.2 # Nivel de Porcentaje 
episodios = 100

def elegir_accion(estado):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(num_acciones))
    else: 
        return np.argmax(Q[estado_a_indice(estado)])
    
def aplicar_accion(estado, accion_idx):
    accion = acciones[accion_idx]
    nuevo_estado = tuple(np.add(estado,accion) % dimensiones)

    if nuevo_estado in obstaculos or nuevo_estado == estado:
        return estado, -100, False
    if nuevo_estado == estado_objetivo:
        return nuevo_estado, 100, True
    return nuevo_estado, -1, False

for episodios in range(episodios):
    estado = estado_incial
    terminado = False

    while not terminado:
        idx_estado = estado_a_indice(estado)
        accion_idx = elegir_accion(estado)
        nuevo_estado, recompensa, terminado = aplicar_accion(estado, accion_idx)
        idx_nuevo_estado = estado_a_indice(nuevo_estado)

        Q[idx_estado, accion_idx] = Q[idx_estado, accion_idx] + alpha + (recompensa + gamma * np.max(Q[idx_nuevo_estado]) - Q[idx_estado, accion_idx])
        estado = nuevo_estado

politica = np.zeros(dimensiones, dtype=int)
print(politica)

for i in range(dimensiones[0]):
    for j in range(dimensiones[1]):
        estado = (i, j)
        idx_estado = estado_a_indice(estado)
        mejor_accion = np.argmax(Q[idx_estado])
        politica[i, j] = mejor_accion
print("Política aprendida (0: arriba, 1: abajo, 2: izquierda, 3: derecha)")
print(politica)