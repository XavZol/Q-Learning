# Q(s,a) = Q(s, a) + + — Q(8, a))

import numpy as np
import random
import matplotlib.pyplot as plt

dimensiones = (4, 4)
estado_inicial = (0, 0)
estado_objetivo = (3, 3)
acciones = [(0, -1), (0, 1), (-1, 0), (1, 0)]
acciones_simbolos = ['↑', '↓', '←', '→']
num_estados = dimensiones[0] * dimensiones[1]
print(num_estados)  

num_acciones = len(acciones)
print(num_acciones)

Q = np.zeros((num_estados, num_acciones))
print(Q)

alpha = 0.1
gamma = 0.99
epsilon = 0.2
episodios = 1000

# Parámetros Sarsa

def estado_indice(estado):
    return estado[0] * dimensiones[1] + estado[1]

print(estado_indice((3, 0)))

def elegir_accion(estado):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, num_acciones - 1)
    else: 
        return np.argmax(Q[estado_indice(estado)])

def aplicar_accion(estado, accion_idx):
    accion = acciones[accion_idx]
    nuevo_estado = tuple(np.add(estado, accion) % np.array(dimensiones))

    if nuevo_estado == estado_objetivo:
        recompensa = 1 
    else: 
        recompensa = -1
    
    return nuevo_estado, recompensa, nuevo_estado == estado_objetivo

for epsilon in range(episodios):
    estado = estado_inicial
    accion_idx = elegir_accion(estado)
    terminado = False

    while not terminado:
        nuevo_estado , recompensa , terminado = aplicar_accion(estado, accion_idx)
        nueva_accion_idx = elegir_accion(nuevo_estado)

        indice = estado_indice(estado)
        Q[indice, accion_idx] += alpha * (recompensa + gamma * Q[estado_indice(nuevo_estado), nueva_accion_idx] - Q[indice, accion_idx])
        # Lógica del algoritmo
        estado, accion_idx = nuevo_estado, nueva_accion_idx

politica_simbolos = np.empty(dimensiones, dtype='<U2')
for i in range(dimensiones[0]):
    for j in range(dimensiones[1]):
        estado = (i, j)
        mejor_accion = np.argmax(Q[estado_indice(estado)])
        politica_simbolos[i, j] = acciones_simbolos[mejor_accion]
        
print(politica_simbolos)


