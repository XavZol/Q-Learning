import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

# Redes Q Profundas técnica de aprendizaje por refuerzo

(imagenes_entrenamiento, etiquetas_entrenamiento), (imagenes_prueba, etiquetas_prueba) = mnist.load_data()

imagenes_entrenamiento = imagenes_entrenamiento / 255.0
imagenes_prueba = imagenes_prueba / 255.0

etiquetas_entrenamiento[0]

etiquetas_entrenamiento = to_categorical(etiquetas_entrenamiento)
etiquetas_prueba = to_categorical(etiquetas_prueba)

etiquetas_entrenamiento[0]

modelo = Sequential([
    Flatten(input_shape=(28, 28)), 
    Dense(128, activation='relu'), 
    Dense(10, activation='softmax')
])         # Pila de Capas

modelo.compile(optimizer='adam',
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

modelo.fit(imagenes_entrenamiento,
            etiquetas_entrenamiento, 
            epochs=5,
            validation_data=(imagenes_prueba, etiquetas_prueba))

predicciones = modelo.predict(imagenes_prueba)

def ver_imagen(array_predicciones, etiqueta_real, img):
    etiqueta_real, img = etiqueta_real.argmax(), img.squeeze()
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)
    
    etiqueta_predicha = np.argmax(array_predicciones)
    if etiqueta_predicha == etiqueta_real:
        color = 'blue'
    else: 
        color = 'red'
    plt.xlabel(f'Pred: {etiqueta_predicha} Real: {etiqueta_real}', color=color)

filas = 5
columnas = 3 
numero_imagenes = filas * columnas
plt.figure(figsize=(2 * 2 * columnas, 2 * filas))

for i in range(numero_imagenes):
    plt.subplot(filas, 2 * columnas, 2 * i + 1)
    ver_imagen(predicciones[i], etiquetas_prueba[i], imagenes_prueba[i])

plt.show()