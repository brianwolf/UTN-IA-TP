from enum import Enum

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

longitud, altura = 150, 150
modelo = './model/model.h5'
pesos_modelo = './model/pesos.h5'

cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)


class TipoAnimal(Enum):
    GATO = 'GATO'
    PERRO = 'PERRO'
    GORILA = 'GORILA'
    DESCONOCIDO = 'DESCONOCIDO'


def predict(file: str) -> TipoAnimal:

    x = load_img(file, target_size=(longitud, altura))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)

    array = cnn.predict(x)
    result = array[0]
    answer = np.argmax(result)

    if answer == 0:
        return TipoAnimal.PERRO
    elif answer == 1:
        return TipoAnimal.GATO
    elif answer == 2:
        return TipoAnimal.GORILA

    return TipoAnimal.DESCONOCIDO


# PRUEBAS
aciertos_gato = 0
veces_gato = 200
for vez_gato in range(veces_gato):

    imagen_gato = f'data/validacion/gato/cat.{4001 + vez_gato}.jpg'
    if predict(imagen_gato) == TipoAnimal.GATO:
        aciertos_gato += 1

aciertos_perro = 0
veces_perro = 100
for vez_perro in range(veces_perro):

    imagen_perro = f'data/validacion/perro/dog.{4001 + vez_perro}.jpg'
    if predict(imagen_perro) == TipoAnimal.PERRO:
        aciertos_perro += 1

aciertos_gorila = 0
veces_gorila = 20
for vez_gorila in range(veces_gorila):

    imagen_gorila = f'data/validacion/gorila/{vez_gorila}.jpeg'
    if predict(imagen_gorila) == TipoAnimal.GORILA:
        aciertos_gorila += 1

print('\n\nRESULTADOS:\n')
print(
    f'GATOS -> aciertos:{aciertos_gato}, pruebas:{veces_gato}, porcentaje:{100 * aciertos_gato / veces_gato}%')
print(
    f'PERROS -> aciertos:{aciertos_perro}, pruebas:{veces_perro}, porcentaje: {100 * aciertos_perro / veces_perro}%')
print(
    f'GORILAS -> aciertos:{aciertos_gorila}, pruebas:{veces_gorila}, porcentaje: {100 * aciertos_gorila / veces_gorila}%')
