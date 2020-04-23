import fnmatch
import os
import random
from enum import Enum

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Folders Setup
# Data Folder
data_dir = os.path.join(os.path.dirname('.'), 'data')
validation_dir = os.path.join(data_dir, 'validation')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
validation_gorillas_dir = os.path.join(validation_dir, 'gorillas')
num_cats_val = len(fnmatch.filter(os.listdir(validation_cats_dir), '*.jpg'))
num_dogs_val = len(fnmatch.filter(os.listdir(validation_dogs_dir), '*.jpg'))
num_gorillas_val = len(fnmatch.filter(os.listdir(validation_gorillas_dir), '*.jpg'))

# Model Folder
model_dir = os.path.join(os.path.dirname('.'), 'model')
model_file = os.path.join(model_dir, 'model.h5')
weights_file = os.path.join(model_dir, 'weights.h5')

# Basic Params
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Model Loading
model = load_model(model_file)
model.load_weights(weights_file)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Categories Enum
class AnimalType(Enum):
    CAT = 'CAT'
    DOG = 'DOG'
    GORILLA = 'GORILLA'
    UNKNOWN = 'UNKNOWN'


# Predict Function
def predict(file: str) -> AnimalType:
    test_image = load_img(file, target_size=(IMG_HEIGHT, IMG_WIDTH))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    classes = model.predict(test_image)
    answer = np.argmax(classes, axis=-1)

    if answer == 0:
        return AnimalType.CAT
    elif answer == 1:
        return AnimalType.DOG
    elif answer == 2:
        return AnimalType.GORILLA

    return AnimalType.UNKNOWN


# CAT TESTS
true_positive_cat = 0
false_positive_cat = 0
true_negative_cat = 0
false_negative_cat = 0
for cat_iteration in range(num_cats_val):

    print(f'Cat: {cat_iteration}')

    cat_image = os.path.join(validation_cats_dir, f'{cat_iteration}.jpg')
    if predict(cat_image) == AnimalType.CAT:
        true_positive_cat += 1
    else:
        false_negative_cat += 1

    random_dog_number = random.randint(0, num_dogs_val - 1)
    dog_image = os.path.join(validation_dogs_dir, f'{random_dog_number}.jpg')
    if predict(dog_image) == AnimalType.CAT:
        false_positive_cat += 1
    else:
        true_negative_cat += 1

# DOG TESTS
true_positive_dog = 0
false_positive_dog = 0
true_negative_dog = 0
false_negative_dog = 0
for dog_iteration in range(num_dogs_val):

    print(f'Dog: {dog_iteration}')

    dog_image = os.path.join(validation_dogs_dir, f'{dog_iteration}.jpg')
    if predict(dog_image) == AnimalType.DOG:
        true_positive_dog += 1
    else:
        false_negative_dog += 1

    random_cat_number = random.randint(0, num_cats_val - 1)
    cat_image = os.path.join(validation_cats_dir, f'{random_cat_number}.jpg')
    if predict(cat_image) == AnimalType.DOG:
        false_positive_dog += 1
    else:
        true_negative_dog += 1

# GORILLA TESTS
true_positive_gorilla = 0
false_positive_gorilla = 0
true_negative_gorilla = 0
false_negative_gorilla = 0
for gorilla_iteration in range(num_gorillas_val):

    print(f'Gorilla: {gorilla_iteration}')

    imagen_gorila = os.path.join(validation_gorillas_dir, f'{gorilla_iteration}.jpg')
    if predict(imagen_gorila) == AnimalType.GORILLA:
        true_positive_gorilla += 1
    else:
        false_negative_gorilla += 1

    random_dog_number = random.randint(0, num_dogs_val - 1)
    dog_image = os.path.join(validation_dogs_dir, f'{random_dog_number}.jpg')
    if predict(dog_image) == AnimalType.GORILLA:
        false_positive_gorilla += 1
    else:
        true_negative_gorilla += 1

print('\n\nRESULTS:\n')
print(
    f'CATS -> True Positives: {true_positive_cat}, '
    f'Total Tests: {num_cats_val}, '
    f'Percentage: {100 * true_positive_cat / num_cats_val}%',
    f'Accuracy: {(true_positive_cat + true_negative_cat) / (true_positive_cat + false_positive_cat + true_negative_cat + false_negative_cat)}%',
    f'Precision: {true_positive_cat / (true_positive_cat + false_positive_cat)}%',
    f'Recall: {true_positive_cat / (true_positive_cat + false_negative_cat)}%',
    )
print(
    f'DOGS -> True Positives: {true_positive_dog}, '
    f'Total Tests: {num_dogs_val}, '
    f'Percentage: {100 * true_positive_dog / num_dogs_val}%',
    f'Accuracy: {(true_positive_dog + true_negative_dog) / (true_positive_dog + false_positive_dog + true_negative_dog + false_negative_dog)}%',
    f'Precision: {true_positive_dog / (true_positive_dog + false_positive_dog)}%',
    f'Recall: {true_positive_dog / (true_positive_dog + false_negative_dog)}%',
    )
print(
    f'GORILLAS -> True Positives: {true_positive_gorilla}, '
    f'Total Tests: {num_gorillas_val}, '
    f'Percentage: {100 * true_positive_gorilla / num_gorillas_val}%',
    f'Accuracy: {(true_positive_gorilla + true_negative_gorilla) / (true_positive_gorilla + false_positive_gorilla + true_negative_gorilla + false_negative_gorilla)}%',
    f'Precision: {true_positive_gorilla / (true_positive_gorilla + false_positive_gorilla)}%',
    f'Recall: {true_positive_gorilla / (true_positive_gorilla + false_negative_gorilla)}%',
    )
