import fnmatch
import os
import random
from enum import Enum
from os import listdir
from os.path import isfile, join
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Folders Setup
# Data Folder
data_dir = os.path.join(os.path.dirname('.'), 'data')
tests_dir = os.path.join(data_dir, 'tests')
tests_bulbasaur_dir = os.path.join(tests_dir, 'Bulbasaur')
tests_charmander_dir = os.path.join(tests_dir, 'Charmander')
tests_pikachu_dir = os.path.join(tests_dir, 'Pikachu')
tests_squirtle_dir = os.path.join(tests_dir, 'Squirtle')
num_bulbasaur_val = (len(fnmatch.filter(os.listdir(tests_bulbasaur_dir), '*')) - 1) * 2
num_charmander_val = (len(fnmatch.filter(os.listdir(tests_charmander_dir), '*')) - 1) * 2
num_pikachu_val = (len(fnmatch.filter(os.listdir(tests_pikachu_dir), '*')) - 1) * 2
num_squirtle_val = (len(fnmatch.filter(os.listdir(tests_squirtle_dir), '*')) - 1) * 2
bulbasaur_files = [f for f in listdir(tests_bulbasaur_dir) if isfile(join(tests_bulbasaur_dir, f))]
charmander_files = [f for f in listdir(tests_charmander_dir) if isfile(join(tests_charmander_dir, f))]
pikachu_files = [f for f in listdir(tests_pikachu_dir) if isfile(join(tests_pikachu_dir, f))]
squirtle_files = [f for f in listdir(tests_squirtle_dir) if isfile(join(tests_squirtle_dir, f))]
bulbasaur_files.remove('.gitignore')
charmander_files.remove('.gitignore')
pikachu_files.remove('.gitignore')
squirtle_files.remove('.gitignore')

# Model Folder
model_dir = os.path.join(os.path.dirname('.'), 'model')
model_file = os.path.join(model_dir, 'model.h5')
weights_file = os.path.join(model_dir, 'weights.h5')

# Basic Params
IMG_HEIGHT = 100
IMG_WIDTH = 100

# Model Loading
model = load_model(model_file)
model.load_weights(weights_file)


# PokemonType Enum
class PokemonType(Enum):
    BULBASAUR = 'BULBASAUR'
    CHARMANDER = 'CHARMANDER'
    PIKACHU = 'PIKACHU'
    SQUIRTLE = "SQUIRTLE"


# Predict Function
def predict(file: str) -> PokemonType:
    test_image = load_img(file, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale')
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = np.vstack([test_image])
    classes = model.predict(test_image)
    answer = np.argmax(classes, axis=-1)

    if answer == 0:
        return PokemonType.BULBASAUR
    elif answer == 1:
        return PokemonType.CHARMANDER
    elif answer == 2:
        return PokemonType.PIKACHU
    elif answer == 3:
        return PokemonType.SQUIRTLE


# BULBASAUR TESTS
true_positive_bulbasaur = 0
false_positive_bulbasaur = 0
true_negative_bulbasaur = 0
false_negative_bulbasaur = 0
for bulbasaur_file in bulbasaur_files:

    print(f'Bulbasaur: {bulbasaur_file}')

    bulbasaur_image = os.path.join(tests_bulbasaur_dir, bulbasaur_file)
    if predict(bulbasaur_image) == PokemonType.BULBASAUR:
        true_positive_bulbasaur += 1
    else:
        false_negative_bulbasaur += 1

    random_charmander_number = random.randint(0, len(charmander_files) - 1)
    charmander_image = os.path.join(tests_charmander_dir, charmander_files[random_charmander_number])
    if predict(charmander_image) == PokemonType.BULBASAUR:
        false_positive_bulbasaur += 1
    else:
        true_negative_bulbasaur += 1

# CHARMANDER TESTS
true_positive_charmander = 0
false_positive_charmander = 0
true_negative_charmander = 0
false_negative_charmander = 0
for charmander_file in charmander_files:

    print(f'Charmander: {charmander_file}')

    charmander_image = os.path.join(tests_charmander_dir, charmander_file)
    if predict(charmander_image) == PokemonType.CHARMANDER:
        true_positive_charmander += 1
    else:
        false_negative_charmander += 1

    random_pikachu_number = random.randint(0, len(pikachu_files) - 1)
    pikachu_image = os.path.join(tests_pikachu_dir, pikachu_files[random_pikachu_number])
    if predict(pikachu_image) == PokemonType.CHARMANDER:
        false_positive_charmander += 1
    else:
        true_negative_charmander += 1

# PIKACHU TESTS
true_positive_pikachu = 0
false_positive_pikachu = 0
true_negative_pikachu = 0
false_negative_pikachu = 0
for pikachu_file in pikachu_files:

    print(f'Pikachu: {pikachu_file}')

    pikachu_image = os.path.join(tests_pikachu_dir, pikachu_file)
    if predict(pikachu_image) == PokemonType.PIKACHU:
        true_positive_pikachu += 1
    else:
        false_negative_pikachu += 1

    random_squirtle_number = random.randint(0, len(squirtle_files) - 1)
    squirtle_image = os.path.join(tests_squirtle_dir, squirtle_files[random_squirtle_number])
    if predict(squirtle_image) == PokemonType.PIKACHU:
        false_positive_pikachu += 1
    else:
        true_negative_pikachu += 1


# SQUIRTLE TESTS
true_positive_squirtle = 0
false_positive_squirtle = 0
true_negative_squirtle = 0
false_negative_squirtle = 0
for squirtle_file in squirtle_files:

    print(f'Squirtle: {squirtle_file}')

    squirtle_image = os.path.join(tests_squirtle_dir, squirtle_file)
    if predict(squirtle_image) == PokemonType.SQUIRTLE:
        true_positive_squirtle += 1
    else:
        false_negative_squirtle += 1

    random_bulbasaur_number = random.randint(0, len(bulbasaur_files) - 1)
    bulbasaur_image = os.path.join(tests_bulbasaur_dir, bulbasaur_files[random_bulbasaur_number])
    if predict(bulbasaur_image) == PokemonType.SQUIRTLE:
        false_positive_squirtle += 1
    else:
        true_negative_squirtle += 1

print('\n\nRESULTS:\n')
print(
    f'Bulb -> TP: {true_positive_bulbasaur}, '
    f'Bulb -> FP: {false_positive_bulbasaur}, '
    f'Bulb -> TN: {true_negative_bulbasaur}, '
    f'Bulb -> FN: {false_negative_bulbasaur}, '
    f'Total Tests: {num_bulbasaur_val}, '
    f'Accuracy: {(true_positive_bulbasaur + true_negative_bulbasaur) / (true_positive_bulbasaur + false_positive_bulbasaur + true_negative_bulbasaur + false_negative_bulbasaur)}, ',
    f'Precision: {true_positive_bulbasaur / (true_positive_bulbasaur + false_positive_bulbasaur)}, ',
    f'Recall: {true_positive_bulbasaur / (true_positive_bulbasaur + false_negative_bulbasaur)}',
    )
print(
    f'Char -> TP: {true_positive_charmander}, '
    f'Char -> FP: {false_positive_charmander}, '
    f'Char -> TN: {true_negative_charmander}, '
    f'Char -> FN: {false_negative_charmander}, '
    f'Total Tests: {num_charmander_val}, '
    f'Accuracy: {(true_positive_charmander + true_negative_charmander) / (true_positive_charmander + false_positive_charmander + true_negative_charmander + false_negative_charmander)}, ',
    f'Precision: {true_positive_charmander / (true_positive_charmander + false_positive_charmander)}, ',
    f'Recall: {true_positive_charmander / (true_positive_charmander + false_negative_charmander)}',
    )
print(
    f'Pika -> TP: {true_positive_pikachu}, '
    f'Pika -> FP: {false_positive_pikachu}, '
    f'Pika -> TN: {true_negative_pikachu}, '
    f'Pika -> FN: {false_negative_pikachu}, '
    f'Total Tests: {num_pikachu_val}, '
    f'Accuracy: {(true_positive_pikachu + true_negative_pikachu) / (true_positive_pikachu + false_positive_pikachu + true_negative_pikachu + false_negative_pikachu)}, ',
    f'Precision: {true_positive_pikachu / (true_positive_pikachu + false_positive_pikachu)}, ',
    f'Recall: {true_positive_pikachu / (true_positive_pikachu + false_negative_pikachu)}',
    )
print(
    f'Squi -> TP: {true_positive_squirtle}, '
    f'Squi -> FP: {false_positive_squirtle}, '
    f'Squi -> TN: {true_negative_squirtle}, '
    f'Squi -> FN: {false_negative_squirtle}, '
    f'Total Tests: {num_squirtle_val}, '
    f'Accuracy: {(true_positive_squirtle + true_negative_squirtle) / (true_positive_squirtle + false_positive_squirtle + true_negative_squirtle + false_negative_squirtle)}, ',
    f'Precision: {true_positive_squirtle / (true_positive_squirtle + false_positive_squirtle)}, ',
    f'Recall: {true_positive_squirtle / (true_positive_squirtle + false_negative_squirtle)}',
    )
