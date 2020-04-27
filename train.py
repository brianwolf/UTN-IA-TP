from tensorflow.python.keras.layers import Flatten, Dropout, Reshape
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import tensorflow.keras.backend as backend

# Using these two Pokemon Generation One datasets
# https://www.kaggle.com/thedagger/pokemon-generation-one
# https://github.com/Akshay090/pokemon-image-dataset

# From these datasets we mixed the folders "Charmander, Pikachu, Bulbasaur and Squirtle"
# Then we got only relevant pictures, removing unrelated or noisy pictures, pictures of toys, etc.
# We then normalized the training files and removed the background.
# In the output folders you can see automatically resized and grayscaled images.
# We use a 80/20 training/validation ratio.
# In the tests folder you would put the "target" pictures to try to predict.

# Folders Setup
# Data Folder
data_dir = os.path.join(os.path.dirname('.'), 'data')
train_dir = os.path.join(data_dir, 'train')
validation_dir = os.path.join(data_dir, 'validation')
output_train_dir = os.path.join(data_dir, 'output_train')
output_validation_dir = os.path.join(data_dir, 'output_validation')
total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_val = sum([len(files) for r, d, files in os.walk(validation_dir)])

# Model Folder
model_dir = os.path.join(os.path.dirname('.'), 'model')
model_file = os.path.join(model_dir, 'model.h5')
weights_file = os.path.join(model_dir, 'weights.h5')
model_plot_file = os.path.join(model_dir, 'Model.png')

# Basic Params
batch_size = 2
epochs = 100
IMG_HEIGHT = 100
IMG_WIDTH = 100
total_classes = 4  # 4 different types of pokemons in our dataset
learning_rate = 0.001
if backend.image_data_format == "channels_last":
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 1)
else:
    input_shape = (1, IMG_HEIGHT, IMG_WIDTH)

# Get Images
train_image_generator = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    # vertical_flip=True,
    # rotation_range=45,
    # zoom_range=0.5,
    # shear_range=0.5,
    # width_shift_range=0.5,
    # height_shift_range=0.5,)
    )

validation_image_generator = ImageDataGenerator(
    rescale=1. / 255)

train_data_gen = train_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode="grayscale",
    class_mode='categorical',
    save_to_dir=output_train_dir)

val_data_gen = validation_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=validation_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode="grayscale",
    class_mode='categorical',
    save_to_dir=output_validation_dir)

# Generate the model
model = Sequential([
    Reshape((IMG_HEIGHT * IMG_WIDTH,), input_shape=(IMG_HEIGHT, IMG_WIDTH,)),
    Dense(units=20480, activation='relu'),
    Dense(total_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=optimizers.Adam(lr=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print a summary of the model
model.summary()

# Plot the model to understand it
plot_model(model, to_file=model_plot_file, show_shapes='true', show_layer_names='true')

# Train the model
history = model.fit(
    x=train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

# Save the model results
model.save(model_file)
model.save_weights(weights_file)

# Plot the results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
