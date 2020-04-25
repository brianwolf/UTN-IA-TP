import fnmatch

from tensorflow import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import matplotlib.pyplot as plt

# Based on https://www.tensorflow.org/tutorials/images/classification

# Folders Setup
# Data Folder
data_dir = os.path.join(os.path.dirname('.'), 'data')
train_dir = os.path.join(data_dir, 'train')
validation_dir = os.path.join(data_dir, 'validation')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
train_gorillas_dir = os.path.join(train_dir, 'gorillas')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
validation_gorillas_dir = os.path.join(validation_dir, 'gorillas')

# Model Folder
model_dir = os.path.join(os.path.dirname('.'), 'model')
model_file = os.path.join(model_dir, 'model.h5')
weights_file = os.path.join(model_dir, 'weights.h5')

# Summary
num_cats_tr = len(fnmatch.filter(os.listdir(train_cats_dir), '*.jpg'))
num_dogs_tr = len(fnmatch.filter(os.listdir(train_dogs_dir), '*.jpg'))
num_gorillas_tr = len(fnmatch.filter(os.listdir(train_gorillas_dir), '*.jpg'))
num_cats_val = len(fnmatch.filter(os.listdir(validation_cats_dir), '*.jpg'))
num_dogs_val = len(fnmatch.filter(os.listdir(validation_dogs_dir), '*.jpg'))
num_gorillas_val = len(fnmatch.filter(os.listdir(validation_gorillas_dir), '*.jpg'))
total_train = num_cats_tr + num_dogs_tr + num_gorillas_tr
total_val = num_cats_val + num_dogs_val + num_gorillas_val
print('Total training cat images:', num_cats_tr)
print('Total training dog images:', num_dogs_tr)
print('Total training gorillas images:', num_gorillas_tr)
print('Total validation cat images:', num_cats_val)
print('Total validation dog images:', num_dogs_val)
print('Total validation gorillas images:', num_gorillas_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

# Basic Params
batch_size = 256
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150
total_classes = 3  # cats, dogs and gorillas
learning_rate = 0.000001

# Get Images
train_image_generator = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    zoom_range=0.2,
    shear_range=0.2)

validation_image_generator = ImageDataGenerator(
    rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=False,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical')

val_data_gen = validation_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=validation_dir,
    shuffle=False,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical')

# Visualization # Might remove this section
# sample_training_images, _ = next(train_data_gen)


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
# def plot_images(images_arr):
#    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
#    axes = axes.flatten()
#    for img, ax in zip(images_arr, axes):
#        ax.imshow(img)
#        ax.axis('off')
#    plt.tight_layout()
#    plt.show()


# lot_images(sample_training_images[:5])

# Generate the model
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(),
    Dropout(0.5),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.5),
    Flatten(),
    Dense(1024, activation='relu'),
    Dense(total_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=optimizers.Adam(lr=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print a summary of the model
model.summary()

# Train the model
history = model.fit_generator(
    train_data_gen,
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
