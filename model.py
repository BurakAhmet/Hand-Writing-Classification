import tensorflow as tf
import numpy as np

# Load the pre prepared data from TensofFlow
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Define a CNN model with data augmentation
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    # Dropout to avoid from the overfitting
    tf.keras.layers.Dropout(0.5),

    # There are 10 different (0-9) output. 
    # So use 10 units dense layer with softmax activation 
    tf.keras.layers.Dense(10, activation="softmax"),  
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Data Augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1
)

datagen.fit(x_train)

# Train the model with data augmentation
model.fit(datagen.flow(x_train, y_train, batch_size=128),
          steps_per_epoch=len(x_train) / 128,
          epochs=10,
          validation_data=(x_test, y_test))
