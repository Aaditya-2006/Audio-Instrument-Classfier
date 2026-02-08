import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape = (128,), num_classes = 20):
    model = models.Sequential()

    model.add(layers.Dense(1024, input_shape = input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(512))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.4))

    model.add(layers.Dense(num_classes, activation = 'sigmoid'))

    return model