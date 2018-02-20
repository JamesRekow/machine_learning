"""
Script to import binary sequence csv data, create a neural network for binary
classification of input data into classes of 0 or 1, train the nn, and test the
nn using keras.
"""

# Imports.
from keras.models import Sequential
from keras.layers import Dense
import keras
import numpy as np

# Import data.
x_train = np.random.randint(2, size=1000)
#y_train = keras.utils.to_categorical(x_train, num_classes=2)
y_train = x_train

x_test = np.random.randint(2, size=10)
#y_test = keras.utils.to_categorical(x_test, num_classes=2)
y_test = x_test

# Create nn.
model = Sequential()
model.add(Dense(1, activation='relu', input_dim=1))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Train nn.
model.fit(x_train, y_train, epochs=100, batch_size=10)

# Test nn.
score = model.evaluate(x_test, y_test, batch_size=2)

print(score)
