"""
Module to load MNIST dataset from Keras package, then create and train a nn
on the loaded MNIST data before returning accuracy on test samples.
"""

# Imports
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

def create_nn():
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(784, )))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
        metrics=['accuracy'])

    return(model)
# End create_nn.

# Implement CLI.
if __name__ == '__main__':

    # Load MNIST data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = x_train / 255
    x_test = x_test / 255

    n_classes = 10
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)

    model = create_nn()

    model.fit(x_train, y_train, epochs=20, batch_size=100)

    score = model.evaluate(x_test, y_test, batch_size=100)

    print(score)
