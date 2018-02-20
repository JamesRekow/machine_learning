"""
Module to create a test nn binary classifier. For learning purposes.
"""

# Imports.
import csv
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation


def load_data(data_file=None):
    """
    Function to load binary data from csv file.
    """

    # Set defaults.
    if data_file is None:
        data_file = 'bin_seq.csv'

    bin_array = np.genfromtxt(data_file, delimiter=',')

    return(bin_array)
# End load data.

def partition_data(data, test_ratio=None):
    """
    Function to partition input data into training data and test data. Takes
    an np array of any shape as input, and assumes that the array is a list
    of input vectors (so that the final element of the shape vector is the
    number of input vectors).
    """

    # Set defaults.
    if test_ratio is None:
        test_ratio = 0.2

    num_inputs = len(bin_array)

    num_test_ix = int(np.floor(test_ratio * num_inputs))

    test_ix = np.random.choice(range(num_inputs), size=num_test_ix,
        replace=False)

    training_ix = np.array([ix for ix in range(num_inputs) if
        (not ix in test_ix)])

    training_array = bin_array[training_ix]
    test_array = bin_array[test_ix]

    data_dict = {'test_data': test_array, 'training_data': training_array}

    return(data_dict)
# End partition data.

def create_nn():
    """
    Function to create nn model to classify input data.
    """

    model = Sequential()
    model.add(Dense(1, activation='relu', input_dim=1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
        metrics=['accuracy'])

    return(model)
# End create_nn.

# Implement CLI.
if __name__ == '__main__':
    bin_array = load_data()
    data_dict = partition_data(data=bin_array, test_ratio=0.1)
    training_data = data_dict['training_data']
    test_data = data_dict['test_data']

    labels = training_data

    model = create_nn()

    model.fit(training_data, labels, epochs=100, batch_size=100)

    score = model.evaluate(test_data, test_data, batch_size=10)

    print(score)
