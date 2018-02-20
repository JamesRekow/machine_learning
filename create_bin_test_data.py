"""
Script to create a pseudorandom sequence of 0s and 1s to train and test a
binary NN classifier, and save this sequence to a csv file in the same directory
as this script.
"""

# Imports.
import random
import csv

def create_bin_seq(length=None):
    """
    Function to create a pseudorandom tuple of 0s and 1s to use as input data
    for the NN.
    """

    # Set defaults.
    if length is None:
        length = 100

    # Create random sequence.
    bin_seq = tuple([random.randint(0, 1) for _ in range(length)])

    return(bin_seq)
# End create_bin_seq.

def create_bin_csv(data_len=None):
    """
    Function to create a sequence of random 0s and 1s and save it to a csv file
    in the same directory as this one.
    """

    # Create binary sequence.
    bin_seq = create_bin_seq(length=data_len)

    # Write binary sequence to csv file.
    with open('bin_seq.csv', 'w', newline='') as csvfile:
        # Create csv writer object.
        csv_writer = csv.writer(csvfile, delimiter=' ')

        # Write each number to a separate line in the csv file.
        for num in bin_seq:
            csv_writer.writerow(str(num))

# Implement CLI.
if __name__ == '__main__':
    create_bin_csv()
