import os
import numpy as np

def PrintOutput(weights, filename="y_test.csv"):

    with open(filename, 'w+') as fp:

        for i in range(len(weights)):
            line = str(weights[i]) + "\n"
            fp.write(line)
