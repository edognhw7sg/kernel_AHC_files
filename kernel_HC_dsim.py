import numpy as np
from scipy.linalg import expm

def kernel_HC_dsim(data):
    return_list = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            return_list[i, j] = data[i, i] - (2 * data[i, j]) + data[j, j]

    return return_list
