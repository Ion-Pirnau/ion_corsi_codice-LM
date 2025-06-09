import numpy as np
from Code_Jpeg.utils_method.decorator_functions import measure_time

def compute_D(N: int) -> tuple[float, float]:
    """
        Function to initialize the Transform matrix DCT D

        :param N: dimension of the matrix

        :return: Matrix D representing the DCT transformation
    """
    D = np.zeros((N, N))

    for k in range(N):
        alpha = np.sqrt(1/N) if k == 0 else np.sqrt(2/N)
        for n in range(N):
            D[k, n] = alpha * np.cos(np.pi * (2* n + 1) * k / (2 * N))
    
    return D

@measure_time
def dctTwo(f_mat: tuple[float, float]) -> tuple[float, float]:
    """
        Function to calculate the DCT-2D on the Input Signal
        Determine the coefficient. Domain's Frequency of the Bidimensional Signal

        :param f_mat: matrix. Input signal (2D) to transform

        :return: Matrix
    """
    N = f_mat.shape[0]
    D = compute_D(N)
    c_mat = D @ f_mat @ D.T

    return c_mat