import numpy as np
from Code_Jpeg.methods.computate_dct import dctTwo
from Code_Jpeg.utils_method.utils import MyUtils 
from scipy.fft import dct as lib_dct
from Code_Jpeg.utils_method.decorator_functions import measure_time

class DCT:
    """
        Class used to simulated DCT2 (Discrete Cosine Transformation 2-Dimensional) method
        
        This class implement the method studied at the University's Course
        Metodi del Calcolo Scientifio + implmente the (FFT) DCT2 of another library
    """

    def __init__(self, n_dim:int) -> None:
        """
            Constructor
            
            :param n_dim: dimension of the matrix n_dim x n_dim

            :return: None
        """
        self.N = n_dim
        self.mutils = MyUtils()
        self.choose_title = 0

    def myfunction(self, x: float, y: float) -> float:
        """
            Define a function used for the transformation
            
            :param x: value 1
            :param y: value 2

            :return: float values

        """
        #np.sign(x - 0.5) * np.sign(y - 0.5)
        #value_cal = np.sign(x - 0.5) * np.sign(y - 0.5)
        #value_cal = 1
        value_cal = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) 
        return value_cal
    
    def build_matrix_center_point(self) -> tuple[float, float]:
        """
            Function to initialize a matrix with center point

            :return: Quadratic Matrix 
        """
        f_mat = np.zeros((self.N, self.N))

        for j in range(self.N):
            for ell in range(self.N):
                x_val = (2 * j + 1) / (2 * self.N)
                y_val = (2* ell + 1) / (2 * self.N)

                f_mat[j, ell] = self.myfunction(x_val, y_val)
        
        return f_mat
    
    def get_matrix_test(self) -> tuple[float,float]:
        """
            Function to get the Matrix test
            
            :param None:

            :return: Matrix Vector
        """
        return self.mutils.define_vector_test()
    
    def apply_dct2_personalized(self, mat: tuple[float, float]) -> (tuple)[tuple[float, float],float]:
        """
            Execute DCT-2D personalized

            :param mat: matrix

            :return: None
        """
        c_mat, time_calc = dctTwo(mat)
        self.choose_title = 1

        return c_mat, time_calc


    def my_plot3D(self, data_matrix:tuple[float, float]) -> None:
        """
            Plot on 3D Graph

            :param data_matrix: data to plot
            
            :return: None
        """
        titles = ["Original bidemensional array f", "DCT of the original bidimensional f (Personalized Method)", 
                  "DCT of the original bidimensional f (From Library)"]
        self.mutils.plot_bar3d(p_mat=data_matrix, title=titles[self.choose_title])


    def my_plot_times(self, times_vect:tuple[float]) -> None:
        """
            Function to plot the times on graph

            :param times_vect: vector with methods' times

            :return: None
        """
        self.mutils.plot_time_execution_methods(time_data=self.mutils.build_dictionary_dct(times_set=times_vect))


    @measure_time
    def apply_dct2_lib(self, mat: tuple[float, float]) -> tuple[float, float]:
        """
            Execute DCT-2D from SciPy library

            :param mat: matrix

            :return: None
        """
        self.choose_title = 2
        norm_type = ["ortho", None]
        #MonoDimnesional
        mono_dim = lib_dct(mat, norm=norm_type[0])

        two_dim = lib_dct(lib_dct(mat, norm=norm_type[0], axis=0), axis=1, norm=norm_type[0])
        
        return two_dim

    def set_N_var(self, n:int) -> None:
        """
            Function to set the N variable

            :param None:

            :return: None
        """
        self.N = n
            