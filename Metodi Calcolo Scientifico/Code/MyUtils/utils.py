import numpy as np
import scipy as scp
from scipy.io import mmread
from scipy.sparse import spmatrix
from Code.MyUtils.dec_functions import measure_time
from Code.MyUtils.dec_functions import plot_grap_errors, get_path, check_path_exists, plot_graph_useful_info, plot_time_execution_methods


class UtilityMethods:
    """
        Class that implements different methods to solve a Linear System, methods to create,
        calculate and other
    """

    def __init__(self, n:int=0, rnum: int=0, rhigh: int=0, rlow: int=0) -> None:
        """
        Class Constructor

        :param n: Integer variable, specify the dimension of the Quadratic Matrix NxN
        :param rnum: Number to generate number from 0 to rnum
        :param rhigh: Range to highest number to generate
        :param rlow: Range to lowest number to generate
        """
        self.n = n
        self.rnum = rnum
        self.rlow = rlow
        self.rhigh = rhigh
        self.errors_Jac = []
        self.erros_RJac = []
        self.errors_Gaub = []
        self.errors_RGaub = []
        self.errors_Grad = []
        self.errors_CGrad = []
        self.convergence_method = [[True, True], [True, True], [True, True]]

    def set_n(self, n: int=0) -> None:
        """
        Function to set the n var

        :param n: Integer variable, specify the dimension of the Quadratic Matrix NxN
        
        :return: None
        """
        self.n = n

    def get_n(self) -> int:
        """
        Function to get the n var
        
        :return: None
        """
        return self.n

    def generate_ones_vector(self) -> list:
        """
        Generate vector with only ONEs. The dimension of the vector has been
        specified int the Constructor

        :return: List of value equal to 1
        """
        vect = np.ones(self.n)

        list_vect = vect.tolist()

        return list_vect

    def generate_range_vector_random(self) -> list:
        """
        Generate vector with range values. The dimension of the vector has been
        specified in the Constructor (same dimension as the Matrix)

        :return: List of int value
        """
        vect_b = np.random.randint(low=0, high=20, size=self.n)
        vect_b = vect_b + self.generate_ones_vector()
        return vect_b.tolist()

    def generate_matrix(self, values : list[float], type_matrix:str='u') -> list:
        """
        Generate a Triangular Matrix based on the type_matrix variable.

        :param values: Tuple Floating values to insert into the matrix. u - Upper, l - Lower
        :param type_matrix: String value, define the type of matrix

        :return: List of Floating values
        """
        index_ul = tuple()
        if type_matrix == 'u':
            index_ul = np.triu_indices(self.n)
        elif type_matrix == 'l':
            index_ul = np.tril_indices(self.n)
        mat_zeros = np.zeros((self.n, self.n))

        if values:
            print(values)
        else:
            a = np.random.random((1, len(index_ul[0]))) + 1
            mat_zeros[index_ul] = a[0]

        return mat_zeros.tolist()


    def generate_vector_B(self, matrix_a : list, vector_x : list) -> list:
        """
        Generate vector B by solving the Ax = b equation

        :param matrix_a: Triangular Matrix
        :param vector_x: Unit Vector

        :return: List from the Product between matrix_a and vector_x
        """
        mat = np.array(matrix_a)
        vect = np.array(vector_x)

        vector_b = mat @ vect

        #print(vector_b.tolist())

        return vector_b.tolist()

    def my_solve(self, matrix_a : list, vector_b : list) -> list:
        """
        Generate the approximate x

        :param matrix_a: Triangular Matrix
        :param vector_b: vector B calculated

        :return: List, approximate x
        """
        mat = np.array(matrix_a)
        vect = np.array(vector_b)

        diagonal_mat = np.linalg.diagonal(mat)
        determinant_mat = np.prod(diagonal_mat)

        if determinant_mat > 10**-12:
            x_approx = np.zeros(self.n)
            x_approx[self.n-1] = vect[self.n-1] / mat[self.n-1, self.n-1]
            start_index = self.n-2
            for i in range(start_index, -1, -1):
                x_approx[i] = (vect[i] - (np.dot(mat[i], x_approx))) / mat[i,i]

            return x_approx.tolist()
        else:
            raise ValueError("Determinant Lower. Change the values in the Matrix")

    def calculate_error_relative(self, x_approx : list, x_right : list) -> float:
        """
        Calculate the relative error

        :param x_approx: approximated solution
        :param x_right: right solution

        :return: Float value of Relative Error
        """
        vect_a = np.array(x_approx)
        vect_exact = np.array(x_right)
    
        return np.linalg.norm(vect_exact-vect_a, np.inf)/np.linalg.norm(vect_exact, np.inf)

    def calculate_error_conditional(self, matrix : list) -> float:
        """
        Calculate conditional error from eigenvalues of the matrix

        :param matrix: quadratic matrix

        :return: Float error
        """
        mat = np.array(matrix)
        eigvalues, eigvectors = np.linalg.eig(mat)

        max_eigvalue = np.amax(eigvalues)
        min_eigvalue = np.amin(eigvalues)

        return  np.abs(max_eigvalue)/np.abs(min_eigvalue)

    def rotate_matrix(self, matrix : list) -> list:
        """
        Rotate the matrix of 90°

        :param matrix: quadratic matrix

        :return: List
        """
        mat = np.array(matrix)

        mat = np.rot90(mat)
        mat = np.rot90(mat)

        return mat.tolist()

    def reverse_vector(self, vect : list) -> list:
        """
        Reverse the vector's elements order

        :param vect: vector
        
        :return: List
        """
        vect_x = np.array(vect)
        vect_x = np.flipud(vect_x)

        return vect_x.tolist()


    def print_general_vect(self, vect: list, text: str) -> None:
        """
        Method to print matrix
        
        :param vect: define the matrix
        :param text: define the name of the vect representation
        
        :return: None
        """
        my_matrix = np.array(vect)
        print(text)
        print(my_matrix)

    def create_diagonal_matrix(self, choose_value: int) -> list:
        """
        Create a Matrix with only diagonal value
        
        :param choose_value: int value to add at the diagonal matrix
        
        :return: List
        """
        m_matrix = np.diag(np.full(self.n, choose_value))

        return m_matrix.tolist()

    def create_random_matrix(self) -> list:
        """
        Create a random Matrix NxN
        
        :return: List
        """
        mat = np.random.randint(self.rlow, self.rhigh, (self.n, self.n))
        diag_mat = np.diag(np.diag(mat))

        mat = mat-diag_mat

        intero_random = np.random.randint(self.rnum)
        final_diag = self.create_diagonal_matrix(intero_random)

        mat = mat + final_diag

        det_calc = scp.linalg.det(mat)
        if det_calc < 1e-12:
            raise ValueError("Matrix Not Invertible!")

        "Section for making the Matrix Positive and Symmetric"
        "Matrix Symmetric"
        mat = (mat + mat.T) / 2

        return mat.tolist()

    def is_diagonal_dominant(self, matrix: list) -> bool:
        """
        Determine if the matrix is diagonal dominant on rows
        
        :param matrix: list to operate on
        
        :return: True:- is Diagonal Dominant, False:- is not Diagonal Dominant
        """
        mat = np.array(matrix)
        abs_mat = np.abs(mat)
        D = np.diag(abs_mat)
        S = np.sum(abs_mat, axis=1) - D
        bool_value = np.all(D > S)

        if bool_value:
            return True
        else:
            return False
        
    def is_symmetric_positive(self, matrix: list) -> bool:
        """
        Determine if a matrix is symmetric and positive (SDP)

        :param matrix: matrix to determine if it is symmetric and positive
        
        :return: True:- is SDP, False:- is not SDP
        """

        mat = np.array(matrix)
        #print("Matrice Letta:")
        #print(mat)

        if not np.allclose(mat, mat.T):
            return False

        eigenvalues = np.linalg.eigvals(mat)
        if np.all(eigenvalues > 0):
            return True
        else:
            return False
    
    def is_matrix_valid_to_execute_method(self, typem: str, cond_suff: bool, matrix: list) -> bool:
        """
        Determine if matrix can be execute on methods if one of the conditions are valid.
        Conditions:
        1 - DIAGONAL DOMINANT,
        2 - SYMMETRIC AND POSITIVE

        :param typem: string to define the method for which condition should be applied
        :param cond_suff: boolean value to validate the condition of Diagonal Dominant Matrix
        :param matrix: matrix to do the validation

        :return: boolean value
        """

        val_one = True
        val_two = True
        mytype_m = ['j', 'jor', 'g', 'sor']

        if cond_suff:
            if typem in mytype_m:
                if self.is_diagonal_dominant(matrix):
                    print("DOMINANT")
                else: 
                    print("NOT DOMINANT")
                    val_one = False

            if self.is_symmetric_positive(matrix):
                print("IS SYMMETRIC AND POSITIVE")
            else: 
                print("IS NOT SYMMETRIC AND POSITIVE")
                val_two = False
            
        if typem == 'grad' and (not val_two):
            return val_two
        else:
            return val_one or val_two

    def print_result_calculated(self, title: str, xguess: list, residual_error: float, iter_count: int,
                                time_calc: str, error_relative: float) -> list:
        """
        Function to print the result calculate by every methods that has solve a Linear System Ax=b
        
        :param title : define the Method that has solve the Linear System
        :param xguess : the solution calculated
        :param residual_error: the residual calculated
        :param iter_count: number of iteration
        :param time_calc: Time (ms) used to calculate and find the solution
        :param error_relative: relative error between correct solution and approximanted solution
        
        :return: list
        """
        list_comments = list()

        list_comments.append("------------------")
        list_comments.append("------------------")
        list_comments.append(f"Type: {title}")
        list_comments.append("------------------")
        #print(f"Solution: {xguess}")
        list_comments.append(f"Scaled Residual: {residual_error}")
        list_comments.append(f"N-Iteration: {iter_count}")
        list_comments.append(f"Time: {time_calc}")
        list_comments.append(f"Relative Error: {error_relative}")
        list_comments.append("------------------")
        list_comments.append("\n")

        return list_comments

    
    def initialize_plot_show(self, time_a:list=(), time_unit:str='ms') -> None:
        """
        Method to initialize the dictionary and show the graph
        
        :param time_a: list with time-string of each method
        :param time_unit: str to define the Time Unit to display

        :return: None
        """
        error_dict = {
            'Jacobi': self.errors_Jac,
            'Jacobi-Relaxation': self.erros_RJac,
            'Gaub-Seidel': self.errors_Gaub,
            'Gaub-Seidel-Relaxation': self.errors_RGaub,
            'Gradient': self.errors_Grad,
            'Conjugate-Gradient': self.errors_CGrad
        }

        convergence_dict = {
            'Jacobi': self.convergence_method[0][0],
            'Jacobi-Relaxation': self.convergence_method[0][1],
            'Gaub-Seidel': self.convergence_method[1][0],
            'Gaub-Seidel-Relaxation': self.convergence_method[1][1],
            'Gradient': self.convergence_method[2][0],
            'Conjugate-Gradient': self.convergence_method[2][1]
        }

        time_dic = {
            'Jacobi': time_a[0],
            'Jacobi-Relaxation': time_a[1],
            'Gaub-Seidel': time_a[2],
            'Gaub-Seidel-Relaxation': time_a[3],
            'Gradient': time_a[4],
            'Conjugate-Gradient': time_a[5]
        }

        index_plot = plot_grap_errors(errors_calc=error_dict)
        plot_graph_useful_info(m_convergence=convergence_dict, index=index_plot)
        plot_time_execution_methods(time_data=time_dic, target_unit=time_unit, index=index_plot)


    def load_matrix_from_mtx(self, file_name: str="", to_dense: bool = False):
        """
        Load the matrix from the file .mtx (Matrix Market), the matrixes are stored in data_ion's folder
        
        :param file_name: name of the file .mtx
        :param to_dense: if True, retunr matrix as NumPy array
                         if False, return matrix as Sparse Matrix (sciPy) 
        
        :return: Matrix as NumPy or Sparse Matrix
        """
        path_dir = get_path()
        path_complete = path_dir+file_name
        check_path_exists(pathname=path_complete)
        #print(f"Path: {path_complete}")
        matrix = mmread(path_complete)

        if not isinstance(matrix, spmatrix):
            raise ValueError("The Matrix read it is not a VALID Sparse Matrix!")
       
        return matrix.toarray() if to_dense else matrix

    "-  -   -   -   -   -   -   -   -   -   -   -   -   -   -   "
    "-  -   -   -   -   -   -   -   -   -   -   -   -   -   -   "
    "-  -   -   -   -   -   -   -   -   -   -   -   -   -   -   "
    "-  -   -   -   -   -   -   -   -   -   -   -   -   -   -   "
    "-  -   -   -   -   -   -   -   -   -   -   -   -   -   -   "
    "-  -   -   -   -   -   -   -   -   -   -   -   -   -   -   "
    "-  -   -   -   -   -   -   -   -   -   -   -   -   -   -   "
    "-  -   -   -   -   -   -   -   -   -   -   -   -   -   -   "
    "-  -   -   -   -   -   -   -   -   -   -   -   -   -   -   "
    "-  -   -   -   -   -   -   -   -   -   -   -   -   -   -   "
    "From this section: Methods to Solve a Linear System Ax = b"
    "-  -   -   -   -   -   -   -   -   -   -   -   -   -   -   "
    "-  -   -   -   -   -   -   -   -   -   -   -   -   -   -   "
    "-  -   -   -   -   -   -   -   -   -   -   -   -   -   -   "
    @measure_time
    def jacobi_iterative_solver(self, matrix_start: list, b: list, tol: float, max_iter: int, xguess: list,
                                w_value: float, is_relax: bool=False) -> (
            tuple)[list, float, int]:
        """
        Solve the linear system with jacobi method
        
        :param matrix_start: matrix A
        :param b: B vector
        :param tol: tollerance
        :param max_iter: iteration for calculation
        :param xguess: starting solution
        :param is_relax: bool value for using the relaxation method
        :param w_value: value for w var relaxation
        
        :return: Solution of x calculated, Error, Iteration Counter
        """
        a_mat = np.array(matrix_start, dtype=float)
        b_arr = np.array(b, dtype=float)
        iter_counter = 0
        x_new = np.zeros(self.n, dtype=float)
        x_approx = np.zeros(self.n, dtype=float)
        my_error = float('inf')

        if not is_relax:
            w_relax = 1.0
        else:
            if 0.0 < w_value <= 1.0:
                w_relax = w_value
            else:
                raise ValueError("w - value should be in this range (0, 1]")

        while my_error > tol and iter_counter < max_iter:
            for i in range (0, self.n):
                sum_ax = sum(a_mat[i,j] * x_approx[j] for j in range(self.n) if j != i)
                x_new_i = (b_arr[i] - sum_ax) / a_mat[i][i]

                x_new[i] = (w_relax * x_new_i) + ((1.0-w_relax) * x_new[i])
            

            my_error = (np.linalg.norm((a_mat @ x_new) - b_arr) /
                        np.linalg.norm(b_arr))

            x_approx = x_new.copy()
           #my_error = self.calculate_error_relative(x_new.tolist(), xguess)
            if not is_relax:
                self.errors_Jac.append(my_error)
            else:
                self.erros_RJac.append(my_error)

            iter_counter += 1

        if iter_counter > max_iter:
            if not is_relax:
                self.convergence_method[0][0] = False
            else:
                self.convergence_method[0][1] = False

        return x_approx.tolist(), my_error, iter_counter
    

    @measure_time
    def gaub_seidel_iterative_solver(self, matrix_start: list, b: list, tol: float, max_iter: int, xguess: list,
                                w_value: float, is_relax: bool=False) -> (
            tuple)[list, float, int]:
        """
        Solve the linear system with Gaud-Seidel method
        
        :param matrix_start: matrix A
        :param b: B vector
        :param tol: tollerance
        :param max_iter: number of iteration for calculation
        :param xguess: starting solution
        :param is_relax: bool value for using the relaxation method
        :param w_value: value for w var relaxation
        
        :return: Solution of x calculated, Error, Iteration Counter
        """
        a_mat = np.array(matrix_start, dtype=float)
        iter_counter = 0
        x_new = np.zeros(self.n, dtype=float)
        x_approx = np.zeros(self.n, dtype=float)
        my_error = float('inf')
        b_arr = np.array(b, dtype=float)

        if not is_relax:
            w_relax = 1.0
        else:
            if 0.0 < w_value < 2.0:
                w_relax = w_value
            else:
                raise ValueError("w - value should be in this range (0, 2)")

        
        while my_error > tol and iter_counter < max_iter:
            x_new = np.copy(x_approx)
            for i in range (0, self.n):
                sum_ax_less = sum(a_mat[i,j] * x_new[j] for j in range(self.n) if j < i)
                sum_ax_greater = sum(a_mat[i,j] * x_approx[j] for j in range(self.n) if j > i)
                x_new_i = (b_arr[i] - sum_ax_less - sum_ax_greater)/a_mat[i][i]
                #x_new[i] = w_relax * (b_arr[i] - sum_ax_less - sum_ax_greater)/a_mat[i][i]
                #x_new[i] += (1.0-w_relax) * x_new[i]
                x_new[i] = (w_relax * x_new_i) + ((1-w_relax) * x_approx[i])
                
            
            my_error = (np.linalg.norm(((a_mat @ x_new) - b_arr), ord=np.inf) /
                        np.linalg.norm(b_arr, ord=np.inf))
            #my_error = self.calculate_error_relative(x_new.tolist(), xguess)
            x_approx = x_new.copy()
            if not is_relax:
                self.errors_Gaub.append(my_error)
            else:
                self.errors_RGaub.append(my_error)

           
            iter_counter += 1

        if iter_counter > max_iter:
            if not is_relax:
                self.convergence_method[1][0] = False
            else:
                self.convergence_method[1][1] = False

        return x_approx.tolist(), my_error, iter_counter
    

    @measure_time
    def gradient_descent_solver(self, matrix_start: list, b: list, tol: float, max_iter: int, xguess: list) -> (
            tuple)[list, float, int]:
        """
        Solve the linear system with Gradient Descent Method
        
        :param matrix_start: matrix symmetric and positive
        :param b: vector b from Ax = b
        :param tol: tollerance
        :param max_iter: number of iteration for calculation
        :param xguess: starting solution
        
        :return: Solution of x calculated, Error, Iteration Counter
        """
        x_approx = np.zeros(self.n, dtype=float)
        my_error = float('inf')
        matrix = np.array(matrix_start)
        iter_counter = 0
        vect_b = np.array(b)

        while my_error > tol and iter_counter < max_iter:
            residual = vect_b - (matrix @ x_approx)
            alpha = (residual.T @ residual) / (residual.T @ (matrix @ residual))
            x_new = x_approx + alpha*residual

            my_error = (np.linalg.norm((matrix @ x_new) - vect_b) /
                        np.linalg.norm(vect_b))
            
            x_approx = x_new.copy()
            
            self.errors_Grad.append(my_error)

            iter_counter += 1

        if iter_counter > max_iter:
                self.convergence_method[2][0] = False
        
        return x_approx.tolist(), my_error, iter_counter
    

    @measure_time
    def conjugate_gradient_descent_solver(self, matrix_start: list, b: list, tol: float, max_iter: int, xguess: list) -> (
            tuple)[list, float, int]:
        """
        Solve the linear system with Conjugate Gradient Method
        
        :param matrix_start: matrix symmetric and positive
        :param b: vector b from Ax = b
        :param tol: tollerance
        :param max_iter: number of iteration for calculation
        :param xguess: starting solution
        
        :return: Solution of x calculated, Error, Iteration Counter
        """
        x_approx = np.zeros(self.n, dtype=float)
        my_error = float('inf')
        iter_counter = 0
        matrix = np.array(matrix_start)
        vect_b = np.array(b)
        residual = vect_b - (matrix @ x_approx)
        p = residual.copy()
        rs_old = residual.T @ residual

        while my_error > tol and iter_counter < max_iter:
            Ap = matrix @ p
            alpha = rs_old / (p.T @ Ap)
        
            x_approx = x_approx + alpha*p

            residual = residual - alpha*Ap

            rs_new = residual.T @ residual

            p = residual + (rs_new / rs_old) * p
            rs_old = rs_new

            my_error = (np.linalg.norm((matrix @ x_approx) - vect_b) /
                        np.linalg.norm(vect_b))
            if iter_counter > max_iter:
                self.convergence_method[2][1] = False
                break
            
            self.errors_CGrad.append(my_error)

            iter_counter += 1

        return x_approx.tolist(), my_error, iter_counter
    
