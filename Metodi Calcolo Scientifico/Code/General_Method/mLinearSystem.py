from Code.MyUtils.utils import UtilityMethods as utm
import numpy as np

class LinearSystemAxb:
    """
        Class that use different methods to solve a Linear System Ax = b
    """

    def __init__(self, type_mat: str="", dim_mat: int=2, rnum: int=0, rlow: int=0, rhigh: int=0,
                 nec_suff_cond: bool=True, tollerance: float=1e-10, max_iteration: int=1, matrix_name_file: str="", 
                 time_unit:str="") -> None:
        """
        Constructor

        :param type_mat: define the type of the Matrix. l = Triangular Inferior; u = Triangular Superior
        :param dim_mat: dimension of the Quadratic Matrix
        :param rnum: Number to generate number from 0 to rnum - on diagonal
        :param rhigh: Range to highest number to generate - no diagonal position
        :param rlow: Range to lowest number to generate - no diagonal position
        :param nec_suff_cond: boolean value to validate the condition of Diagonal Dominant Matrix
        :param tollerance: tollerance for iterative method
        :param max_iteration: max number of iteration for iterative method
        :param matrix_name_file: name of the file .mtx
        :param time_unit: str to define the Time Unit used to display on the graph

        """
        self.type_mat = type_mat
        self.dim_mat = dim_mat
        self.rnum = rnum
        self.rlow = rlow
        self.rhigh = rhigh
        if not matrix_name_file:
            self.myutils = utm(self.dim_mat, self.rnum, self.rhigh, self.rlow)
        self.mat_vect = None
        self.b_vect = None
        self.xguess_vect = None
        self.nec_suff_cond = nec_suff_cond
        self.tollerance = tollerance
        self.max_iteration = max_iteration
        self.matrix_name_file = matrix_name_file
        self.array_time = [None] * 6
        self.time_unit = time_unit
        self.msg_resume = []

    def solve_linear_system(self) -> None:
        """
        Solve the linear system with method Ax = b.

        :return: None
        """

        mat = self.myutils.generate_matrix([], type_matrix=self.type_mat)
        if not self.type_mat == 'l':
            mat = self.myutils.rotate_matrix(matrix=mat)
        vect = self.myutils.generate_ones_vector()
        vect_b = self.myutils.generate_vector_B(matrix_a=mat, vector_x=vect)
        vect_h = self.myutils.my_solve(matrix_a=mat, vector_b=vect_b)

        if self.type_mat == 'l':
            vect_h = self.myutils.reverse_vector(vect=vect_h)

        print(f"Relative's Error: {self.myutils.calculate_error_relative(x_pre=vect, x_approx=vect_h)}")
        print(f"Cond: {self.myutils.calculate_error_conditional(matrix=mat)}")

    def initialize_linear_system(self) -> None:
        """
        Method to initialize matrix, b, and xguess

        :return: None
        """
        if not self.matrix_name_file:
            self.mat_vect = self.myutils.create_random_matrix()
            #self.myutils.print_general_vect(self.mat_vect, text="Matrix Diagonal Dominant")
        else:
            self.mat_vect = self.read_matrix()

        self.xguess_vect = self.myutils.generate_ones_vector()
        #self.myutils.print_general_vect(self.xguess_vect, text="Starting Solution (Guess)")

        #self.b_vect = self.myutils.generate_range_vector_random()
        self.b_vect = self.myutils.generate_vector_B(matrix_a=self.mat_vect, vector_x=self.xguess_vect)
        #self.myutils.print_general_vect(self.b_vect, text="B")

    def solve_jacobi_system(self, omega_value: float=1.0, relax_value: bool=False) -> None:
        """
        Iterative Method with Jacobi approach

        :param omega_value: w's value for relaxation
        :param relax_value: bool to activate the Jacobi relaxation

        :return: None
        """
        if relax_value:
            my_title = "Jacobi Relaxation"
            type_m = 'jor'
        else:
            my_title = "Jacobi"
            type_m = 'j'

        if self.myutils.is_matrix_valid_to_execute_method(typem=type_m, cond_suff=self.nec_suff_cond, matrix=self.mat_vect):
            x, error_calc, iter_calc, calc_time = self.myutils.jacobi_iterative_solver(self.mat_vect, self.b_vect, self.tollerance,
                                                                         self.max_iteration, self.xguess_vect,
                                                                         w_value=omega_value, is_relax=relax_value)
            if type_m == 'jor':
                self.array_time[1] = calc_time
            else:
                self.array_time[0] = calc_time
            
            self.msg_resume += self.myutils.print_result_calculated(title=my_title, xguess=x, 
                                             residual_error=error_calc, iter_count=iter_calc, time_calc=calc_time, 
                                             error_relative=self.myutils.calculate_error_relative(x, self.xguess_vect))
        else:
            self.msg_resume.append("Jacobi's Method does not satisfy the condition for DOMINANT MATRIX OR SYMMETRIC AND POSITIVE")
        

    def solve_gaub_seidel_system(self, omega_value: float=1.0, relax_value: bool=False) -> None:
        """
        Iterative Method with Gaub-Seidel approach

        :param tollerance: tollerance for iterative method
        :param max_iteration: max number of iteration for iterative method
        :param omega_value: w's value for relaxation
        :param relax_value: bool to activate the Jacobi relaxation

        :return: None
        """
        if relax_value:
            my_title = "Gaub-Seidel Relaxation"
            type_m = 'sor'
        else:
            my_title = "Gaub-Seidel"
            type_m = 'g'
        

        if self.myutils.is_matrix_valid_to_execute_method(typem=type_m, cond_suff=self.nec_suff_cond, matrix=self.mat_vect):
            x, error_calc, iter_calc, calc_time = self.myutils.gaub_seidel_iterative_solver(self.mat_vect, self.b_vect, self.tollerance,
                                                                         self.max_iteration, self.xguess_vect,
                                                                         w_value=omega_value, is_relax=relax_value)
            if type_m == 'sor':
                self.array_time[3] = calc_time
            else:
                self.array_time[2] = calc_time
            
            self.msg_resume += self.myutils.print_result_calculated(title=my_title, xguess=x, 
                                             residual_error=error_calc, iter_count=iter_calc, time_calc=calc_time, 
                                             error_relative=self.myutils.calculate_error_relative(x, self.xguess_vect))
        else:
            self.msg_resume.append("Gaud-Siedel's Method does not satisfy the condition for DOMINANT MATRIX OR SYMMETRIC AND POSITIVE")
        
        

    def solve_gradient_system(self) -> None:
        """
        Iterative Method No-Stationary with Gradient approach

        :return: None
        """

        if self.myutils.is_matrix_valid_to_execute_method(typem='grad', cond_suff=self.nec_suff_cond, matrix=self.mat_vect):
            x, error_calc, iter_calc, calc_time = self.myutils.gradient_descent_solver(self.mat_vect, self.b_vect, self.tollerance,
                                                                         self.max_iteration, self.xguess_vect)
            self.msg_resume += self.myutils.print_result_calculated(title="Gradient Descent Method", xguess=x, 
                                             residual_error=error_calc, iter_count=iter_calc, time_calc=calc_time, 
                                             error_relative=self.myutils.calculate_error_relative(x, self.xguess_vect))
        else:
            self.msg_resume.append("Gradient's Method does not satisfy the condition for SYMMETRIC AND POSITIVE")

        self.array_time[4] = calc_time
        
        
        
    def solve_conjugate_gradient_system(self) -> None:
        """
        Iterative Method No-Stationary with Conjugate Gradient approach

        :return: None
        """
        if self.myutils.is_matrix_valid_to_execute_method(typem='grad', cond_suff=self.nec_suff_cond, matrix=self.mat_vect):
            x, error_calc, iter_calc, calc_time = self.myutils.conjugate_gradient_descent_solver(self.mat_vect, self.b_vect, self.tollerance,
                                                                        self.max_iteration, self.xguess_vect)
            
            self.msg_resume += self.myutils.print_result_calculated(title="Conjugate Gradient Descent Method", xguess=x, 
                                             residual_error=error_calc, iter_count=iter_calc, time_calc=calc_time, 
                                             error_relative=self.myutils.calculate_error_relative(x, self.xguess_vect))
        else:
            self.msg_resume.append("Conjugate Gradient's Method does not satisfy the condition for SYMMETRIC AND POSITIVE")

        self.array_time[5] = calc_time
        
    
    def print_msg_resume(self, pathfile_txt:str) -> None:
        """
        Print the message resume for each method

        :param pathfile_txt: absolute path of the current work folder
        
        :return: None
        """
        with open(pathfile_txt+"details.txt", "w") as ftxt:
            ftxt.write(f"Work on matrix: {self.matrix_name_file}\n")
            ftxt.write(f"\nTollerance: {self.tollerance}\n\n")
            for line in self.msg_resume:
                ftxt.write(line+"\n")

        print("\n")
        print("------------------------------------------")
        print("------------------RESUME------------------")
        print("------------------------------------------")
        print("\n")
        if self.msg_resume:
            for msg in self.msg_resume:
                print(f"- {msg};")
        else:
            print("All Method work with NO PROBLEM!")
        print("\n")


    def show_graph(self) -> None:
        """
        Function to show a matplot graph

        :return: None
        """
        self.myutils.initialize_plot_show(self.array_time,time_unit=self.time_unit)
    

    def read_matrix(self):
        """
        Function to read the Matrix from the .mtx file
        """
        self.myutils = utm()
        matrix_a = self.myutils.load_matrix_from_mtx(file_name=self.matrix_name_file, to_dense=True)

        self.myutils.set_n(n=matrix_a.shape[0])
        return matrix_a
        
