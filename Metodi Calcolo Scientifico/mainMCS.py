from Code.General_Method.mLinearSystem import LinearSystemAxb as ls_axb
from Code.MyUtils.dec_functions import get_path_general, check_path_exists, get_method_active_values
import json

if __name__ == '__main__':

    path = get_path_general(filename="config.json")
    check_path_exists(pathname=path)

    path2 = get_path_general()
    check_path_exists(pathname=path2)

    with open(path, "r") as f:
        config = json.load(f)
    
    my_linearsystem = ls_axb(tollerance=config["tollerance"][2], max_iteration=config["max_iter"], 
                             matrix_name_file=config["matrix_name"], time_unit=config["time_unit"][0])

    my_linearsystem.read_matrix()
    
    my_linearsystem.initialize_linear_system()

    act_list = get_method_active_values(active_values=config["active_methods"])

    if act_list[1]:
        my_linearsystem.solve_jacobi_system(omega_value=config["omega_v_jacobi"], relax_value=act_list[1])
    if act_list[0]:
        my_linearsystem.solve_jacobi_system()

    
    if act_list[3]:
        my_linearsystem.solve_gaub_seidel_system(omega_value=config["omega_v_gaub"], relax_value=act_list[3])
    if act_list[2]:
        my_linearsystem.solve_gaub_seidel_system()

    if act_list[4]:
        my_linearsystem.solve_gradient_system()

    if act_list[5]:
        my_linearsystem.solve_conjugate_gradient_system()

    if act_list[0] or act_list[1] or act_list[2] or act_list[3] or act_list[4] or act_list[5]:
        my_linearsystem.show_graph()
    my_linearsystem.print_msg_resume(pathfile_txt=path2)

    