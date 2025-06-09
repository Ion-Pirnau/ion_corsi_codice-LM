import time
from functools import wraps
import matplotlib.pyplot as plt
import os

def measure_time(method):
    """
    Decorator for measuring the execution method's time.
    Add the time as the fourth element of the return's values

    :param method: Any type of method, that use this function as a decorator
    """

    @wraps(method)
    def timed_method(*args, **kwargs):
        start = time.time()
        result = method(*args, **kwargs)
        end = time.time()
        elapsed = (end - start)

        if elapsed < 1e-6:
            formatted_time = f"{elapsed * 1_000_000_000:.2f} ns"
        elif elapsed < 1e-3:
            formatted_time = f"{elapsed * 1_000_000:.2f} µs"
        elif elapsed < 1:
            formatted_time = f"{elapsed * 1000:.2f} ms"
        else:
            formatted_time = f"{elapsed:.2f} s"

        if isinstance(result, tuple):
            return(*result, formatted_time)
        else:
            return result, formatted_time
    
    return timed_method

def parse_time_string(time_str: str) -> float:
    """
    Parse the time-string into float values

    :param time_str: string to parse into float

    :return: float value of the time
    """

    val, unit = time_str.split()
    val = float(val)

    if unit in ['s', 'sec', 'seconds']:
        return val
    elif unit in ['ms', 'milliseconds']:
        return val / 1000
    elif unit in ['µs', 'us', 'microsecondi']:
        return val / 1_000_000
    elif unit in ['ns', 'nanoseconds']:
        return val / 1_000_000_000
    else:
        raise ValueError(f"Time unit not acceptable!: {unit}" )

def plot_grap_errors(errors_calc: dict) -> list:
    """
    Visualize the scaled residual for each iteration and for different methods on a single graph

    :param erros_calc: Dictionary where the Key:-name of the methods, Values:-scaled residual for iterations.
    
    :return: list of indexes
    """
    plt.figure(figsize=(10,6))
    index = []
    iter = 0

    for method, errors in errors_calc.items():
        if not errors:
            print(f"{method}: has not being used.")
        else:
            iterations = range(1, len(errors)+1)
            plt.plot(iterations, errors, label = method)
            index.append(iter)
        iter += 1

    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Scaled Residual')
    plt.title('Comparison between Methods')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

    return index

def plot_graph_useful_info(m_convergence: dict, index: list=[]) -> None:
    """
    Visualize useful information of the Method for solving a Linear System

    :param m_convergence: Dictionary where the Key:-name of the methods, Values:-boolean to verify if the 
    method converge or not.
    :param index: list of index of current Methods used
    
    :return: None
    """

    y = []
    x = []
    colors = []

    plt.figure(figsize=(10,6))
    iter = 0

    for method, converge in m_convergence.items():
        if iter in index:
            y.append(1 if converge else 0.02)
            x.append(method)
            colors.append('green' if converge else 'red')

        iter += 1

    plt.bar(x, y, color=colors, align='center', width=0.5)
    plt.yticks([0.02,1], ['No-Convergence', 'Convergence'])
    plt.title('Convergence of Iterative Methods')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, axis='y')
    plt.show()


def plot_time_execution_methods(time_data: dict={}, target_unit: str="ms", index: list=[]) -> None:
    """
    Visualize the time execution of each method

    :param time_data: dictionary. Key:-Methods' name. Value:-Time (str format)
    :param target_unit: string value for common unit between values
    :param index: list of index of current Methods used

    :return: None
    """

    conversion = {
        's': 1,
        'ms': 1000,
        'µs': 1_000_000,
        'ns': 1_000_000_000
    }

    if target_unit not in conversion:
        raise ValueError(f'Unit value not valid! Use these Keys: {conversion.keys()}')


    factor = conversion[target_unit]

    methods = []
    values = []
    iter = 0
    for method, time_str in time_data.items():
        if iter in index:
            f_time = parse_time_string(time_str)
            methods.append(method)
            values.append(f_time * factor)

        iter += 1
    
    plt.figure(figsize=(10,6))
    
    plt.xscale("log")
    plt.barh(methods, values, color='darkcyan')
    plt.xlabel(f"Execution Time ({target_unit})")
    plt.title("Comparison between Methods' Time-Execution")
    plt.grid(axis='x', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def get_path() -> str:
    """
    Get current path from the project

    :return: path
    """
    BASE_DIR = os.getcwd()
    dirpath = 'Metodi Calcolo Scientifico\Code\MyUtils\data_ion\\'
    path_complete = os.path.join(BASE_DIR, dirpath)
    
    check_path_exists(pathname=path_complete)

    return path_complete

def check_path_exists(pathname: str="") -> None:
    """
    Method to check if Path exist or not

    :param pathname: path to check the existence
    
    :return: None
    """
    if not os.path.exists(pathname):
        raise ValueError(f"Path: {pathname}, not valid!")
    
def get_path_general(filename:str="") -> str:
    """
    Get current path from the project folder

    :param filename: name of the file or the file's path (no need to add the Main's Folder) to add to the path

    :return: path
    """
    BASE_DIR = os.getcwd()
    dirpath = f'Metodi Calcolo Scientifico\{filename}'
    path_complete = os.path.join(BASE_DIR, dirpath)
    
    check_path_exists(pathname=path_complete)

    return path_complete

def get_method_active_values(active_values:dict={}) -> list:
    """
    Read the dictionary and get the boolean value for determine if a method is active or not

    :param active_values: dictionary from the JSON file

    :return: list
    """
    list_active = [None]*6
    iter = 0

    for method, setting in active_values.items():
        if setting.get("attivo", False):
            list_active[iter] = True
        else:
            list_active[iter] = False
        iter += 1

    return list_active
    