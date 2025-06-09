import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os
from PIL import Image

class MyUtils:
    """
        Utility class for plotting on 3D axes, user-interface and other
        various stuff
    """
    def __init__(self) -> None:
        """
            Constructor

            :param None:

            :return: None
        """

    def normalized_in_range(self, d_mat:tuple[float], size_before:int, a:float=0.0, b:float=1.0) -> tuple[float]:
        """
            Function to normalized a vector into a range [a, b]

            :param d_mat: vector to normalized
            :param size_before: size of the matrix before flatten
            :param a: starting range
            :param b: ending range

            :return: Vector normalized
        """
        N = size_before*size_before
        normalized_vector = np.zeros(N)

        min_value = np.min(d_mat)
        max_value = np.max(d_mat)

        for i in range(N):
            normalized_vector[i] = a + ((d_mat[i] - min_value)*(b-a) / (max_value - min_value)) 

        normalized_vector = np.reshape(normalized_vector, shape=(size_before, size_before))
        return normalized_vector

    def plot_bar3d(self, p_mat:tuple[float, float], title:str="") -> None:
        """
            Function to plot a matrix on a 3D grid

            :param p_mat: data to plot on the 3D grid
            :param title: title for the graph

            :return: None
        """
        N = p_mat.shape[0]

        xpos, ypos = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        normalized_matrix = self.normalized_in_range(p_mat.flatten(), size_before=p_mat.shape[0], a=-1.0, b=1.0)


        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros_like(xpos)


        xsize = ysize = 0.8 * np.ones_like(zpos)
        zsize = p_mat.flatten()

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.bar3d(xpos, ypos, zpos, xsize, ysize, zsize)

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("F(x, y)")

        plt.tight_layout()
        plt.show()


    def parse_time_string(self, time_str: str) -> float:
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

    def plot_time_execution_methods(self, time_data: dict={}, target_unit: str="ms") -> None:
        """
        Visualize the time execution of each method

        :param time_data: dictionary. Key:-Methods' name. Value:-Time (str format)
        :param target_unit: string value for common unit between values

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
        
            f_time = self.parse_time_string(time_str)
            methods.append(method)
            values.append(f_time * factor)

            iter += 1
        
        plt.figure(figsize=(10,6))
        
        plt.xscale("log")
        plt.barh(methods, values, color='darkcyan')
        plt.xlabel(f"Execution Time ({target_unit})")
        plt.title("Comparison between DCT Methods' Time-Execution")
        plt.grid(axis='x', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()


    def build_dictionary_dct(self, times_set:tuple[float]) -> dict[str, float]:
        """
            Function to construct a Dictionary

            :param times_set: vector of times

            :return: a Dictiornay
        """

        my_dict = {
            "DCT-SCHOOL":times_set[0],
            "DCT-LIBRARY":times_set[1]
        }

        return my_dict
    
    def define_vector_test(self) -> tuple[float, float]:
        """
            Function to define the vector to test the DCT2

            :param None:

            :return: Matrix
        """
        vector = [
            [231, 32, 233, 161, 24, 71, 140, 245],
            [247, 40, 248, 245, 124, 204, 36, 107],
            [234, 202, 245, 167, 9, 217, 239, 173],
            [193, 190, 100, 167, 43, 180, 8, 70],
            [11, 24, 210, 177, 81, 243, 8, 112],
            [97, 195, 203, 47, 125, 114, 165, 181],
            [193, 70, 174, 167, 41, 30, 127, 245],
            [87, 149, 57, 192, 65, 129, 178, 228]
        ]
        
        return np.array(vector)
    
    def check_value_range(self, F:int, d:int) -> bool:
        """
            Function to check the value d if is in range from 0 to 2F-2

            :param F: value used to calculate the range
            :param d: value to check

            :return: True/False
        """

        return True if 0 <= d <= (2 * F) -2 else False

    def check_path_image(self, image_name:str) -> bool:
        """
            Function to check if image name exist in the 'immagini' folder

            :param image_name: str value of an image file name

            :return: True/False
        """

        #Path of the current .py file
        #project_path = os.path.dirname(os.path.abspath(__file__))
        #file_extension = [".bmp", ".png", ".jpeg"]
        r_ext = None
        is_good = False

        #print(image_name)
        #project_path = os.getcwd()
        n_file_folder = []
        n_file_ext = []
        f_array = self.get_images_files()
        
        for item in f_array:
            i_name, ext = item.split('.')
            n_file_folder.append(i_name)
            n_file_ext.append(ext)

        if "." in image_name:
            image_name, m_ext = image_name.split('.')
            if m_ext.lower() not in [ext.lower() for ext in n_file_ext]:
                raise ValueError("Extension not in the DB")
        
        i = 0
        for item in n_file_folder:
            if image_name == item:
                r_ext = "."+n_file_ext[i]
                is_good = True
            i+=1

        if is_good:
            full_path = os.path.join(self.get_full_path(), image_name+r_ext)
        else:
            raise ValueError("File Not Found!")
            
        #print("File Path:"+full_path)

        #print(os.path.exists(full_path))

        return os.path.exists(full_path), full_path
    
    def get_full_path(self) -> str:
        """
            Function to get the full path where the Images are

            :param None:

            :return: Path
        """

        project_path = os.getcwd()

        full_path = os.path.join(project_path,"Metodi Calcolo Scientifico","Code_Jpeg","immagini")

        return full_path
    
    def load_image(self, path:str) -> np.array:
        """
            Function to load image file 

            :param path: path of the file to load

            :return: np.array()
        """

        #img = Image.open(path).convert('L')
        img = Image.open(path)
        img = np.array(img)
        if img.ndim == 2:
            return img
        elif img.ndim == 3 and (img.shape[2] == 3 or img.shape[2] == 4):
            return img[:,:,:3]
        else:
            raise ValueError("Image Format not supported!")
    
    def visualize_two_images(self, img1:np.array, img2:np.array) -> None:
        """
            Function for showing the two images and compare them
        """

        fig, axs = plt.subplots(1, 2, figsize=(10,4))

        axs[0].imshow(img1, cmap='gray')
        axs[0].set_title('Original')
        axs[0].axis('off')

        axs[1].imshow(img2, cmap='gray')
        axs[1].set_title('Processed')
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()


    def get_images_files(self) -> tuple[str]:
        """
            Function to get list of images

            :param None:

            :return: Array of String
        """
        
        files = os.listdir(self.get_full_path())

        return files





