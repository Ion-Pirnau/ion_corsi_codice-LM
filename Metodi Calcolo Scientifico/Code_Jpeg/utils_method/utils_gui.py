from Code_Jpeg.utils_method.utils import MyUtils
import os

class UtilsGui:
    """
        Utility class for GUI
    """
    def __init__(self) -> None:
        """
            Constructor

            :param None:

            :return: None
        """
        self.myutls = MyUtils()

    def get_color_themes(self) -> tuple[str]:
        """
            Function to get colors themes

            :param None:

            :return: Array of String
        """

        color_theme = ["System", "Light", "Dark"]

        return color_theme
    
    def get_color_in_window(self) -> tuple[str]:
        """
            Function to get colors text in gui

            :param None:

            :return: Array of String
        """

        color_in_window = ["blue", "green", "dark-blue"]

        return color_in_window
    
    def get_type_param(self) -> tuple[str]:
        """
            Function to get variable for methods developed

            :param None:

            :return: Array of String
        """

        params_for_method = ["Image", "N", "F", "d"]

        return params_for_method

    def get_images_files(self) -> tuple[str]:
        """
            Function to get list of images

            :param None:

            :return: Array of String
        """
        
        files = os.listdir(self.myutls.get_full_path())

        return files
        
    def get_dimension_screen(self) -> str:
        """
            Function to get size of the GUI screen

            :param None:

            :return: Array of String
        """

        return '600x500'
    
    def get_title_screen(self) -> str:
        """
            Function to get the title of GUI screen

            :param None:

            :return: Array of String
        """

        return 'GUI - DCT Image and Other'
    
    def fetch_return_data(self, data:list[str]) -> tuple[str, int, int, int]:
        """
            Function to fetch the data and return the parse data for every single variable

            :param data: list of strings

            :return: Tuple[str, int, int, int]
        """
        if data[0]:
            image_name = data[0]
        else:
            raise ValueError("Insert a valid value!")
        
        if data[1]:
            N = int(data[1])
        else:
            raise ValueError("Insert a valid value!")
        
        if data[2]:
            F = int(data[2])
        else:
            raise ValueError("Insert a valid value!")
        
        if data[3]:
            d = int(data[3])
        else:
            raise ValueError("Insert a valid value!")

        return image_name, N, F, d
    
    def get_default_param_input(self) -> tuple[str]:
        """
            Function to set default value for the input in GUI

            :param None:

            :return: Tuple[str, int, int, int]
        """
        default_values = ["None", "0", "0", "0"]

        return default_values