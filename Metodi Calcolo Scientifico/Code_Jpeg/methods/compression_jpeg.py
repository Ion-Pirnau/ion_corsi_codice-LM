from Code_Jpeg.utils_method.utils import MyUtils as MyUtl
import numpy as np
from scipy.fft import dct as lib_dct
from scipy.fft import idct as lib_idct

class CompJpeg:
    """
        Class used to compress file image throught DCT
    """

    def __init__(self, image_name:str, F:int=10, d:int=0) -> None:
        """
            Constructor

            :param image_name: name of the image
            :param F: breadth's value for each box the DCT2 should be applied
            :param d: threshold's cut of the frequency

            :return: None
        """
        self.myutils = MyUtl()

        bool_value, self.full_path = self.myutils.check_path_image(image_name=image_name)

        if bool_value:
            self.image_path = image_name
        else:
            raise ValueError(f"The image: {image_name} does NOT EXISTS!")

        
        self.F = F
        if self.myutils.check_value_range(F=F, d=d):
            self.d = d
        else:
            raise ValueError(f"{self.d} IS NOT acceptable! Not in range [{0} - {2*self.F - 2}]")
        
        self.img_array = None
        self.img_processed = None
        self.img_processed_rgb = None

    def load_image(self) -> None:
        """
            Function to load the image

            :return: None
        """
        self.img_array = self.myutils.load_image(path=self.full_path)

    def subdivide_in_block(self) -> np.array:
        """
            Function to subdivide the image into FxF blocks of pixels

            :return: np.array Matrix
        """
        #print("Shape: "+str(self.img_array.shape))
        #print(self.img_array)
        h, w = self.img_array.shape

        h_crop = h - (h % self.F)
        w_crop = w - (w % self.F)

        return self.img_array[:h_crop, :w_crop]
    
    def subdivide_in_block_rgb(self, data_i:np.array) -> np.array:
        """
            Function to subdivide the image into FxF blocks of pixels

            :param data_i: data to subdivide

            :return: np.array Matrix
        """
        #print("Shape: "+str(self.img_array.shape))
        #print(self.img_array)
        h, w = data_i.shape

        h_crop = h - (h % self.F)
        w_crop = w - (w % self.F)

        return data_i[:h_crop, :w_crop]

    def get_image_original(self) -> np.array:
        """
            Function to return Image array Original

            :return: np.array Matrix
        """
        return self.img_array
    
    def get_image_processed(self) -> np.array:
        """
            Function to return Image array Processed

            :return: np.array Matrix
        """
        return self.img_processed
    
    def get_image_processed_rgb(self):
        """
            Function to return Image array Processed RGB

            :return: np.array Matrix
        """
        return self.img_processed_rgb
    
    def apply_dct2_to_blocks(self, img_block:np.array) -> None:
        """
            Execute DCT from SciPy library to array's blocks

            Remove the frequency

            Apply IDCT

            Round to nearest integer

            :param img_block: matrix

            :return: None
        """
        h, w = img_block.shape

        self.img_processed = np.zeros_like(img_block)

        norm_type = ["ortho", None]
        #MonoDimnesional

        for i in range(0, h, self.F):
            for j in range(0, w, self.F):
                block = img_block[i:i+self.F, j:j+self.F]
                #print(block)
                c = lib_dct(lib_dct(block, norm=norm_type[0], axis=0), axis=1, norm=norm_type[0])

                #Remove Frequency
                for k in range(self.F):
                    for l in range(self.F):
                        if k + l >= self.d:
                            c[k, l] = 0
                #Apply IDCT
                ff = lib_idct(lib_idct(c, norm=norm_type[0], axis=0), axis=1, norm=norm_type[0])

                #Fix the data - Round etc..
                ff = np.rint(ff).astype(np.int32)
                ff = np.clip(ff, 0, 255).astype(np.uint8)

                self.img_processed[i:i+self.F, j:j+self.F] = ff

    def show_images(self, i_ori:np.array, i_pro:np.array) -> None:
        """
            Function to display Images: Original and Processed

            :param i_ori: original array image
            :param i_pro: processed array image

            :return: None
        """
        self.myutils.visualize_two_images(img1=i_ori, img2=i_pro)


    def set_image_var(self, image_name:str) -> None:
        """
            Function to set the Image variable

            :param None:

            :return: None
        """
        bool_value, self.full_path = self.myutils.check_path_image(image_name=image_name)

        if bool_value:
            self.image_path = image_name
        else:
            raise ValueError(f"The image: {image_name} does NOT EXISTS!")
    
    def set_F_var(self, f:int) -> None:
        """
            Function to set the F variable

            :param None:

            :return: None
        """
        self.F = f
    
    def set_d_var(self, d:int) -> None:
        """
            Function to set the d variable

            :param None:

            :return: None
        """
        if self.myutils.check_value_range(F=self.F, d=d):
            self.d = d
        else:
            raise ValueError(f"{d} IS NOT acceptable! Not in range [{0} - {2*self.F - 2}]")
    
    def image_rgb_process(self) -> None:
        """
            Method to process RGB images

            :retunr: None
        """
        processed_channels = []

        for i in range(3):
            channel = self.img_array[:, :, i]
            #print(channel.shape)
            #print("----")
            blocks = self.subdivide_in_block_rgb(data_i=channel)
            self.apply_dct2_to_blocks(img_block=blocks)
            processed_block = self.img_processed
            #print(processed_block.shape)
            #print("----")
            processed_channels.append(processed_block)
        self.img_processed_rgb = np.stack(processed_channels, axis=2)
    
    def is_gray_scale(self) -> bool:
        """
            Function to define if the image is in gray scale or rbg
        """

        if self.img_array.ndim == 2:
            return True
        elif self.img_array.ndim == 3:
            return False
        else:
            raise ValueError("Image Format not supported!")