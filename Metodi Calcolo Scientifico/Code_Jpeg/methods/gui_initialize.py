import customtkinter as ctk
from tkinter import *
from Code_Jpeg.methods.compression_jpeg import CompJpeg as CJpeg
from Code_Jpeg.methods.dct_two import DCT
from Code_Jpeg.utils_method.utils_gui import UtilsGui as ugui

class UserGui:

    def __init__(self) -> None:
        """
            Constructor of the class
        """
        self.app_gui = ctk.CTk()

        self.my_guitls = ugui()

        self.theme_options = self.my_guitls.get_color_themes()

        self.selected_theme = ctk.StringVar(value=self.theme_options[0])

        self.parameters = [ctk.StringVar() for _ in range(4)]

        self.input_frame = ctk.CTkFrame(self.app_gui)

        self.m_dct = DCT(n_dim=10)
        self.m_comp = None
        self.image_name = None
        self.N = None
        self.F = None
        self.d = None
    
    def set_appearance_mode(self) -> None:
        """
            Function to set the appearance mode

            :return: None
        """
        settings_frame = ctk.CTkFrame(self.app_gui)
        settings_frame.pack(pady=10, padx=10, fill="x")

        ctk.CTkLabel(settings_frame, text="Select Theme").pack()
        ctk.CTkOptionMenu(settings_frame, variable=self.selected_theme, values=self.theme_options).pack()

        ctk.CTkButton(settings_frame, text="Apply Theme and Color", command=self.apply_appearance_settings).pack(pady=10)

    def apply_appearance_settings(self) -> None:
        """
            Function to apply the Theme and Color selected

            :param None:

            :return: None
        """
        ctk.set_appearance_mode(self.selected_theme.get())


    def method_partone(self):
        """
            Function to execute method of Part I

            :param None:

            :return: None
        """
        print("Method Part I is Executing...")
        print(f"N: {self.N}")
        self.m_dct.set_N_var(self.N)
        f_mat = self.m_dct.build_matrix_center_point()
        #f_mat = self.m_dct.get_matrix_test()
        #self.m_dct.my_plot3D(f_mat)
        
        res_dct2_p, time1 = self.m_dct.apply_dct2_personalized(mat=f_mat)
        #print("Personale")
        #print(res_dct2_p)
        print(time1)
        #m_dct.my_plot3D(res_dct2_p)

        res_dct2_l, time2 = self.m_dct.apply_dct2_lib(mat=f_mat)
        #print("Libreria")
        #print(res_dct2_l)
        print(time2)
        #m_dct.my_plot3D(res_dct2_l)

        self.m_dct.my_plot_times(times_vect=[time1,time2])

        print("Method Part I Finished")
        
    
    def method_parttwo(self):
        """
            Function to execute method of Part II

            :param None:

            :return: None
        """
        print("Method Part II is Executing...")
        print(f"Image: {self.image_name}")
        print(f"F: {self.F}")
        print(f"d: {self.d}")

        self.m_comp = CJpeg(image_name=self.image_name)

        self.m_comp.set_image_var(image_name=self.image_name)
        self.m_comp.set_F_var(f=self.F)
        self.m_comp.set_d_var(d=self.d)

        self.m_comp.load_image()
        arr_o =  self.m_comp.get_image_original()
        print(f"Shape: {arr_o.shape}")

        if self.m_comp.is_gray_scale():
            self.m_comp.apply_dct2_to_blocks(img_block=self.m_comp.subdivide_in_block())
            arr_p =  self.m_comp.get_image_processed()
        else:
            self.m_comp.image_rgb_process()
            arr_p =  self.m_comp.get_image_processed_rgb()
        
    
        #print(arr_o.shape)
        #print(arr_p.shape)
        self.m_comp.show_images(i_ori=arr_o, i_pro=arr_p)

        print("Method Part II Finished")
        

    def print_parameters(self) -> None:
        """
            Function to show the parameters

            :param None:

            :return: None
        """
        print("Parameters set:")
        params = [param.get() for param in self.parameters]
        self.image_name, self.N, self.F, self.d = self.my_guitls.fetch_return_data(data=params)
        print("Image Name: "+ self.image_name + ";\n" + "N: "+ str(self.N) + ";\n" + "F: " + str(self.F) + ";\n" + "d: " + str(self.d))   

    def set_input_parameters(self) -> None:
        """
            Function to set the input parameters

            :param None:

            :return: None
        """

        self.input_frame.pack(pady=20)
        default_values = self.my_guitls.get_default_param_input()
        param_methods = self.my_guitls.get_type_param()
        for i, param in enumerate(self.parameters):
            param.set(default_values[i])
            ctk.CTkLabel(self.input_frame, text=f"{param_methods[i]}:").grid(row=0, column=i, padx=5)
            ctk.CTkEntry(self.input_frame, textvariable=param).grid(row=1, column=i, padx=5)

        ctk.CTkButton(self.app_gui, text="Confirm Input", command=self.print_parameters).pack(pady=10)

    def set_scrollable_window(self) -> None:
        """
            Function to set a scrollable window

            :param None:

            :return: None
        """
        self.scroll_frame = ctk.CTkFrame(self.app_gui)
        self.scroll_frame.pack(fill=BOTH, expand=True, pady=10)

        self.textbox = ctk.CTkTextbox(self.scroll_frame, width=500, height=150, wrap="word")
        self.textbox.pack(fill=BOTH, expand=True)

        images_list = self.my_guitls.get_images_files()
        i=1
        for item in images_list:
            self.textbox.insert("end","- "+ item + "\n")
            i+=1
        
        self.textbox.configure(state='disabled')

    def set_methods_start(self) -> None:
        """
            Function to set button for starting execute methods

            :param None:

            :return: None
        """
        ctk.CTkButton(self.app_gui, text="Execute Part 1", command=self.method_partone).pack(side="left", pady=5)
        ctk.CTkButton(self.app_gui, text="Execute Part 2", command=self.method_parttwo).pack(side="right", pady=5)


    def create_widgets(self) -> None:
        """
            Function to set the widgets in the GUI

            :param None:

            :return: None
        """
        self.set_scrollable_window()
        self.set_input_parameters()
        self.set_methods_start()
        self.set_appearance_mode()


    def my_app_initialize(self) -> None:
        self.app_gui.title('GUI - DCT Image and Other')
        self.app_gui.geometry(self.my_guitls.get_dimension_screen())

        self.create_widgets()        

        self.app_gui.mainloop()


