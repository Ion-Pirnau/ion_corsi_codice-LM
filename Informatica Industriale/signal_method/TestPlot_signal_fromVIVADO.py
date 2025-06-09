import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import resample
import os

def decipher_signal():
    # Parametri
    k = 13.8e-24
    T = 300
    Nb = 8
    nfig = 11

    # Path del file
    mypath = 'input_signal_filtered'
    mypath_out = 'output_signal_noise'
    myfilename = 'output_results_mdm_TEST.txt'
    myfilename_out = 'mysignal.txt'

    # Importazione e conversione
    is_bool, full_path = is_path_accepted(strfoldername=mypath)
    if is_bool:
        d = np.loadtxt(os.path.join(full_path, myfilename), dtype=str)
        ysignal_bin = np.array([int(x, 2) for x in d])
    else:
        raise FileNotFoundError(f"File {os.path.join(full_path, myfilename)} not found.")
    
    # Conversione e filtraggio
    ysignal = ysignal_bin
    ysignalFS = ysignal / (2**Nb)
    window_size = 64
    #ysignalFS = uniform_filter1d(ysignalFS, size=window_size)
    ysignalFS_f = np.convolve(ysignalFS, np.ones(window_size)/window_size, mode='same')

    ysignal_noise = read_filtered_data_txt(foldername=mypath_out)

    # Asse temporale
    tclk = 20e-9
    N = len(ysignal)
    N_noise = len(ysignal_noise)
    Tperiod = (2**Nb) * tclk
    fperiod = 1 / Tperiod
    fs = 1 / tclk
    TSTOP = tclk * N
    t = np.linspace(0, TSTOP, N)
    t_noise = np.linspace(0, TSTOP, N_noise)

    ysignal_interp = np.interp(t_noise, t, ysignal)

    #traslazione del valore medio (o centering) -- TECNICA
    ysignal_shifted = ysignal_interp + (np.mean(ysignal_noise) - np.mean(ysignal_interp))

    # Potenza del rumore
    pnoise_power_rms = np.sqrt(np.mean((ysignal - np.mean(ysignal))**2))

    # Plot dominio del tempo (valori digitali)
    plt.figure(nfig)
    plt.plot(t_noise, ysignal_shifted, '-b', linewidth=2, label='ysignal')
    plt.plot(t_noise, ysignal_noise, '-r', linewidth=3, label='ysignal_filtlib')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time - [s]')
    plt.ylabel('Digital - [-]')

    # Plot dominio del tempo (scalato tra 0 e 1)
    nfig += 1
    plt.figure(nfig)
    plt.plot(t, ysignalFS, '-y', linewidth=4)
    plt.grid(True)
    plt.xlabel('Time - [s]')
    plt.ylabel('Digital - [-]')
    # plt.axis([1e-6, 5e-6 + (2**Nb)*2*tclk, -0.1, 1.1])  # Scommentare se serve
    plt.tight_layout()
    plt.show()


def is_path_accepted(strfoldername:str,strfilename:str=''):
    full_path = get_full_path()
    full_path = os.path.join(full_path, strfoldername, strfilename)
    if os.path.exists(full_path):
        return True, full_path
    else:
        return False, 'No valid'
    

def get_full_path() -> str:
        """
            Function to get the full path where the Images are

            :param None:

            :return: Path
        """

        project_path = os.getcwd()

        full_path = os.path.join(project_path,"Informatica Industriale")

        return full_path

def read_filtered_data_txt(foldername:str, mfilename:str="filtered_data.txt"):
    """
    Legge da file

    """
    data = []

    # Scrittura dei dati su file di testo
    is_bool, full_path = is_path_accepted(strfoldername=foldername)
    if is_bool:
        with open(os.path.join(full_path, mfilename), 'r') as f:
            lines = f.readlines()[1:]
            #print(lines)
            for value in lines:
                data.append(float(value.strip()))
    
    return np.array(data)