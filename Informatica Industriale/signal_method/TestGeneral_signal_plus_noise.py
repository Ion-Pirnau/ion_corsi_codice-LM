from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz
from scipy.ndimage import uniform_filter1d
from signal_method.quantization_signal import quantization  
from signal_method.updateres_signal import updateres
import os

def generate_noise_signal(nfifo:int, noise_on:int=0):
    # Costanti globali
    k = 13.8e-24
    T = 300
    nfig = 1

    # Impostazioni di output
    mypath = 'output_signal_noise'
    myfilename = 'mysignal.txt'
    writemyfile = True

    # Risoluzione dati
    Nb = 8
    tclk = 20e-9
    N = Nb
    NFIFO = nfifo

    # Segnale sinusoidale e rumore per riempire il range FS
    SNRdB = -6
    FS = 256
    SNR = 10 ** (SNRdB / 20)
    pnoise = FS / (2 * SNR * np.sqrt(2) + 3)
    A0 = pnoise * SNR * np.sqrt(2)
    A0off = A0

    # Periodo del segnale, oversampling, asse dei tempi
    T0 = (2 ** Nb) * tclk
    f0 = 1 / T0
    fs = 1 / tclk
    fcutoff = 0.5 * fs / NFIFO
    OVR = fs / (2 * f0)
    Mperiods = 10
    Nsamples_tot = Mperiods * 2 ** Nb

    t = np.arange(1, Mperiods * 2 ** Nb + 1)
    t = t / len(t)
    t = t * T0 * Mperiods

    # Generazione del segnale e rumore
    ysignal = noise_on * (A0off + (A0 * np.sin(2 * np.pi * f0 * t)))
    ynoise = pnoise * (1 + np.random.randn(len(t)))
    yns = ysignal + ynoise

    # Quantizzazione
    ynsq, eq = quantization(t, yns, FS, Nb)

    # Filtro di media mobile
    #ynsq_f = np.convolve(ynsq, mode=NFIFO)
    ynsq_f = uniform_filter1d(ynsq, size=NFIFO)
    save_filtered_data_txt(ynsq_f=ynsq_f, foldername=mypath)

    # Salvataggio su file
    if writemyfile:
        ynsqb = np.array(ynsq, dtype=int)
        ynsqb_bin = [format(v, f'0{Nb}b') for v in ynsqb]
        is_bool, full_path = is_path_accepted(strfoldername=mypath)
        if is_bool:
            with open(os.path.join(full_path, myfilename), 'w') as f:
                for b in ynsqb_bin:
                    f.write(b + '\n')

    # Potenza del rumore
    pnoise_power_rms = np.sqrt(np.mean((ynoise - np.mean(ynoise)) ** 2))

    # Parametri della simulazione
    res = updateres(Nb, tclk, NFIFO, SNRdB, FS, pnoise, A0, T0, f0, fs, OVR, Mperiods, Nsamples_tot)

    # Dominio della frequenza
    b = firwin(NFIFO + 1, fcutoff / (0.5 * fs))
    w, hf = freqz(b, worN=512)
    f = 0.5 * fs * w / np.pi

    plt.figure(nfig)
    plt.subplot(2, 1, 1)
    plt.semilogx(f, 20 * np.log10(np.abs(hf)), '-k', linewidth=2)
    plt.grid(True)
    plt.ylabel('[dB]')

    plt.subplot(2, 1, 2)
    plt.semilogx(f, np.angle(hf, deg=True), '-k', linewidth=2)
    plt.grid(True)
    plt.xlabel('Frequency - [Hz]')
    plt.ylabel('[deg]')
    nfig += 1

    # Dominio del tempo
    plt.figure(nfig)
    plt.plot(t, ynoise, '-r', linewidth=2, label='yn')
    plt.plot(t, ysignal, '-k', linewidth=5, label='ys')
    plt.plot(t, FS * np.ones_like(t), '--k', label='FS')
    plt.plot(t, 0*np.ones_like(t), '--k')
    plt.grid(True)
    plt.xlabel('Time - [s]')
    plt.ylabel('Values - [-]')
    plt.legend()
    plt.axis([0, Nsamples_tot * tclk, -100, 300])
    nfig += 1

    plt.figure(nfig)
    plt.plot(t, yns, '-b', linewidth=1, label='yns')
    plt.plot(t, FS * np.ones_like(t), '--k')
    plt.plot(t, np.zeros_like(t), '--k')
    plt.xlabel('Time - [s]')
    plt.ylabel('Values - [-]')
    plt.grid(True)
    plt.legend()
    plt.axis([0, Nsamples_tot * tclk, -100, 300])
    nfig += 1

    plt.figure(nfig)
    plt.plot(t, yns, '-b', linewidth=1, label='yns')
    plt.plot(t, ynsq, '-g', linewidth=3, label='ynsq')
    plt.plot(t, ynsq_f, '-m', linewidth=4, label='ynsqf')
    plt.xlabel('Time - [s]')
    plt.ylabel('Values - [-]')
    plt.grid(True)
    plt.legend()
    plt.axis([0, Nsamples_tot * tclk, -100, 300])
    nfig += 1

    plt.figure(nfig)
    plt.plot(t, ynsq_f, '-m', linewidth=6, label='ynsqf')
    plt.plot(t, ysignal + np.mean(ynoise), ':k', linewidth=4, label='ysignal')
    plt.xlabel('Time - [s]')
    plt.ylabel('Values - [-]')
    plt.grid(True)
    plt.legend()
    plt.axis([0, Nsamples_tot * tclk, -100, 300])

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

def save_filtered_data_txt(ynsq_f, foldername:str, mfilename:str="filtered_data.txt"):
    """
    Applica uniform_filter1d a ynsq e salva i dati filtrati su un file di testo.

    :param ynsq_f: Array di dati filtrato
    """

    # Scrittura dei dati su file di testo
    is_bool, full_path = is_path_accepted(strfoldername=foldername)
    if is_bool:
        with open(os.path.join(full_path, mfilename), 'w') as f:
            f.write("Filtered Data:\n")  # Intestazione
            for value in ynsq_f:
                f.write(f"{value}\n")

    print(f"Dati filtrati salvati in {mfilename}")