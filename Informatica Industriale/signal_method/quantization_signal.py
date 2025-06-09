import numpy as np
import matplotlib.pyplot as plt

def quantization(tsh, vinsh, FS, Nb):
    """
    Quantizza vinsh rispetto a FS e Nb bit.
    
    Args:
        tsh (array): asse temporale.
        vinsh (array): segnale da quantizzare.
        FS (float): full scale.
        Nb (int): numero di bit.

    Returns:
        vinq (ndarray): segnale quantizzato.
        eq (ndarray): errore di quantizzazione.
    """
    LSB = FS / (2 ** Nb)
    dr = np.arange(0, FS, LSB)

    vinq = np.zeros_like(vinsh)
    eq = np.zeros_like(vinsh)

    for i in range(len(vinsh)):
        idx = np.argmin(np.abs(vinsh[i] - dr))
        vinq[i] = dr[idx]
        eq[i] = vinsh[i] - vinq[i]

    #show_grap(tsh, vinq, eq)

    return vinq, eq


def show_grap(tsh, vinq, eq):
    plt.figure()
    plt.plot(tsh, vinq, 'b+', linewidth=2, label='vinq')
    plt.plot(tsh, eq, '-m', linewidth=2, label='Quantization Error')
    plt.xlabel('Time - [sec]')
    plt.ylabel('Amplitude - [V]')
    plt.legend()
    plt.grid(True)
    plt.title('Quantization Output and Error')
    plt.show()
