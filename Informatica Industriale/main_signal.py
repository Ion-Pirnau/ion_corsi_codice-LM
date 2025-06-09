from signal_method.TestGeneral_signal_plus_noise import is_path_accepted as ipa
from signal_method.TestGeneral_signal_plus_noise import generate_noise_signal as gns
from signal_method.TestPlot_signal_fromVIVADO import decipher_signal as decsign
from signal_method.TestPlot_signal_fromVIVADO import is_path_accepted as ipa_dec
import os


if __name__ == '__main__':
    #is_bool, full_path = ipa_dec(strfoldername="input_signal_filtered")
    #print(str(is_bool) + "\n" + os.path.join(full_path, 'mysignal.txt'))
    #gns(nfifo=128, noise_on=1)
    decsign()