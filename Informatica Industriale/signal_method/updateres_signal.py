from dataclasses import dataclass

@dataclass
class SimConfig:
    Nb: int
    tclk: float
    NFIFO: int
    SNRdB: float
    FS: float
    pnoise: float
    A0: float
    T0: float
    f0: float
    fs: float
    fcutoff: float
    OVR: float
    Mperiods: int
    Nsamples_tot: int
    TOTAL_SIM_TIME: float

def updateres(Nb, tclk, NFIFO, SNRdB, FS, pnoise, A0, T0, f0, fs, OVR, Mperiods, Nsamples_tot):
    return SimConfig(
        Nb, tclk, NFIFO, SNRdB, FS, pnoise, A0, T0, f0, fs,
        0.5 * fs / NFIFO, OVR, Mperiods, Nsamples_tot, Nsamples_tot * tclk
    )