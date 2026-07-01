from scripts.Transformations import Transformations
import numpy as np
import h5py
import matplotlib.pyplot as plt 
from scipy.signal import morlet2 as morlet
import os
import ecg_plot

lines = {"CD": 24, "MI": 70, "STTC": 21, "NORM": 2, "HYP": 225}
ids = {"CD": 157, "MI": 514, "STTC": 128, "NORM": 9, "HYP": 225}


for clase in ["CD", "HYP", "MI", "NORM", "STTC"]:
    plt.close('all')
    with h5py.File("data/test.hdf5", 'r') as file:
        ecg = file["tracings"][lines[clase]-2]
    transformer = Transformations(np.array([ecg]))
    stft = transformer.stft()[0]
    f_stft, t_stft = transformer.get_stft_arrays()
    cwt_ricker = transformer.cwt()[0]
    f_cwt_ricker, t_cwt_ricker = transformer.get_cwt_arrays()
    cwt_morlet = transformer.cwt(morlet, scales=np.linspace(8,637,100))[0]
    f_cwt_morlet, t_cwt_morlet = transformer.get_cwt_arrays(morlet, scales=np.linspace(8,637,100))
    batch_stft = stft.shape[0] // 12
    batch_cwt_ricker = cwt_ricker.shape[0] // 12
    batch_cwt_morlet = cwt_morlet.shape[0] // 12
    os.makedirs(f"out/trans_clases/{clase}", exist_ok=True)
    plt.imshow(stft[0:batch_stft,:], extent=[t_stft[0], t_stft[-1], f_stft[0], f_stft[-1]], origin='lower', aspect='auto', cmap='viridis')
    plt.title(f"ECG_ID: {ids[clase]} ({clase})")
    plt.gca().axes.set_ylabel("Frequency (Hz)")
    plt.gca().axes.set_xlabel("Time (s)")
    plt.savefig(f"out/trans_clases/{clase}/stft.png")

    plt.imshow(cwt_ricker[0:batch_cwt_ricker,:], extent=[t_cwt_ricker[0], t_cwt_ricker[-1], f_cwt_ricker[0], f_cwt_ricker[-1]], origin='lower', aspect='auto', cmap='viridis')
    plt.title(f"ECG_ID: {ids[clase]} ({clase})")
    plt.gca().axes.set_ylabel("Scales")
    plt.gca().axes.set_xlabel("Time (s)")
    plt.savefig(f"out/trans_clases/{clase}/cwt_ricker.png")
    

    plt.imshow(cwt_morlet[0:batch_cwt_morlet,:], extent=[t_cwt_morlet[0], t_cwt_morlet[-1], f_cwt_morlet[0], f_cwt_morlet[-1]], origin='lower', aspect='auto', cmap='viridis')
    plt.title(f"ECG_ID: {ids[clase]} ({clase})")
    plt.gca().axes.set_ylabel("Scales")
    plt.gca().axes.set_xlabel("Time (s)")
    plt.savefig(f"out/trans_clases/{clase}/cwt_morlet.png")

    ecg_plot.plot(np.array([ecg.T[0]]), sample_rate=400, columns=1, title="")
    fig = plt.gcf()
    plt.title(f"ECG_ID: {ids[clase]} ({clase})")
    fig.subplots_adjust(
        left = 0.05,
        right=0.95,
        bottom=0.25,
        top=0.75
    )
    ax = fig.axes[0]
    ax.tick_params(
        labelbottom=False,
        labelleft=False,
        bottom=False,      
        left=False,
        length=0
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    plt.savefig(f"out/trans_clases/{clase}/ecg.png")
