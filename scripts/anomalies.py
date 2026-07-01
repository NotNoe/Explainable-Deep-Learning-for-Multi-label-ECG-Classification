import ecg_plot
import os
import matplotlib
import matplotlib.pyplot as plt
import h5py

matplotlib.use('Agg')
ids = {"CD": 24, "MI": 70, "STTC": 21, "NORM": 2, "HYP": 225}
os.makedirs(f"./out/ecg_clases", exist_ok=True)
for type in ["CD", "HYP", "MI", "NORM", "STTC"]:
    if type != "HYP":
        #Generates a string from a number ensuring it has 5 digits
        path = "./ptbxl/records500/00000/" + str(ids[type]).zfill(5) + "_hr"
    else:
        path = "./ptbxl/records500/01000/" + str(ids[type]).zfill(5) + "_hr"
    with h5py.File("data/test.hdf5", 'r') as file:
        signals = file["tracings"][ids[type]-2].T # type: ignore
    ecg_plot.plot(signals, sample_rate=400, title=f"Clase: {type}")
    fig = plt.gcf()
    for ax in fig.get_axes():
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Hide numbers on X-axis
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)  # Hide numbers on Y-axis
    ecg_plot.save_as_png(f"./out/ecg_clases/{type}")