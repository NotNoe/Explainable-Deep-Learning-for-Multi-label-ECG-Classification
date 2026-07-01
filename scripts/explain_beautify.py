import json
from math import ceil
import os
import numpy as np
import h5py
import ecg_plot
import matplotlib.pyplot as plt

def plot_ecg_with_explanations(N_ECG, explanation_path):
    # Load the explanation vector from the .npy file and transpose it to match the expected shape
    explanations = np.load(os.path.join(explanation_path, "explanation.npy")).T
    with h5py.File("./data/test.hdf5") as f:
        ecg = f["tracings"][N_ECG-2].T # type: ignore
    # The title containts the ECG id and the label
    ecg_plot.plot(ecg, sample_rate=400, title=f"ECG_ID: {explanation_path.split('/')[-1]} ({explanation_path.split('/')[-2]})", style="bw")
    fig = plt.gcf()
    fig.subplots_adjust(
        left = 0.1,
        right=0.95,
        bottom=0.1,
        top=0.95
    )
    # Clear the ticks and labels from the axes
    ax = fig.axes[0]
    ax.tick_params(
        labelbottom=False,
        labelleft=False,
        bottom=False,      
        left=False,
        length=0
    )
    # Basically, we replicate the code that the plot function does (which calculates offsets to know where to draw)
    # and then we put the heatmap in the same position
    sample_rate = 400
    secs = ecg.shape[1] / sample_rate

    columns = 2
    leads = ecg.shape[0] # It should be 12
    rows = int(ceil(leads / columns))
    row_height = 6
    lead_order = list(range(12))

    for c in range(columns):
        for i in range(rows):
            # For each derivation
            idx = c * rows + i
            if idx >= leads:
                break
            
            t_lead = lead_order[idx]
            
            # === Same offsets as the plot function ===
            x_offset = secs * c
            y_offset = -(row_height / 2) * (i % rows)
            
            # === We prepare the image that we are going to paint on top ===
            exp_lead = explanations[t_lead]         # (4096,)
            exp_lead_2d = exp_lead.reshape(1, -1)    # (1, 4096)
            
            # === We define the rectangle (extent) where that strip will be painted ===
            local_x_min = x_offset
            local_x_max = x_offset + secs
            
            # We choose the width of the strip so that it looks good.
            local_y_min = y_offset - 1.5
            local_y_max = y_offset + 1.5

            # We draw the heatmap with imshow
            #   zorder=-1 so that it is BELOW the ECG wave
            #   alpha=0.5 to allow the line to be seen
            heatmap = ax.imshow(
                exp_lead_2d,
                extent=(local_x_min, local_x_max, local_y_min, local_y_max),
                cmap='Reds',
                origin='lower',
                aspect='auto',
                alpha=0.5,
                zorder=-1
            )

    # We add a colorbar
    cbar = fig.colorbar(heatmap, ax=ax, shrink=1)
    cbar.set_label("Relevance")
    plt.savefig(os.path.join(explanation_path, "better_explanation.png"))


if __name__ == "__main__":
    perfect = json.load(open("perfects.json"))
    LABELS = ["CD", "HYP", "MI", "NORM", "STTC"]
    for label in LABELS:
        path = f"./out/explanations/{label}"
        array = perfect[label][:5]
        for item in array:
            print(f"Ploting {label} id {item['ecg_id']}")
            plot_ecg_with_explanations(item["linea"], os.path.join(path, str(item["ecg_id"])))