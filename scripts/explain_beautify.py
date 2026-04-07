import json
from math import ceil
import os
import numpy as np
import h5py
import ecg_plot
import matplotlib.pyplot as plt

def plot_ecg_with_explanations(N_ECG, explanation_path):
    # Cargar el vector de numpy
    explicaciones = np.load(os.path.join(explanation_path, "explanation.npy")).T
    with h5py.File("./data/test.hdf5") as f:
        ecg = f["tracings"][N_ECG-2].T
    #El título contiene el id del ECG y la etiqueta
    ecg_plot.plot(ecg, sample_rate=400, title=f"ECG_ID: {explanation_path.split('/')[-1]} ({explanation_path.split('/')[-2]})", style="bw")
    fig = plt.gcf()
    fig.subplots_adjust(
        left = 0.1,
        right=0.95,
        bottom=0.1,
        top=0.95
    )
    #Quitamos las etiquetas de los ejes, que se concentran demasiado
    ax = fig.axes[0]
    ax.tick_params(
        labelbottom=False,
        labelleft=False,
        bottom=False,      
        left=False,
        length=0
    )
    #En esencia replicamos el código que hace la función plot (que calcula offsets para saber donde dibujar)
    #para luego poner el heatmap en la misma posición
    sample_rate = 400
    secs = ecg.shape[1] / sample_rate

    columns = 2
    leads = ecg.shape[0] #Debería de ser doce
    rows = int(ceil(leads / columns))
    row_height = 6
    lead_order = list(range(12))

    for c in range(columns):
        for i in range(rows):
            #Eso es para cada derivación
            idx = c * rows + i
            if idx >= leads:
                break
            
            t_lead = lead_order[idx]
            
            # === Mismos offsets que en plot
            x_offset = secs * c
            y_offset = -(row_height / 2) * (i % rows)
            
            # Preparamos la imagen que vamos a pintar po encima
            exp_lead = explicaciones[t_lead]         # (4096,)
            exp_lead_2d = exp_lead.reshape(1, -1)    # (1, 4096)
            
            # === Definimos el rectángulo (extent) donde se pintará esa franja ===
            local_x_min = x_offset
            local_x_max = x_offset + secs
            
            # Escogemos el ancho de la franja para que quede bien.
            local_y_min = y_offset - 1.5
            local_y_max = y_offset + 1.5
            
            # Dibujamos el heatmap con imshow
            #   zorder=-1 para que quede POR DEBAJO de la onda ECG
            #   alpha=0.5 (p.ej.) para dejar ver la línea
            heatmap = ax.imshow(
                exp_lead_2d,
                extent=(local_x_min, local_x_max, local_y_min, local_y_max),
                cmap='Reds',
                origin='lower',
                aspect='auto',
                alpha=0.5, #Para que no tape el resto de cosas
                zorder=-1 #Para que quede por debajo de la onda
            )

    # Añadimos una barra de color
    cbar = fig.colorbar(heatmap, ax=ax, shrink=1)
    cbar.set_label("Relevance")
    plt.savefig(os.path.join(explanation_path, "better_explanation.png"))


if __name__ == "__main__":
    perfect = json.load(open("out.json"))
    ETIQUETAS = ["CD", "HYP", "MI", "NORM", "STTC"]
    for label in ETIQUETAS:
        path = f"./out/explanations/{label}"
        array = perfect[label][:5]
        for item in array:
            print(f"Ploting {label} id {item['ecg_id']}")
            plot_ecg_with_explanations(item["linea"], os.path.join(path, str(item["ecg_id"])))