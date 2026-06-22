import json
from tensorflow.keras.models import load_model
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR
import pandas as pd
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)

perfect = json.load(open("perfects.json"))
ETIQUETAS = ["CD", "HYP", "MI", "NORM", "STTC"]
# Load the datasets
test_x = h5py.File('./data/test.hdf5', 'r')["tracings"]
# Load the labels from the CSV file
df = pd.read_csv('./data/test_db.csv')
test_y = df.drop(columns=['ecg_id']).to_numpy()
# Load model
model = load_model("./final_models/original_model.hdf5")

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

for label in ETIQUETAS:
    os.makedirs(f"./out/explanations/original_model/{label}", exist_ok=True)
    array = perfect[label][:5]
    for item in array:
        ITEM_IDX = item["linea"] - 2
        ecg_id = item["ecg_id"]


        with strategy.scope():
            explainer = TSR(
                model, test_x.shape[-2], test_x.shape[-1], mode="time", method="IG", device="cuda")
            item = test_x[ITEM_IDX:ITEM_IDX+1, :, :]
            prediction = model.predict(item)[0]
            # Imprime los índices de las etiquetas >0.5
            print(prediction > 0.5)
            # Explica solo la primera etiqueta "true"
            for label_idx in range(prediction.shape[0]):
                if prediction[label_idx] > 0.5:
                    explanation = explainer.explain(item, label_idx)
                    break

        os.makedirs(
            f"./out/explanations/original_model/{label}/{ecg_id}", exist_ok=True)
        # save the explanation ndarray
        np.save(
            f"./out/explanations/original_model/{label}/{ecg_id}/explanation.npy", explanation)
        with plt.ioff():
            explainer.plot(item, explanation, figsize=(
                15, 60), save=f"./out/explanations/original_model/{label}/{ecg_id}/explanation.png")
