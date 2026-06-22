import json
import os
import numpy as np
import h5py
import pandas as pd
import sys

label_names = ["CD", "HYP", "MI", "NORM", "STTC"]

# --- Data loading ---
test_x = h5py.File('./data/test.hdf5', 'r')["tracings"]

# Loading labels from CSV file.
df = pd.read_csv('./data/test_db.csv')
test_y = df.drop(columns=['ecg_id']).to_numpy()

# --- Model execution ---

# Get predictions using the original Ribeiro script
os.system("python ./ribeiro/predict.py data/test.hdf5 final_models/original_model.hdf5 --output_file ./tmp.npy --dataset_name tracings")

# Load the predictions
predictions = np.load("./tmp.npy")

# Convert predictions to boolean values
predictions = predictions > 0.5

# --- Results comparison ---

# Check if the number of test samples and predictions match
if predictions.shape[0] != test_y.shape[0]:
    print("The number of test samples and predictions do not match")
    sys.exit(1)
if predictions.shape[1] != test_y.shape[1] or predictions.shape[1] != 5:
    print("The number of classes is not 5")
    sys.exit(1)

# We create an out.txt file where the results will be saved, if it already exists we delete it
index_list = {}
with open("out.json", "w") as f:
    # For each label, we create a vector representing the prediction and the real value that needs to be "adequate" to explain it
    for label_idx in range(5):
        #Desired_real is a vector with 1.0 in the position label_idx and 0.0 in the rest
        desired_real = np.zeros(5, dtype=np.float32)
        desired_real[label_idx] = 1.0
        desired_predictions = [False, False, False, False, False]
        desired_predictions[label_idx] = True
        desired_predictions = np.asarray(desired_predictions)
        index_list[label_names[label_idx]] = []
        for i in range(predictions.shape[0]):
            # For each prediction, we check if it matches the desired arrays
            if np.array_equal(test_y[i], desired_real) and np.array_equal(predictions[i], desired_predictions):
                # If it matches, we save the line from the CSV where it is (which corresponds to the index in the dataset plus two) and the ECG id
                index_list[label_names[label_idx]].append({"linea": i + 2, "ecg_id": int(df.iloc[i]['ecg_id'])})
    json.dump(index_list, f, indent=4)
os.remove("./tmp.npy")