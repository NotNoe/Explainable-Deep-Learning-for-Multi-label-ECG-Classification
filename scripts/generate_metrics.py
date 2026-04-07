import numpy as np
import pandas as pd
from Metrics import Metrics, ComparisonMetrics
import os

MODELS = {"original_model", "stft_model", "cwt_morlet_model"}
MODELS_TO_COMPARE = MODELS - {"original_model"}
TEST_DATA_CSV = "./data/test_db.csv"

expected_values = np.array(pd.read_csv(TEST_DATA_CSV, index_col='ecg_id').values.tolist())
class_names = pd.read_csv(TEST_DATA_CSV).columns[1:].tolist()

for model in MODELS:
    results_path = f"./test/{model}"
    predicted_values = np.load(os.path.join(results_path, 'tmp', 'predictions.npy'))
    metrics = Metrics(expected_values, predicted_values, class_names=class_names)
    metrics.dump_to_json(os.path.join(results_path, 'metrics.json'))

original_model_predicted_values = np.load("./test/original_model/tmp/predictions.npy")
for model in MODELS_TO_COMPARE:
    predicted_values = np.load(os.path.join(f"./test/{model}", 'tmp', 'predictions.npy'))
    comparison_metrics = ComparisonMetrics(expected_values, original_model_predicted_values, predicted_values, class_names)
    comparison_metrics.dump_to_json(os.path.join(f"./test/{model}", 'comparison_to_original.json'))