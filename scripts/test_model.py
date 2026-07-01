import argparse
import os
import shutil
import sys
import numpy as np
import pandas as pd
from Metrics import Metrics


TEST_DATA = "./data/test"


def get_predictions(model_path, test_data_hdf5, results_path, dataset_name):
    os.system(f'python ./ribeiro/predict.py {test_data_hdf5} {model_path} --output_file ' + os.path.join(results_path, 'tmp', 'predictions.npy') + " --dataset_name " + dataset_name)
def main():
    parser = argparse.ArgumentParser(description='Test a model.')
    parser.add_argument('model', type=str, help='The name of the model to be used')
    parser.add_argument('dataset', type=str, help='The name of the dataset to be used')
    parser.add_argument('--model-dir', type=str, default='./final_models', help='The directory where the model is located')
    parser.add_argument('--result-dir', type=str, default='./test', help='The directory where the results will be saved')
    parser.add_argument('--use-cached', action="store_true", help='Use cached predictions if available')
    parser.add_argument('--show-cmatrix', action="store_true", help='Show confusion matrix')

    
    args = parser.parse_args()
    model_path = os.path.join(args.model_dir, args.model + '.hdf5')
    results_path = os.path.join(args.result_dir, args.model)
    test_data_csv = TEST_DATA + '_db.csv'
    test_data_hdf5 = TEST_DATA + '.hdf5'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        sys.exit(1)
    if not os.path.exists(test_data_hdf5):
        print(f"Test data file not found at {test_data_hdf5}")
        sys.exit(1)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if not os.path.exists(test_data_csv):
        print(f"Label data file not found at {test_data_csv}")
        sys.exit(1)
    else:
        class_names = pd.read_csv(test_data_csv).columns[1:].tolist()
    if args.use_cached:
        if not os.path.exists(os.path.join(results_path, 'tmp', 'predictions.npy')):
            os.makedirs(os.path.join(results_path, 'tmp'), exist_ok=True)
            get_predictions(model_path, test_data_hdf5, results_path, args.dataset)
    else:
        if os.path.exists(os.path.join(results_path, 'tmp')):
            shutil.rmtree(os.path.join(results_path, 'tmp'))
        os.mkdir(os.path.join(results_path, 'tmp'))
        get_predictions(model_path, test_data_hdf5, results_path, args.dataset)

    expected_values = np.array(pd.read_csv(test_data_csv, index_col='ecg_id').values.tolist())
    predicted_values = np.load(os.path.join(results_path, 'tmp', 'predictions.npy'))
    metrics = Metrics(expected_values, predicted_values, class_names=class_names)
    metrics.dump_to_json(os.path.join(results_path, 'metrics.json'))
    if args.show_cmatrix:
        metrics.plot_confusion_matrix()


if __name__ == "__main__":
    main()