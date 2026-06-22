import json
import os
from collections import defaultdict

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def compare_metrics(new_metrics, old_metrics, model_name):
    differences = defaultdict(dict)
    max_abs_diff = 0
    max_perc_diff = 0
    max_abs_metric = None
    max_perc_metric = None

    def compare_recursive(new, old, path=""):
        nonlocal max_abs_diff, max_perc_diff, max_abs_metric, max_perc_metric
        if isinstance(new, dict) and isinstance(old, dict):
            for key in set(new.keys()) & set(old.keys()):
                compare_recursive(new[key], old[key], f"{path}.{key}" if path else key)
        elif isinstance(new, (int, float)) and isinstance(old, (int, float)):
            abs_diff = abs(new - old)
            perc_diff = abs_diff / abs(old) * 100 if old != 0 else float('inf')
            differences[model_name][path] = {'abs': abs_diff, 'perc': perc_diff}
            if abs_diff > max_abs_diff:
                max_abs_diff = abs_diff
                max_abs_metric = path
            if perc_diff > max_perc_diff and perc_diff != float('inf'):
                max_perc_diff = perc_diff
                max_perc_metric = path

    compare_recursive(new_metrics, old_metrics)
    return differences, max_abs_diff, max_perc_diff, max_abs_metric, max_perc_metric

def main():
    base_dir = '/home/noe/Workspace/TFG-Info'
    test_dir = os.path.join(base_dir, 'test')
    test_metrics_dir = os.path.join(base_dir, 'test_metrics')

    models = ['cwt_morlet', 'cwt_ricker', 'original', 'stft']
    all_differences = {}
    model_max_abs = {}
    model_max_perc = {}

    for model in models:
        new_file = os.path.join(test_dir, f'{model}_model', 'metrics.json')
        old_file = os.path.join(test_metrics_dir, f'{model}.json')

        if os.path.exists(new_file) and os.path.exists(old_file):
            new_metrics = load_json(new_file)
            old_metrics = load_json(old_file)

            diffs, max_abs, max_perc, max_abs_met, max_perc_met = compare_metrics(new_metrics, old_metrics, model)
            all_differences.update(diffs)
            model_max_abs[model] = (max_abs, max_abs_met)
            model_max_perc[model] = (max_perc, max_perc_met)
        else:
            print(f"Files for model {model} not found.")

    # Print results
    print("Differences per metric:")
    for model, metrics in all_differences.items():
        print(f"\nModel: {model}")
        for metric, diffs in metrics.items():
            print(f"  {metric}: Abs diff: {diffs['abs']:.6f}, Perc diff: {diffs['perc']:.2f}%")

    print("\nMax differences per model:")
    for model in models:
        if model in model_max_abs:
            abs_val, abs_met = model_max_abs[model]
            perc_val, perc_met = model_max_perc[model]
            print(f"{model}: Max abs diff: {abs_val:.6f} ({abs_met}), Max perc diff: {perc_val:.2f}% ({perc_met})")

if __name__ == "__main__":
    main()