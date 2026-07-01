import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tabulate import tabulate

TEST_DIR = Path("./test")
OUT_DIR = Path("./out/metrics")

if not TEST_DIR.exists():
    print(f"❌ Error: Directory not found: {TEST_DIR}")
    exit(1)
OUT_DIR.mkdir(parents=True, exist_ok=True)
CLASSES = ["CD", "HYP", "MI", "NORM", "STTC"]
METRICS = ["f1_score", "recall", "precision", "roc_auc"]

def generate_metrics_tables(print_tables: bool = True, save_tables: bool = False, latex_mode: bool = False) -> None:
    SAVE_PATH = OUT_DIR / 'metrics_tables'
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    for model_dir in TEST_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        metrics_path = model_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"metrics.json not found in {model_dir}")
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        model_name = model_dir.name
        model_title = model_name.replace('_', ' ').title()

        table_list = {model_title: ["F1", "Recall", "Precision", "ROC AUC"]}
        for ecg_class in CLASSES:
            metrics_values = [metrics.get(metric, {}).get('by_class', {}).get(ecg_class, np.nan) for metric in METRICS]
            confidence_intervals = [metrics.get('confidence_intervals', {}).get(metric, {}).get(ecg_class, [np.nan, np.nan]) for metric in METRICS]
            table_list[ecg_class] = [f"{val:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]" for val, ci in zip(metrics_values, confidence_intervals)]
        global_metrics_values = [metrics.get(metric, {}).get('global_average', np.nan) for metric in METRICS]
        global_confidence_intervals = [metrics.get('confidence_intervals', {}).get(f'{metric}_global', [np.nan, np.nan]) for metric in METRICS]
        table_list["Global"] = [f"{val:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]" for val, ci in zip(global_metrics_values, global_confidence_intervals)]

        table = tabulate(table_list, headers = 'keys', tablefmt = 'psql')
        if print_tables:
            print(f"\nMMetrics for {model_title}:\n")
            print(table)
        if save_tables:
            with open(SAVE_PATH / f'{model_name}_metrics.txt', 'w') as f:
                f.write(tabulate(table_list, headers = 'keys', tablefmt = 'psql'))
        if latex_mode:
            latex_table = tabulate(table_list, headers = 'keys', tablefmt = 'latex')
            with open(SAVE_PATH / f'{model_name}_metrics.tex', 'w') as f:
                f.write(latex_table)

def plot_roc_curves_by_model(show_plots: bool = False, save_plots: bool = True) -> None:
    SAVE_PATH = OUT_DIR / 'roc_curves_by_model'
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.Set1(np.linspace(0, 1, len(CLASSES)))
    for model_dir in TEST_DIR.iterdir():
        fig, ax = plt.subplots()
        model_name = model_dir.name
        model_title = model_name.replace('_', ' ').title()
        metrics_path = model_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"metrics.json not found in {model_dir}")
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
            roc_curves = metrics.get('roc_curve', None)
        if roc_curves is None:
            print(f"roc_curve not found in {model_dir}/metrics.json")
            continue

        roc_auc = metrics.get('roc_auc', {}).get('by_class', {})
        plotted = False
        for class_name in CLASSES:
            if class_name not in roc_curves:
                continue
            curve_data = roc_curves[class_name]
            fpr = curve_data.get('fpr', [])
            tpr = curve_data.get('tpr', [])
            auc = roc_auc.get(class_name, np.nan)
            if not fpr or not tpr:
                continue
            auc_label = f'{auc:.3f}' if not pd.isna(auc) else 'n/a'
            ax.plot(fpr, tpr, color=colors[CLASSES.index(class_name)], linewidth=2, label=f'{class_name} (AUC = {auc_label})')
            plotted = True

        if not plotted:
            ax.text(0.5, 0.5, f'No ROC data for {model_title}', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC = 0.5)')
            ax.legend(loc="lower right", fontsize=8)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title(f'{model_name.replace("_", " ").title()}', fontweight='bold')
        ax.grid(alpha=0.3)

        fig.tight_layout()

        if show_plots:
            plt.show()
        if save_plots:
            plt.savefig(SAVE_PATH / f'{model_name}_roc_curve.pdf', dpi=300, bbox_inches='tight')
            print(f"  ✓ Guardado: {model_name}_roc_curve.pdf")

def plot_roc_curves_by_class(show_plots: bool = False, save_plots: bool = True) -> None:
    colors = plt.cm.Set2(np.linspace(0, 1, len(list(TEST_DIR.iterdir()))))
    SAVE_PATH = OUT_DIR / 'roc_curves_by_class'
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    for class_name in CLASSES:
        fig, ax = plt.subplots()
        class_title = class_name
        plotted = False
        
        for model_idx, model_dir in enumerate(sorted(TEST_DIR.iterdir())):
            model_name = model_dir.name
            metrics_path = model_dir / "metrics.json"
            if not metrics_path.exists():
                print(f"metrics.json not found in {model_dir}")
                continue
            with open(metrics_path) as f:
                metrics = json.load(f)
                roc_curves = metrics.get('roc_curve', None)
            if roc_curves is None:
                print(f"roc_curve not found in {model_dir}/metrics.json")
                continue

            if class_name not in roc_curves:
                continue
            
            roc_auc = metrics.get('roc_auc', {}).get('by_class', {})
            curve_data = roc_curves[class_name]
            fpr = curve_data.get('fpr', [])
            tpr = curve_data.get('tpr', [])
            auc = roc_auc.get(class_name, np.nan)
            if not fpr or not tpr:
                continue
            auc_label = f'{auc:.3f}' if not pd.isna(auc) else 'n/a'
            ax.plot(fpr, tpr, color=colors[model_idx], linewidth=2, label=f'{model_name.replace("_", " ").title()} (AUC = {auc_label})')
            plotted = True

        if not plotted:
            ax.text(0.5, 0.5, f'No ROC data for {class_title}', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC = 0.5)')
            ax.legend(loc="lower right", fontsize=8)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title(f'Clase: {class_title}', fontweight='bold')
        ax.grid(alpha=0.3)

        fig.tight_layout()

        if show_plots:
            plt.show()
        if save_plots:
            plt.savefig(SAVE_PATH / f'{class_name}_roc_curve.pdf', dpi=300, bbox_inches='tight')
            print(f"Saved: {class_name}_roc_curve.pdf")

def generate_comparison_table(print_tables: bool = True, save_tables: bool = False, latex_mode: bool = False) -> None:
    table_list = {"Comparison to original model": CLASSES}

    for model_path in TEST_DIR.iterdir():
        if not model_path.is_dir():
            continue
        comparison_path = model_path / "comparison_to_original.json"
        if not comparison_path.exists():
            print(f"comparison_to_original.json not found in {model_path}")
            continue
        with open(comparison_path) as f:
            comparison = json.load(f)
        
        model_name = model_path.name
        model_title = model_name.replace('_', ' ').title()
        delong_results = comparison.get('delong_test', None)
        if not delong_results:
            print(f"delong_test not found in {model_path}/comparison_to_original.json")
            table_list[model_title] = ["N/A"] * len(CLASSES)
            continue
        delong_results = [delong_results.get(class_name, np.nan) for class_name in CLASSES]
        table_list[model_title + " (DeLong)"] = [f"{val: .4f} ({'Sí' if val < 0.05 else 'No'})" if isinstance(val, (int, float)) and np.isfinite(val) else "N/A" for val in delong_results]
        mcnemar_results = comparison.get('mcnemar_test', {})
        if not mcnemar_results:
            print(f"mcnemar_test not found in {model_path}/comparison_to_original.json")
            table_list[model_title + " (McNemar)"] = ["N/A"] * len(CLASSES)
            continue
        mcnemar_results = [mcnemar_results.get(class_name, np.nan) for class_name in CLASSES]
        table_list[model_title + " (McNemar)"] = [f"{val: .4f} ({'Sí' if val < 0.05 else 'No'})" if isinstance(val, (int, float)) and np.isfinite(val) else "N/A" for val in mcnemar_results]

        
        

    table = tabulate(table_list, headers = 'keys', tablefmt = 'psql')
    if save_tables:
        with open(OUT_DIR / 'comparison_table.txt', 'w') as f:
            f.write(table)
    if print_tables:
        print(f"\nOriginal model comparison:\n")
        print(table)
    if latex_mode:
        latex_table = tabulate(table_list, headers = 'keys', tablefmt = 'latex')
        with open(OUT_DIR / 'comparison_table.tex', 'w') as f:
            f.write(latex_table)

if __name__ == "__main__":
    generate_metrics_tables(print_tables=False, save_tables=True, latex_mode=True)
    plot_roc_curves_by_class(show_plots=False, save_plots=True)
    plot_roc_curves_by_model(show_plots=False, save_plots=True)
    generate_comparison_table(print_tables=False, save_tables=True, latex_mode=True)