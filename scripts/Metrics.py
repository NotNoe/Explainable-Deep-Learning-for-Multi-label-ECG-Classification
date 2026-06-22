import numpy as np
import json
from scipy import stats
from sklearn.metrics import (precision_recall_fscore_support, fbeta_score,
                             roc_auc_score, roc_curve)
from statsmodels.stats.contingency_tables import mcnemar
from MLstatkit import Delong_test

class Metrics:
    def __init__(self, y_true, y_pred, class_names=None):
        """
        Inits Metrics class with true labels and predictions matrix.

        Parameters:
        y_true (np.ndarray): Real values array (probabilities between 0 and 1).
                             Dimension: (n_samples, n_classes)
        y_pred (np.ndarray): Array of predictions (probabilities between 0 and 1).
                             Dimension: (n_samples, n_classes)
        class_names (list, optional): List with the names of the classes.
                                      Length: n_classes
        """
        self.y_true_prob = y_true
        self.y_pred_prob = y_pred
        self.class_names = class_names
        self.n_classes = y_true.shape[1]

        # Make labels binary using a threshold of 0.5
        self.y_true = (self.y_true_prob >= 0.5).astype(int)
        self.y_pred = (self.y_pred_prob >= 0.5).astype(int)

        # Check if any class has no representatives in y_true
        self.classes_with_no_samples = []
        for i in range(self.n_classes):
            y_true_sum = np.sum(self.y_true[:, i])
            if y_true_sum == 0:
                if self.class_names:
                    class_name = self.class_names[i]
                else:
                    class_name = f'Class_{i}'
                self.classes_with_no_samples.append(class_name)

        if self.classes_with_no_samples:
            print(f"Warning: The following classes have no representatives in the test sample: {', '.join(self.classes_with_no_samples)}")

    def calculate_precision_recall_f1(self):
        """
        Calculates precision, recall and F1-score for each class and globally.

        Returns:
        dict: Dictionary with the calculated metrics.
        """
        precision_dict = {}
        recall_dict = {}
        f1_dict = {}
        
        precision_list, recall_list, f1_list, _ = precision_recall_fscore_support(self.y_true, self.y_pred, zero_division=np.nan)
        for i in range(self.n_classes): #Para cada clase
            class_name = self.class_names[i] if self.class_names else f'Class_{i}'
            precision_dict[class_name] = precision_list[i]
            recall_dict[class_name] = recall_list[i]
            f1_dict[class_name] = f1_list[i]


        global_precision, global_recall, global_f1, _ = precision_recall_fscore_support(self.y_true, self.y_pred, average='micro', zero_division=np.nan)

        metrics = {
            'precision': {
                'by_class': precision_dict,
                'global_average': global_precision
            },
            'recall': {
                'by_class': recall_dict,
                'global_average': global_recall
            },
            'f1_score': {
                'by_class': f1_dict,
                'global_average': global_f1
            }
        }

        return metrics

    def calculate_adjusted_f_score(self):
        """
        Calculates the custom metric 'adjusted_f_score' by combining F0.5 for 'NORM' and F2 for the other classes.

        Returns:
        float or None: Value of the custom metric.
        """
        if self.class_names is None:
            special_class = 3
            print("Warning: No class names provided. It will be assumed that the NORM class is the third class.")
        else:
            special_class = self.class_names.index('NORM')
        metrics = []
        for cls in range(self.n_classes):
            beta = 0.5 if cls == special_class else 2
            f_score = fbeta_score(self.y_true[:, cls], self.y_pred[:, cls], beta=beta)
            metrics.append(f_score)
        adjusted_f_score = np.mean(metrics)

        return adjusted_f_score
    
    def calculate_roc_auc(self):
        """
        Calculates the ROC AUC for each class and the micro-average.

        Returns:
        dict: Dictionary with AUC ROC by class and micro-average.
        """
        roc_auc_dict = {}
        for i in range(self.n_classes):
            class_name = self.class_names[i] if self.class_names else f'Class_{i}'
            if np.unique(self.y_true[:, i]).size < 2:
                roc_auc_dict[class_name] = np.nan
            else:
                roc_auc_dict[class_name] = roc_auc_score(self.y_true[:, i], self.y_pred_prob[:, i])

        micro_auc = roc_auc_score(self.y_true, self.y_pred_prob, average='micro')
        return {
            'by_class': roc_auc_dict,
            'global_average': micro_auc
        }

    def bootstrap_confidence_intervals(self, n_bootstrap=200, alpha=0.05, random_state=42):
        """
        Estimates confidence intervals by bootstrap for AUC ROC and basic metrics.

        Parameters:
        n_bootstrap (int): Number of resamples.
        alpha (float): Significance level (0.05 => 95% CI).
        random_state (int): Seed for reproducibility.

        Returns:
        dict: Confidence intervals of the main indicators.
        """
        if n_bootstrap <= 0:
            return {}

        rng = np.random.default_rng(random_state)
        n_samples = self.y_true.shape[0]

        class_names = [self.class_names[i] if self.class_names else f'Class_{i}'
                       for i in range(self.n_classes)]

        ci_store = {
            'precision': {name: [] for name in class_names},
            'recall': {name: [] for name in class_names},
            'f1_score': {name: [] for name in class_names},
            'roc_auc': {name: [] for name in class_names},
            'precision_global': [],
            'recall_global': [],
            'f1_score_global': [],
            'roc_auc_global': []
        }

        for _ in range(n_bootstrap):
            sample_idx = rng.choice(n_samples, size=n_samples, replace=True)
            y_true_b = self.y_true[sample_idx]
            y_pred_b = self.y_pred[sample_idx]
            y_pred_prob_b = self.y_pred_prob[sample_idx]

            precision_list, recall_list, f1_list, _ = precision_recall_fscore_support(
                y_true_b, y_pred_b, zero_division=np.nan)
            for i, class_name in enumerate(class_names):
                ci_store['precision'][class_name].append(precision_list[i])
                ci_store['recall'][class_name].append(recall_list[i])
                ci_store['f1_score'][class_name].append(f1_list[i])

            global_precision, global_recall, global_f1, _ = precision_recall_fscore_support(
                y_true_b, y_pred_b, average='micro', zero_division=np.nan)
            ci_store['precision_global'].append(global_precision)
            ci_store['recall_global'].append(global_recall)
            ci_store['f1_score_global'].append(global_f1)

            for i, class_name in enumerate(class_names):
                if np.unique(y_true_b[:, i]).size < 2:
                    ci_store['roc_auc'][class_name].append(np.nan)
                else:
                    ci_store['roc_auc'][class_name].append(
                        roc_auc_score(y_true_b[:, i], y_pred_prob_b[:, i]))

            try:
                micro_auc = roc_auc_score(y_true_b, y_pred_prob_b, average='micro')
            except ValueError:
                micro_auc = np.nan
            ci_store['roc_auc_global'].append(micro_auc)

        def _percentile(values):
            return [float(np.nanpercentile(values, 100 * alpha / 2)),
                    float(np.nanpercentile(values, 100 * (1 - alpha / 2)))]

        result = {
            'precision': {name: _percentile(ci_store['precision'][name]) for name in class_names},
            'recall': {name: _percentile(ci_store['recall'][name]) for name in class_names},
            'f1_score': {name: _percentile(ci_store['f1_score'][name]) for name in class_names},
            'roc_auc': {name: _percentile(ci_store['roc_auc'][name]) for name in class_names},
            'precision_global': _percentile(ci_store['precision_global']),
            'recall_global': _percentile(ci_store['recall_global']),
            'f1_score_global': _percentile(ci_store['f1_score_global']),
            'roc_auc_global': _percentile(ci_store['roc_auc_global'])
        }
        return result

    def get_roc_curve_data(self):
        """
        Calculates the data for the ROC curves for each class and for the micro-average.

        Returns:
        dict: Structure with the arrays fpr, tpr and thresholds for each curve.
        """
        roc_curve_dict = {}
        class_names = [self.class_names[i] if self.class_names else f'Class_{i}'
                       for i in range(self.n_classes)]

        for i, class_name in enumerate(class_names):
            if np.unique(self.y_true[:, i]).size < 2:
                roc_curve_dict[class_name] = {
                    'fpr': [],
                    'tpr': [],
                    'thresholds': []
                }
                continue

            fpr, tpr, thresholds = roc_curve(self.y_true[:, i], self.y_pred_prob[:, i])
            roc_curve_dict[class_name] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            }

        # Micro-average ROC curve (flatten all classes)
        try:
            fpr_micro, tpr_micro, thresholds_micro = roc_curve(
                self.y_true.ravel(), self.y_pred_prob.ravel())
            roc_curve_dict['micro'] = {
                'fpr': fpr_micro.tolist(),
                'tpr': tpr_micro.tolist(),
                'thresholds': thresholds_micro.tolist()
            }
        except ValueError:
            roc_curve_dict['micro'] = {
                'fpr': [],
                'tpr': [],
                'thresholds': []
            }

        return roc_curve_dict

    def dump_to_json(self, path, bootstrap_samples=200, confidence_level=0.95):
        """
        Writes the calculated metrics to a JSON file.

        Parameters:
        path (str): Path to the output JSON file.
        bootstrap_samples (int): Number of bootstrap resamples for confidence intervals.
        confidence_level (float): Confidence level.
        """
        metrics = {}
        adjusted_f_score = self.calculate_adjusted_f_score()
        metrics['adjusted_f_score'] = adjusted_f_score

        prf_metrics = self.calculate_precision_recall_f1()
        metrics.update(prf_metrics)

        roc_metrics = self.calculate_roc_auc()
        metrics['roc_auc'] = roc_metrics
        metrics['roc_curve'] = self.get_roc_curve_data()

        if bootstrap_samples > 0:
            ci_metrics = self.bootstrap_confidence_intervals(
                n_bootstrap=bootstrap_samples,
                alpha=1 - confidence_level)
            metrics['confidence_intervals'] = ci_metrics

        with open(path, 'w') as json_file:
            json.dump(metrics, json_file, indent=4)


class ComparisonMetrics:
    def __init__(self, y_true, y_pred_a, y_pred_b, class_names=None):
        """
        Inits Metrics class with true labels and predictions matrix.

        Parameters:
        y_true (np.ndarray): Real values array (probabilities between 0 and 1).
                             Dimension: (n_samples, n_classes)
        y_pred (np.ndarray): Predictions array (probabilities between 0 and 1).
                             Dimension: (n_samples, n_classes)
        class_names (list, optional): List with the names of the classes.
                                      Length: n_classes
        """
        self.y_true_prob = y_true
        self.y_pred_a_prob = y_pred_a
        self.y_pred_b_prob = y_pred_b
        self.class_names = class_names
        self.n_classes = y_true.shape[1]

        # Make labels binary using a threshold of 0.5
        self.y_true = (self.y_true_prob >= 0.5).astype(int)
        self.y_pred_a = (self.y_pred_a_prob >= 0.5).astype(int)
        self.y_pred_b = (self.y_pred_b_prob >= 0.5).astype(int)

        # Verify if any class has no representatives in y_true
        self.classes_with_no_samples = []
        for i in range(self.n_classes):
            y_true_sum = np.sum(self.y_true[:, i])
            if y_true_sum == 0:
                if self.class_names:
                    class_name = self.class_names[i]
                else:
                    class_name = f'Class_{i}'
                self.classes_with_no_samples.append(class_name)

        if self.classes_with_no_samples:
            print(f"Warning: The following classes have no representatives in the test sample: {', '.join(self.classes_with_no_samples)}")

    def delong_test(self):
        """
        Performs the DeLong test to compare ROC AUC between model A and B by class using bootstrap.

        Returns:
        dict: Per class P-values for the AUC difference (A vs B).
        """
        delong_results = {}
        class_names = [self.class_names[i] if self.class_names else f'Class_{i}'
                       for i in range(self.n_classes)]

        for i, class_name in enumerate(class_names):
            if np.unique(self.y_true[:, i]).size < 2:
                delong_results[class_name] = np.nan
                continue

            y_true_class = self.y_true[:, i]
            y_pred_a_class = self.y_pred_a_prob[:, i]
            y_pred_b_class = self.y_pred_b_prob[:, i]

            _, p_value = Delong_test(y_true_class, y_pred_a_class, y_pred_b_class, return_auc=False, return_ci=False)

            delong_results[class_name] = float(p_value)

        return delong_results

    def mcnemar_test(self):
        """
        Performs the McNemar test to compare binary predictions between model A and B by class using scipy.stats.mcnemar.

        Returns:
        dict: P-values by class for the prediction difference.
        """
        mcnemar_results = {}
        class_names = [self.class_names[i] if self.class_names else f'Class_{i}'
                       for i in range(self.n_classes)]

        for i, class_name in enumerate(class_names):
            if np.unique(self.y_true[:, i]).size < 2:
                mcnemar_results[class_name] = np.nan
                continue

            y_true_class = self.y_true[:, i]
            y_pred_a_class = self.y_pred_a[:, i]
            y_pred_b_class = self.y_pred_b[:, i]

            # Construct complete contingency table for McNemar
            a = np.sum((y_pred_a_class == y_true_class) & (y_pred_b_class == y_true_class))
            b = np.sum((y_pred_a_class == y_true_class) & (y_pred_b_class != y_true_class))
            c = np.sum((y_pred_a_class != y_true_class) & (y_pred_b_class == y_true_class))
            d = np.sum((y_pred_a_class != y_true_class) & (y_pred_b_class != y_true_class))

            table = np.array([[a, b], [c, d]])

            # Verify if there are sufficient data for the test
            if b + c == 0:
                mcnemar_results[class_name] = 1.0  # No difference
            else:
                result = mcnemar(table, exact=False, correction=True)
                mcnemar_results[class_name] = float(result.pvalue)

        return mcnemar_results

    def dump_to_json(self, path):
        """
        Writes the comparison results to a JSON file.

        Parameters:
        path (str): Path to the output JSON file.
        """
        comparison_results = {
            'delong_test': self.delong_test(),
            'mcnemar_test': self.mcnemar_test()
        }

        with open(path, 'w') as json_file:
            json.dump(comparison_results, json_file, indent=4)