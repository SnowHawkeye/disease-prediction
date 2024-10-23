import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from mipha.framework import MachineLearningModel, Evaluator
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    cohen_kappa_score,
    log_loss, precision_recall_curve,
)


class ClassificationEvaluator(Evaluator):

    def __init__(self, component_name: str = None):
        super().__init__(component_name)
        self.metrics = None

    def evaluate_model(self, model: MachineLearningModel, x_test, y_test, threshold: float = 0.5, *args, **kwargs):
        """
        Evaluate the performance of a trained machine learning model on test data.

        :param model: The trained machine learning model to evaluate.
        :type model: MachineLearningModel
        :param x_test: The input features for the test set.
        :type x_test: np.ndarray or pd.DataFrame
        :param y_test: The true labels for the test set.
        :type y_test: np.ndarray or pd.Series
        :param threshold: The threshold for binary classification to determine predicted classes. Defaults to 0.5.
        :type threshold: float
        :return: A dictionary containing evaluation metrics.
        """

        # Get predictions from the model
        y_pred = model.predict(x_test)

        # Determine predicted classes based on the number of classes

        is_binary_classification = y_pred.ndim == 1 or y_pred.shape[1] == 1

        if is_binary_classification:
            y_pred_classes = (y_pred > threshold).astype(int)
            average = 'binary' # parameter in some metrics
        else:  # Multi-class case
            y_pred_classes = y_pred.argmax(axis=1)
            average = 'weighted'

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred_classes),
            "f1_score": f1_score(y_test, y_pred_classes, average=average),
            "precision": precision_score(y_test, y_pred_classes, average=average),
            "recall": recall_score(y_test, y_pred_classes, average=average),
            "mcc": matthews_corrcoef(y_test, y_pred_classes),
            "kappa": cohen_kappa_score(y_test, y_pred_classes),
            "log_loss": log_loss(y_test, y_pred),  # log loss is computed with probabilities
            "classification_report": classification_report(y_test, y_pred_classes, output_dict=True),
            "normalized_confusion_matrix": confusion_matrix(y_test, y_pred_classes, normalize="true").tolist(),
            "raw_confusion_matrix": confusion_matrix(y_test, y_pred_classes, normalize=None).tolist(),
        }

        # ROC-AUC handling
        if len(set(y_test)) > 2:  # More than two classes
            metrics["roc_auc"] = "N/A"
        else:
            metrics["roc_auc"] = roc_auc_score(y_test, y_pred_classes)

        # Include classification report in metrics
        metrics["classification_report"] = classification_report(y_test, y_pred_classes, output_dict=True)

        # Include precision-recall curve in metrics
        if len(set(y_test)) > 2:  # More than two classes
            metrics["precision_recall_curve"] = "N/A"
        else:
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred, drop_intermediate=True)
            metrics["precision_recall_curve"] = {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": thresholds.tolist(),
            }

        self.metrics = metrics
        return metrics

    def save_metrics_to_json(self, file_path: str) -> None:
        """
        Save performance metrics to a JSON file.

        :param file_path: The path to the JSON file to save metrics.
        :type file_path: str
        :return: None
        """

        if self.metrics is None:
            print("No metrics to save. Please call .evaluate_model() method first.")
        else:
            with open(file_path, 'w') as json_file:
                json.dump(self.metrics, json_file, indent=4)

    def display_results(self) -> None:
        """
        Display confusion matrix and other visualizations.
        """

        if self.metrics is None:
            print("No metrics to display. Please call .evaluate_model() method first.")
        else:
            print("CLASSIFICATION REPORT  ---------")
            print(pd.DataFrame(self.metrics["classification_report"]).transpose())

            print("")
            print("CONFUSION MATRIX  --------------")
            plt.figure(figsize=(15, 5))
            plt.subplot(121)
            sns.heatmap(self.metrics["normalized_confusion_matrix"], annot=True, fmt=".2f")
            plt.title('Normalized Confusion Matrix')
            plt.subplot(122)
            sns.heatmap(self.metrics["raw_confusion_matrix"], annot=True)
            plt.title('Confusion Matrix')
            plt.show()

            try:
                print("")
                print("PRECISION-RECALL CURVE ---------")
                precision, recall, thresholds = self.metrics["precision_recall_curve"].values()
                plt.figure(figsize=(10, 8))
                plt.plot(recall, precision, marker='o', label='Precision-Recall Curve')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve with Thresholds')
                plt.grid()

                # Annotating the thresholds
                for i in range(0, len(thresholds), max(1, len(thresholds) // 10)):  # Adjust the step for fewer annotations
                    plt.annotate(f'{thresholds[i]:.2f}', (recall[i], precision[i]),
                                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

                plt.legend()
                plt.show()
            except Exception as e:
                print(f"Precision-recall curve could not be displayed: {e}")
                return
