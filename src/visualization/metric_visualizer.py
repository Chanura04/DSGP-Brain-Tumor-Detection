from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
)
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
import pandas as pd

from typing import Tuple
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import src.visualization.config as cfg
from src.visualization.base_visualizer import BaseVisualizer


class MetricVisualizer(BaseVisualizer):
    def __init__(self, metrics_df, y_test, y_pred, y_pred_probs, targets):
        super().__init__()
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_probs = y_pred_probs
        self.targets = targets
        self.metrics_df = metrics_df

    def plot_confusion_matrix(
        self, normalize=True, save_path: str = ""
    ) -> Tuple[Figure, Axes]:
        if normalize:
            normalize_plot = "true"
            fmt = ".1%"
            vmin, vmax = 0, 1
            title = "Confusion Matrix (Normalized)"
        else:
            normalize_plot = None
            fmt = "d"
            vmin, vmax = None, None
            title = "Confusion Matrix"

        cm = confusion_matrix(self.y_test, self.y_pred, normalize=normalize_plot)

        fig: Figure
        ax: Axes
        fig, ax = self._create_fig_ax()

        sns.set_theme(style="white", font_scale=1.2)

        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            ax=ax,
            cbar=False,
            linewidths=0.8,
            linecolor="gray",
            vmin=vmin,
            vmax=vmax,
            xticklabels=self.targets,
            yticklabels=self.targets,
            annot_kws={"size": 14, "weight": "bold"},
        )

        ax.set_title(title, fontsize=cfg.TITLE_FONTSIZE, weight=cfg.TITLE_WEIGHT)
        ax.set_xlabel("Predicted Label", fontsize=cfg.AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("True Label", fontsize=cfg.AXIS_LABEL_FONTSIZE)

        self._save_figure(fig, save_path)

        return fig, ax

    def plot_roc_auc(self, save_path: str = "") -> Tuple[Figure, Axes]:
        y_score = self.y_pred_probs[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_score)
        roc_auc = auc(fpr, tpr)

        fig: Figure
        ax: Axes
        fig, ax = self._create_fig_ax()

        ax.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
        )
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")

        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))
        ax.set_title("ROC Curve", fontsize=cfg.TITLE_FONTSIZE, weight=cfg.TITLE_WEIGHT)
        ax.set_xlabel("False Positive Rate", fontsize=cfg.AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("True Positive Rate", fontsize=cfg.AXIS_LABEL_FONTSIZE)
        ax.legend(
            frameon=cfg.LEGEND_FRAME,
            bbox_to_anchor=cfg.LEGEND_BBOX_TO_ANCHOR,
            loc=cfg.LEGEND_LOCATION,
        )
        ax.grid(alpha=cfg.GRID_ALPHA)

        self._save_figure(fig, save_path)

        return fig, ax

    def plot_roc_auc_multiclass(self, save_path: str = "") -> Tuple[Figure, Axes]:
        n_classes = self.y_pred_probs.shape[1]

        # binarize the labels for one-vs-rest ROC
        y_test_bin = label_binarize(self.y_test, classes=np.arange(n_classes))

        fig: Figure
        ax: Axes
        fig, ax = self._create_fig_ax()

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], self.y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"{self.targets[i]} (AUC = {roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))
        ax.set_title(
            "Multiclass ROC Curve", fontsize=cfg.TITLE_FONTSIZE, weight=cfg.TITLE_WEIGHT
        )
        ax.set_xlabel("False Positive Rate", fontsize=cfg.AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("True Positive Rate", fontsize=cfg.AXIS_LABEL_FONTSIZE)
        ax.legend(
            frameon=cfg.LEGEND_FRAME,
            bbox_to_anchor=cfg.LEGEND_BBOX_TO_ANCHOR,
            loc=cfg.LEGEND_LOCATION,
        )
        ax.grid(alpha=cfg.GRID_ALPHA)

        self._save_figure(fig, save_path)

        return fig, ax

    def plot_precision_vs_recall(self, save_path: str = "") -> Tuple[Figure, Axes]:
        y_score = self.y_pred_probs[:, 1]
        precision, recall, _ = precision_recall_curve(self.y_test, y_score)
        ap = average_precision_score(self.y_test, y_score)

        fig: Figure
        ax: Axes
        fig, ax = self._create_fig_ax()

        ax.plot(recall, precision, lw=2, label=f"PR curve (AP={ap:.3f})")
        ax.fill_between(recall, precision, alpha=0.2)
        ax.set_title(
            "Precision–Recall Curve",
            fontsize=cfg.TITLE_FONTSIZE,
            weight=cfg.TITLE_WEIGHT,
        )
        ax.set_xlabel("Recall", fontsize=cfg.AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Precision", fontsize=cfg.AXIS_LABEL_FONTSIZE)
        ax.legend(
            frameon=cfg.LEGEND_FRAME,
            bbox_to_anchor=cfg.LEGEND_BBOX_TO_ANCHOR,
            loc=cfg.LEGEND_LOCATION,
        )
        ax.grid(alpha=cfg.GRID_ALPHA)

        self._save_figure(fig, save_path)

        return fig, ax

    def plot_precision_vs_recall_multiclass(
        self, save_path: str = ""
    ) -> Tuple[Figure, Axes]:
        n_classes = self.y_pred_probs.shape[1]

        # binarize the labels for one-vs-rest ROC
        y_test_bin = label_binarize(self.y_test, classes=np.arange(n_classes))

        fig: Figure
        ax: Axes
        fig, ax = self._create_fig_ax()

        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(
                y_test_bin[:, i], self.y_pred_probs[:, i]
            )
            ap = average_precision_score(y_test_bin[:, i], self.y_pred_probs[:, i])
            ax.plot(recall, precision, lw=2, label=f"{self.targets[i]} (AP={ap:.3f})")
            ax.fill_between(recall, precision, alpha=0.2)

        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))
        ax.set_title(
            "Multiclass Precision–Recall Curve",
            fontsize=cfg.TITLE_FONTSIZE,
            weight=cfg.TITLE_WEIGHT,
        )
        ax.set_xlabel("Recall", fontsize=cfg.AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Precision", fontsize=cfg.AXIS_LABEL_FONTSIZE)
        ax.legend(
            frameon=cfg.LEGEND_FRAME,
            bbox_to_anchor=cfg.LEGEND_BBOX_TO_ANCHOR,
            loc=cfg.LEGEND_LOCATION,
        )
        ax.grid(alpha=cfg.GRID_ALPHA)

        self._save_figure(fig, save_path)

        return fig, ax

    def plot_metrics_vs_threshold(self, save_path: str = "") -> Tuple[Figure, Axes]:
        # Use probability of the positive class
        y_score = self.y_pred_probs[:, 1]

        thresholds = np.linspace(0, 1, 101)
        f1_scores = []
        precision_scores = []
        recall_scores = []
        accuracy_scores = []

        for t in thresholds:
            y_pred_threshold = (y_score >= t).astype(int)
            f1_scores.append(f1_score(self.y_test, y_pred_threshold))
            precision_scores.append(precision_score(self.y_test, y_pred_threshold))
            recall_scores.append(recall_score(self.y_test, y_pred_threshold))
            accuracy_scores.append(accuracy_score(self.y_test, y_pred_threshold))

        # Plotting
        fig: Figure
        ax: Axes
        fig, ax = self._create_fig_ax()
        ax.plot(thresholds, f1_scores, label="F1 Score", lw=2, color="darkorange")
        ax.plot(thresholds, precision_scores, label="Precision", lw=2, color="green")
        ax.plot(thresholds, recall_scores, label="Recall", lw=2, color="blue")
        ax.plot(thresholds, accuracy_scores, label="Accuracy", lw=2, color="purple")

        ax.set_title(
            "Metrics vs Probability Threshold",
            fontsize=cfg.TITLE_FONTSIZE,
            weight=cfg.TITLE_WEIGHT,
        )
        ax.set_xlabel("Threshold", fontsize=cfg.AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Score", fontsize=cfg.AXIS_LABEL_FONTSIZE)
        ax.legend(
            frameon=cfg.LEGEND_FRAME,
            bbox_to_anchor=cfg.LEGEND_BBOX_TO_ANCHOR,
            loc=cfg.LEGEND_LOCATION,
        )
        ax.grid(alpha=cfg.GRID_ALPHA)
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1.05))

        self._save_figure(fig, save_path)

        return fig, ax

    def plot_metrics_vs_threshold_multiclass(
        self, save_path: str = ""
    ) -> Tuple[Figure, np.ndarray]:
        n_classes = self.y_pred_probs.shape[1]
        y_test_bin = label_binarize(self.y_test, classes=np.arange(n_classes))

        # Colors for metrics
        metric_colors = {
            "F1 Score": "darkorange",
            "Precision": "green",
            "Recall": "blue",
            "Accuracy": "purple",
        }

        # Create subplots: one row per class
        fig: Figure
        fig, axes = plt.subplots(
            n_classes, 1, figsize=(12, 4 * n_classes), constrained_layout=True
        )

        # If only one class, axes is not an array
        if n_classes == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            thresholds = np.linspace(0, 1, 101)
            f1_scores = []
            precision_scores = []
            recall_scores = []
            accuracy_scores = []

            for t in thresholds:
                y_pred = (self.y_pred_probs[:, i] >= t).astype(int)
                f1_scores.append(f1_score(y_test_bin[:, i], y_pred))
                precision_scores.append(precision_score(y_test_bin[:, i], y_pred))
                recall_scores.append(recall_score(y_test_bin[:, i], y_pred))
                accuracy_scores.append(accuracy_score(y_test_bin[:, i], y_pred))

            # Plot metrics
            ax.plot(
                thresholds,
                f1_scores,
                lw=2,
                color=metric_colors["F1 Score"],
                label="F1 Score",
            )
            ax.plot(
                thresholds,
                precision_scores,
                lw=2,
                color=metric_colors["Precision"],
                label="Precision",
            )
            ax.plot(
                thresholds,
                recall_scores,
                lw=2,
                color=metric_colors["Recall"],
                label="Recall",
            )
            ax.plot(
                thresholds,
                accuracy_scores,
                lw=2,
                color=metric_colors["Accuracy"],
                label="Accuracy",
            )

            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
            ax.set_title(
                f"Metrics vs Threshold – {self.targets[i]}",
                fontsize=cfg.TITLE_FONTSIZE,
                weight=cfg.TITLE_WEIGHT,
            )
            ax.set_xlabel("Threshold", fontsize=cfg.AXIS_LABEL_FONTSIZE)
            ax.set_ylabel("Score", fontsize=cfg.AXIS_LABEL_FONTSIZE)
            ax.legend(
                frameon=cfg.LEGEND_FRAME,
                bbox_to_anchor=cfg.LEGEND_BBOX_TO_ANCHOR,
                loc=cfg.LEGEND_LOCATION,
            )
            ax.grid(alpha=cfg.GRID_ALPHA)

        self._save_figure(fig, save_path)

        return fig, axes

    def plot_class_distribution(self, save_path: str = "") -> Tuple[Figure, Axes]:
        # Count actual class distribution
        classes = self.targets
        actual_counts = [np.sum(self.y_test == i) for i in range(len(classes))]

        # Count predicted class distribution
        predicted_labels = np.argmax(self.y_pred_probs, axis=1)
        predicted_counts = [np.sum(predicted_labels == i) for i in range(len(classes))]

        x = np.arange(len(classes))  # Class positions

        width = 0.35  # Bar width

        fig, ax = self._create_fig_ax()

        ax.bar(x - width / 2, actual_counts, width, label="Actual", color="skyblue")
        ax.bar(
            x + width / 2, predicted_counts, width, label="Predicted", color="orange"
        )

        # Labels
        ax.set_title(
            "Class Distribution: Actual vs Predicted",
            fontsize=cfg.TITLE_FONTSIZE,
            weight=cfg.TITLE_WEIGHT,
        )
        ax.set_ylabel("Number of Samples", fontsize=cfg.AXIS_LABEL_FONTSIZE)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, fontsize=cfg.AXIS_LABEL_FONTSIZE)
        ax.legend(
            frameon=cfg.LEGEND_FRAME,
            bbox_to_anchor=cfg.LEGEND_BBOX_TO_ANCHOR,
            loc=cfg.LEGEND_LOCATION,
        )
        ax.grid(axis=cfg.GRID_AXIS, alpha=cfg.GRID_ALPHA)

        for i, (a, p) in enumerate(zip(actual_counts, predicted_counts)):
            ax.text(
                i - width / 2,
                a + 0.5,
                str(a),
                ha=cfg.ANNOTATE_HA,
                va=cfg.ANNOTATE_VA,
                fontsize=cfg.ANNOTATE_FONTSIZE,
            )
            ax.text(
                i + width / 2,
                p + 0.5,
                str(p),
                ha=cfg.ANNOTATE_HA,
                va=cfg.ANNOTATE_VA,
                fontsize=cfg.ANNOTATE_FONTSIZE,
            )

        self._save_figure(fig, save_path)

        return fig, ax

    def plot_probability_histogram(self, save_path: str = "") -> Tuple[Figure, Axes]:
        y_score = self.y_pred_probs[:, 1]

        fig: Figure
        ax: Axes
        fig, ax = self._create_fig_ax()

        colors = ["skyblue", "orange", "green"]
        for i, tar in enumerate(self.targets):
            ax.hist(
                y_score[self.y_test == i],
                bins=20,
                alpha=0.6,
                label=tar,
                color=colors[i],
                edgecolor="k",
            )

        ax.set_title(
            "Probability Histogram",
            fontsize=cfg.TITLE_FONTSIZE,
            weight=cfg.TITLE_WEIGHT,
        )
        ax.set_xlabel("Predicted Probability", fontsize=cfg.AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Number of Samples", fontsize=cfg.AXIS_LABEL_FONTSIZE)
        ax.legend(
            frameon=cfg.LEGEND_FRAME,
            bbox_to_anchor=cfg.LEGEND_BBOX_TO_ANCHOR,
            loc=cfg.LEGEND_LOCATION,
        )
        ax.grid(alpha=cfg.GRID_ALPHA)

        self._save_figure(fig, save_path)

        return fig, ax

    def plot_calibration_curve(
        self, n_bins=10, save_path: str = ""
    ) -> Tuple[Figure, Axes]:
        y_score = self.y_pred_probs[:, 1]

        prob_true, prob_pred = calibration_curve(self.y_test, y_score, n_bins=n_bins)

        fig: Figure
        ax: Axes
        fig, ax = self._create_fig_ax()

        ax.plot(prob_pred, prob_true, marker="o", label="Model", color="darkorange")
        ax.plot(
            [0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated"
        )

        ax.set_title(
            "Calibration Curve (Reliability Diagram)",
            fontsize=cfg.TITLE_FONTSIZE,
            weight=cfg.TITLE_WEIGHT,
        )
        ax.set_xlabel("Mean Predicted Probability", fontsize=cfg.AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Fraction of Positives", fontsize=cfg.AXIS_LABEL_FONTSIZE)
        ax.legend(
            frameon=cfg.LEGEND_FRAME,
            bbox_to_anchor=cfg.LEGEND_BBOX_TO_ANCHOR,
            loc=cfg.LEGEND_LOCATION,
        )
        ax.grid(alpha=cfg.GRID_ALPHA)

        self._save_figure(fig, save_path)

        return fig, ax

    def plot_calibration_curve_multiclass(
        self, n_bins=10, save_path: str = ""
    ) -> Tuple[Figure, Axes]:
        n_classes = self.y_pred_probs.shape[1]
        y_test_bin = label_binarize(self.y_test, classes=np.arange(n_classes))

        fig: Figure
        ax: Axes
        fig, ax = self._create_fig_ax()

        for i in range(n_classes):
            prob_true, prob_pred = calibration_curve(
                y_test_bin[:, i], self.y_pred_probs[:, i], n_bins=n_bins
            )
            ax.plot(prob_pred, prob_true, marker="o", color="darkorange")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")

        ax.set_title(
            "Calibration Curve (Reliability Diagram)",
            fontsize=cfg.TITLE_FONTSIZE,
            weight=cfg.TITLE_WEIGHT,
        )
        ax.set_xlabel("Mean Predicted Probability", fontsize=cfg.AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Fraction of Positives", fontsize=cfg.AXIS_LABEL_FONTSIZE)
        ax.legend(
            labels=["Model", "Perfectly calibrated"],
            frameon=cfg.LEGEND_FRAME,
            bbox_to_anchor=cfg.LEGEND_BBOX_TO_ANCHOR,
            loc=cfg.LEGEND_LOCATION,
        )
        ax.grid(alpha=cfg.GRID_ALPHA)

        self._save_figure(fig, save_path)

        return fig, ax

    def plot_loss_curve(self, save_path: str = "") -> Tuple[Figure, Axes]:
        fig: Figure
        ax: Axes
        fig, ax = self._create_fig_ax()

        ax.plot(
            self.metrics_df["epoch"], self.metrics_df["train_loss"], label="Train Loss"
        )
        ax.plot(self.metrics_df["epoch"], self.metrics_df["val_loss"], label="Val Loss")

        ax.set_title(
            "Train / Val Loss Curve",
            fontsize=cfg.TITLE_FONTSIZE,
            weight=cfg.TITLE_WEIGHT,
        )
        ax.set_xlabel("Epochs", fontsize=cfg.AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Loss", fontsize=cfg.AXIS_LABEL_FONTSIZE)
        ax.legend(
            frameon=cfg.LEGEND_FRAME,
            bbox_to_anchor=cfg.LEGEND_BBOX_TO_ANCHOR,
            loc=cfg.LEGEND_LOCATION,
        )
        ax.grid(alpha=cfg.GRID_ALPHA)

        self._save_figure(fig, save_path)

        return fig, ax

    def plot_acc_curve(self, save_path: str = "") -> Tuple[Figure, Axes]:
        fig: Figure
        ax: Axes
        fig, ax = self._create_fig_ax()

        ax.plot(
            self.metrics_df["epoch"],
            self.metrics_df["train_accuracy"],
            label="Train Accuracy",
        )
        ax.plot(
            self.metrics_df["epoch"],
            self.metrics_df["val_accuracy"],
            label="Val Accuracy",
        )

        ax.set_title(
            "Train / Val Accuracy Curve",
            fontsize=cfg.TITLE_FONTSIZE,
            weight=cfg.TITLE_WEIGHT,
        )
        ax.set_xlabel("Epochs", fontsize=cfg.AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Accuracy", fontsize=cfg.AXIS_LABEL_FONTSIZE)
        ax.legend(
            frameon=cfg.LEGEND_FRAME,
            bbox_to_anchor=cfg.LEGEND_BBOX_TO_ANCHOR,
            loc=cfg.LEGEND_LOCATION,
        )
        ax.grid(alpha=cfg.GRID_ALPHA)

        self._save_figure(fig, save_path)

        return fig, ax

    def show_classification_report(self, save_path: str = ""):
        report_dict = classification_report(
            self.y_test, self.y_pred, target_names=self.targets, output_dict=True
        )
        df_report = pd.DataFrame(report_dict).T

        if save_path:
            df_report.to_csv(save_path)
        else:
            print(df_report)
