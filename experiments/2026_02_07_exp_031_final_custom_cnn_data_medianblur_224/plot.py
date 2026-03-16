import pandas as pd

from src.visualization.metric_visualizer import MetricVisualizer


df = pd.read_csv("test_predictions.csv")
metrics_df = pd.read_csv("train_val_metrics.csv")

y_test = df["y_true"].to_numpy()
y_pred = df["y_pred"].to_numpy()
y_pred_probs = df[df.columns[2:]].to_numpy()
classes = ["glioma", "meningioma", "pituitary"]

visualizer = MetricVisualizer(metrics_df=metrics_df, y_test=y_test, y_pred=y_pred, y_pred_probs=y_pred_probs, targets=classes)

visualizer.plot_confusion_matrix(save_path="plots/confusion_matrix.png")
visualizer.plot_roc_auc_multiclass(save_path="plots/roc_auc.png")
visualizer.plot_precision_vs_recall_multiclass(save_path="plots/precision_vs_recall.png")
visualizer.plot_metrics_vs_threshold_multiclass(save_path="plots/metrics_vs_threshold.png")
visualizer.plot_class_distribution(save_path="plots/class_distribution.png")
visualizer.plot_probability_histogram(save_path="plots/probability_histogram.png")
visualizer.plot_calibration_curve_multiclass(save_path="plots/calibration_curve.png")
visualizer.plot_loss_curve(save_path="plots/loss_curve.png")
visualizer.plot_acc_curve(save_path="plots/acc_curve.png")
visualizer.show_classification_report(save_path="plots/classification_report.png")
