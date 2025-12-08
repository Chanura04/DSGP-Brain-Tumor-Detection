# EXPERIMENTS FOLDER

### Purpose:
Each **subfolder** represents **one training experiment**.

## Each experiment may contain (metadata):
- config.yaml : Hyperparameters and settings.
- metrics.json : Evaluation results.
- training_logs.txt : Logs from training.
- predictions.csv : Model predictions.
- plots/ : Curves, graphs, confusion matrices.
- grid search logs.
- test runs.
- evaluation metrics.

### Naming Convention:
- YYYY-MM-DD_model_description
- exp_001_baseline/
- exp_002_tuned_model/

### Example:
- 2025-02-05_bert_lr1e-5