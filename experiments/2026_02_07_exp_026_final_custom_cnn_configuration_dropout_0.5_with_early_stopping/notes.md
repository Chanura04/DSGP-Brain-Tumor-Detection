## Experiment: Early Stopping

All experiments use identical architecture, dataset, seed (42), and training
procedure.

### Early Stopping
- Showed spike around epoch 22 after best epoch
- Training loss continues to decrease smoothly
- Model slightly overfitting
- Poor generalization for a mini-batch
- Training accuracy and loss curve is good and stable

To ensure that the model was not undertrained, the maximum number of epochs was increased to 50. Early stopping based on validation loss was applied to automatically select the optimal training duration.

After epoch 18, validation loss exhibited a sharp increase while training loss continued to decrease, indicating the onset of overfitting. Early stopping successfully selected the optimal model before this degradation.

These results indicate that the model became too confident on a few wrong samples.
