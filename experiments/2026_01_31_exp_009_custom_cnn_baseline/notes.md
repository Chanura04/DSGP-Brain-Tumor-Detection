## Experiment: Learning Rate Ablation

All experiments use identical architecture, dataset, seed (42), and training
procedure. Only the learning rate was varied.

### lr = 0.0005 (baseline)
- Stable convergence
- Best validation loss at epoch 6
- Mild overfitting after epoch 6
- Best balance between convergence speed and generalization

Although the baseline configuration provides a stable reference, it underperforms compared to models trained with larger batch sizes and scaled learning rates. This indicates that the initial baseline was not optimally tuned for the dataset. Learning rate scaling plays a crucial role in improving convergence and generalization, particularly when increasing batch size.