## Experiment: Learning Rate Ablation

All experiments use identical architecture, dataset, seed (42), and training
procedure. Only the learning rate was varied.

### lr = 0.001
- Faster convergence
- Validation loss unstable after epoch 3
- Signs of sharp minima / overfitting
- Requires aggressive early stopping
