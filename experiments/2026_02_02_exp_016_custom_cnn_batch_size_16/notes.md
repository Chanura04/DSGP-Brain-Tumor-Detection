## Experiment: Learning Rate Ablation

All experiments use identical architecture, dataset, seed (42), and training
procedure. lr was changed for a fair comparison.

Small batch → very noisy gradient
Large batch → very smooth gradient

The learning rate controls how big a step you take using that gradient.

### batch size = 16
- Slower convergence
- Validation loss unstable after epoch 3
- Signs of sharp minima / overfitting
