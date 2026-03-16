## Experiment: Optimizer Ablation

All experiments use identical architecture, dataset, seed (42), and training
procedure. lr was changed for a fair comparison.

### optimizer = SGD + momentum
- Faster convergence
- Validation loss becomes unstable after approximately epoch 3
- Clear signs of overfitting
- Increased gradient variance
- Noticeable loss spikes during training
