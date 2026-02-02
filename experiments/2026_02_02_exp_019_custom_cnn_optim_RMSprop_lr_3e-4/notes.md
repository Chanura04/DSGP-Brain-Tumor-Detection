## Experiment: Optimizer Ablation

All experiments use identical architecture, dataset, seed (42), and training
procedure. lr was changed for a fair comparison.

### optimizer = RMSprop
- Faster convergence
- Validation loss becomes unstable after approximately epoch 6
- Clear signs of overfitting
- Increased gradient variance

These results indicate that the learning rate (3e-4) is better matched to this optimizer, compared to the learning rate 1e-4, leading to better optimization and convergence.
