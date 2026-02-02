## Experiment: Optimizer Ablation

All experiments use identical architecture, dataset, seed (42), and training
procedure. lr was changed for a fair comparison.

### optimizer = RMSprop
- Faster convergence
- Validation loss becomes unstable after approximately epoch 4
- Clear signs of overfitting
- Increased gradient variance
- Noticeable loss spikes during training
- Sudden jumps in validation accuracy

These results indicate that the learning rate (1e-4) is not well-matched to this optimizer, leading to unstable optimization and convergence toward sharper minima.
