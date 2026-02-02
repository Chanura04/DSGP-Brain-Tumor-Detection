## Experiment: Optimizer Ablation

All experiments use identical architecture, dataset, seed (42), and training
procedure. lr was changed for a fair comparison.

### optimizer = SGD + momentum
- Faster convergence
- Curves are reversed (accuracy/ loss)
- Clear signs of underfitting
- Increased gradient variance
- Noticeable loss spikes during training
- Very low accuracy

These results indicate that the learning rate (1e-2) is worse to this optimizer, compared to the learning rate 5e-3 which leads to better optimization and convergence.