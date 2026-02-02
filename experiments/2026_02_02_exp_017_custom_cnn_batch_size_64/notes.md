## Experiment: Learning Rate Ablation

All experiments use identical architecture, dataset, seed (42), and training
procedure. lr was changed for a fair comparison.

### batch size = 64
- Faster convergence
- Validation loss unstable after epoch 4
- Signs of sharp minima / overfitting
- high gradient noise
- larger variance in updates
- loss spikes
- accuracy jumps
