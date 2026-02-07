## Experiment: Dropout Ablation

All experiments use identical architecture, dataset, seed (42), and training
procedure.

### dropout = 0.5
- Faster convergence
- Validation loss becomes stable
- Accuracy curve and loss curves are stable
- Still shows signs of overfitting
- Gap between val-train is much smaller
- better regularization

These results indicate that better accuracy curves, shows a higher dropout rate (0.5) significantly improves training stability and reduces overfitting by narrowing the train–validation performance gap.
