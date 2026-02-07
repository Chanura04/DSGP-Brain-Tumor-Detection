## Experiment: Dropout Ablation

All experiments use identical architecture, dataset, seed (42), and training
procedure.

### dropout = 0.5
- Faster convergence
- Validation loss becomes stable
- Accuracy curve and loss curves are stable
- Still shows signs of overfitting
- Gap between val-train is much smaller

These results indicate that better accuracy curves, shows the dropout layer of 0.5 has a major effect for the predictions.
