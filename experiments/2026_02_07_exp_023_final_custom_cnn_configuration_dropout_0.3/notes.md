## Experiment: Dropout Ablation

All experiments use identical architecture, dataset, seed (42), and training
procedure.

### dropout = 0.3
- Faster convergence
- Validation loss becomes unstable after approximately epoch 3
- Still shows signs of overfitting
- Gap between val-train is smaller

These results indicate that almost similar accuracy curves for with a lower validation loss, shows the dropout layer of 0.3 has minimal effect for the predictions.
