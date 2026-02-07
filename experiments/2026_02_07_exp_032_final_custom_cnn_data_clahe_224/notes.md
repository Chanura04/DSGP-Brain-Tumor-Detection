## Experiment: Data Ablation

All experiments use identical architecture, dataset, seed (42), and training
procedure.

# Data Preprocessing = Contrast Limited Adaptive Histogram Equalization (CLAHE)

- training curve is stable, train accuracy approaching close to 100%
- validation curve has slight fluctuations, but converges nicely and quickly
- Shows a little overfitting

Although the experiment 26 gave a stable model with high accuracy, model limitations were reached and other data preprocessing techniques were considered for the same model, to try and increase model accuracy and stability.

These results indicate that this data preprocessing technique had a huge effect on the model's generalization and helping the model learn better patterns.
