## Experiment: Data Ablation

All experiments use identical architecture, dataset, seed (42), and training
procedure.

# Data Preprocessing = Used a multichannel input (3 channels: raw, Fast Non-Local Means Denoising (FNLMD), and Contrast Limited Adaptive Histogram Equalization (CLAHE))

- training curve is stable, train accuracy approaching very close to 100%
- validation curve has slight fluctuations, but converges nicely and quickly
- Mild overfitting is observed

Although the experiment 26 gave a stable model with high accuracy, model limitations were reached and a more complex approach was taken such as multichannel input (3-channels) for the same model, to try and increase model accuracy and stability.

These results indicate that this did not yield additional performance gains and introduced unnecessary model complexity. Therefore, CLAHE preprocessing was selected for the final system due to its superior generalization performance and lower computational complexity (Experiments 30 and 32).
