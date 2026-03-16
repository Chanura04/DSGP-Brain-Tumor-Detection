## Experiment: Data Ablation

All experiments use identical architecture, dataset, seed (42), and training
procedure.

# Data Preprocessing = Contrast Limited Adaptive Histogram Equalization (CLAHE)

- training curve is stable, train accuracy approaching very close to 100%
- validation curve has slight fluctuations, but converges nicely and quickly
- Mild overfitting is observed

Although the experiment 26 gave a stable model with high accuracy, model limitations were reached and other data preprocessing techniques were considered for the same model, to try and increase model accuracy and stability.

These results indicate that this data preprocessing technique had a significant improvement in generalization performance of the model and helping the model learn better patterns.

# Since MRI tumor classification is contrast-driven, so the CLAHE algorithm:
- Enhances local contrast
- Makes tumor boundaries clearer
- Preserves structural information
- Reduces illumination bias across scans

