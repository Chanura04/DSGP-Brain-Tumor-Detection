## Experiment: Learning Rate Ablation

All experiments use identical architecture, dataset, seed (42), and training
procedure. lr was changed for a fair comparison.

### batch size = 64
- Faster initial convergence compared to the baseline batch size (32)
- Validation loss becomes unstable after approximately epoch 4
- Clear signs of overfitting
- Increased gradient variance due to larger batch updates
- Noticeable loss spikes during training
- Sudden jumps in validation accuracy

These results indicate that the baseline learning rate is not well-matched to a larger batch size, leading to unstable optimization and convergence toward sharper minima.

# Comparison with 2026_02_02_exp_015_custom_cnn_batch_size_64 (Learning Rate = 1.4 × Baseline = 0.0007)
- Validation accuracy curves follow a similar overall trend in both experiments
- Overfitting is still present in both configurations
- The scaled learning rate results in smoother loss curves with fewer and lower-magnitude spikes
- Higher peak validation accuracy is achieved with the scaled learning rate

This suggests that learning rate scaling partially compensates for the increased batch size, improving optimization stability and model performance.
