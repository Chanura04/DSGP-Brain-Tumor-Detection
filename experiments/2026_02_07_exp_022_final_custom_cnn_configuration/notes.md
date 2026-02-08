## Experiment: 

### optimizer = RMSprop
### learning rate = 3e-4

These results indicate that the learning rate (3e-4) is better matched to this optimizer, compared to the learning rate 1e-4, leading to better optimization and convergence.

Pretrained architectures were fine-tuned using commonly recommended optimization settings rather than extensive hyperparameter sweeps. The goal was to provide a fair reference comparison rather than to fully optimize each pretrained model.

The focus of this work is the design and optimization of a custom CNN for MRI tumor classification. Pretrained models are included as reference baselines using standard fine-tuning settings to contextualize performance.

Among all evaluated configurations, RMSprop with a learning rate of 3e-4 achieved the highest validation accuracy for the custom CNN, while Adam with higher learning rates converged faster but showed increased instability. SGD required careful tuning and diverged at higher learning rates. These results demonstrate that optimizer and learning rate selection significantly impact MRI tumor classification performance.
