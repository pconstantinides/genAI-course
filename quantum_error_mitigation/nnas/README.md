## Physics-inpired Quantum Error Mitigation by Generative Autoregression

### Usage
In order to

```python

```

### Project structure
tree
```text
nnas
├── artifacts
│   ├── datasets/                    # Pre-generated training and test datasets
│   └── models/                      # Saved model checkpoints for all experiments
│
├── coherent_noise_dataset.py        # Generates datasets with coherent, mixed and drift noise
├── qem_dataset.py                   # Base dataset utilities and data loading
├── real_amplitudes_dataset.py       # Dataset generation for Real-Amplitudes circuits
│
├── model.py                         # Implementation of all NNAS architectures
│                                   # (Original, Dual-State, Physics-Informed, Ablations)
│
├── train_nnas.py                    # Main training script
├── run_experiment.py                # Reproduces the experiments and evaluation results
│
├── error_vs_depth*.png               # Main performance comparison figures
└── results_table.txt                # Numerical evaluation results
```


### Project submition
For the purposes of the Generative AI course I suggest watching the video presentation, as it is more distilled and tailored to the background of the viewer.
The written report is more time and attention demanding.

### Code generation
I give some credit to `Claude Sonnet 5` for generating code and debugging parts of the code where I use pyTorch's API. Also the same model was used to fill in documentation and to refactor the training pipeline.
While developing the code, Github Copilot was enabled, assisting with minor code completion and documentations.

### Hardware
No access to GPU.

>**Caveat**: this is work in progress and it is not to be shared outside the courses universe.
