# Time Series Prediction under Distribution Shift using Differentiable Forgetting
This repository contains the code to reproduce the experiments in the paper "Time Series Prediction under Distribution 
Shift using Differentiable Forgetting" submitted to the 
ICML 2022 PODS workshop https://sites.google.com/view/icml-2022-pods.
The paper can be found in this repo under the filename "ICML_PODS_FINAL.pdf"

## Usage
Experiments can be run using

    python main.py -c "name of config file".yaml


Config files for all experiments can be found in the folder /experiment_configs.
The parallelises the synthetic data experiments across Monte Carlo repetitions and the real data experiments across time series.
While the code runs on CPU to facilitate running the comparator methods, the PyTorch GradForecaster implementation can be easily modified to run on GPU.
