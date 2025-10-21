# Multi-Objective Optimization Experiments for the W-EAM System

This repository contains the procedures to replicate the experiments presented in the paper:

**"On the Multi-objective Hyperparameter Optimization of the Weighted Entropic Associative Memory"**

**Authors:**  
- Juan Antonio Lopez Rivera  
- Carlos Ignacio Hernandez Castellanos  
- Rafael Morales  
- Luis A. Pineda Cortes  

The code was written in **Python 3**, using the **Anaconda Distribution**, and was run on two computers:

**Desktop computer:**  
- CPU: Intel Core i7-6700 @ 3.40 GHz  
- GPU: Nvidia GeForce GTX 1080  
- OS: Ubuntu 16.04 Xenial  
- RAM: 64 GB  

**Server:**  
- CPU: 1 out of 150 Xeon Gold cores (22 cores total)  
- GPU: NVIDIA Tesla P100  

---

## Requirements

The following Python libraries need to be installed beforehand:

- numpy==1.23.4
- pandas==1.5.1
- scikit-learn==1.1.3
- scipy==1.9.3
- matplotlib==3.6.2
- tensorflow==2.10.1
- keras==2.10.0
- smac==1.4.0
- optproblems==1.3
- evoalgos==1.1
- theano==1.0.5
- extra-keras-datasets==1.2.0
- pillow==9.3.0
- requests==2.28.1

> **Note:** This set of experiments is intended to work on top of a pre-existing W-EAM system.  

Three versions of W-EAM were used:

- **MNIST:** https://github.com/eam-experiments/MNIST  
- **EMNIST:** https://github.com/eam-experiments/EMNIST  
- **DIMEX:** https://github.com/eam-experiments/dimex  

---

## Usage

Scripts to run **SMS-EMOA** and **SMAC3** algorithms are available in the folders:

- /optimization_experiments/mnist 
- /optimization_experiments/emnist
- /optimization_experiments/dimex

After downloading and training any of the three mentioned versions of W-EAM, just copy
the experiment you wish to run inside the base directory of the project. 

Some slurm files to queue experiments runs are available in /slurm.

Scripts to generate plots and statistics are also available in /plotting_scripts. 

Hardcoded variables were used across all these experiments, so you should read the code
and make the necessary changes before using it.

## License

Copyright [2025] Luis Alberto Pineda Cortés, Carlos Ignacio Hernandez Castellanos, Juan Antonio Lopez Rivera  and Rafael Morales Gamboa.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
