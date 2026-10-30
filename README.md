# Hypotheses about Shortcut Learning Don't Generalise: Rethinking Approaches to Systematic Evaluation

This repository is for the paper presented at the NeurIPS 2026 Workshop on *Self-Evolving Diversity-Driven Search for Robust AI Systems*.


Running Experiments on Iridis

Experiments are submitted to the Iridis HPC cluster using SLURM. The job submission script Shortcut_learning/my_slurm_output/jupyter.sh loads the conda module, activates the denet environment, and executes main.py with the specified dataset and model.

### 1. Clone the repository:
```bash
git clone https://github.com/nehalyaseen/Shortcut_learning.git
cd Shortcut_learning
```
### 2. Create the Conda environment
```bash 
conda env create -f environment.yml
```
### 3. Environment Setup

  Activate the Conda environment:
```bash
conda activate denet
```    
  On Iridis, the environment is activated by the SLURM script using:
```bash  
module load conda/python3
source activate denet
```
### 4. Submit a SLURM job

   Submit an experiment using the following syntax:
```bash
cd my_slurm_outputs
sbatch jupyter.sh <dataset> <model>
```

**Supported datasets**

| Dataset | Argument |
|---|---|
| CMNIST | `fg_mnist` |
| Domino | `mnist_cifar` |
| Waterbirds | `water_bird` |
| CelebA | `celeba` |

**Supported models**

| Model | Argument |
|---|---|
| MLP (3 layers) | `mlp3` |
| MLP (5 layers) | `mlp5` |
| ResNet18 | `resnet` |
| VGG16 | `vgg` |
| Vision Transformer | `vit` |
| CLIP | `clip` |

### 5. Example Commands

Run an MLP experiment on `fg_mnist`:

```bash
sbatch jupyter.sh fg_mnist mlp3
```

Run a ResNet experiment on Waterbirds:

```bash
sbatch jupyter.sh water_bird resnet
```

Run a Vision Transformer experiment on CelebA:

```bash
sbatch jupyter.sh celeba vit
```

Run a CLIP experiment on MNIST–CIFAR:

```bash
sbatch jupyter.sh mnist_cifar clip
```

### 6. Job Outputs

SLURM job output files are saved in:

```text
/home/ny1g23/denetdm_sgd/my_slurm_outputs/
```

Each output file is named using the dataset, model, and SLURM job ID. For example:

```text
fg_mnist_mlp3_123456.out
```
### 7. Results
Experimental results are stored in the results/ directory. Each row reports the mean accuracy ± standard deviation for a given setting.

```text
dataset,train,val,test_bg0_b100,test_bg100_b0
98,100.00 ± 0.00,91.40 ± 0.90,12.22 ± 1.54,87.78 ± 1.54
```

dataset: Shortcut-correlation setting. 98 means shortcut is 98% predictive while difficult pattern is fully predictive.

train: Training accuracy.

val: Validation accuracy.

test_bg0_b100, test_bg100_b0: Test accuracy under two different test settings for CMNIST dataset.

> **Note:** Refer to the paper for details of the experimental setup, evaluation protocol, and interpretation of the results.
