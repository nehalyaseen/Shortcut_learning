#!/bin/bash -l
#SBATCH -p swarm_a100,a100,scavenger_mathsa100,scavenger_4a100,dual_h200,quad_h200,l4,swarm_l4,l40
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH -c 8
#SBATCH --time=12:00:00
#SBATCH -D /home/ny1g23/denetdm_sgd/shallow_vs_deep

# capture last two arguments
args=("$@")
arg_count=$#
name="${args[$arg_count-2]}_${args[$arg_count-1]}"
echo "$@"

exec >"/home/ny1g23/denetdm_sgd/my_slurm_outputs/${name}_${SLURM_JOB_ID}.out"

module load conda/python3
source activate denet

python main.py "$@"