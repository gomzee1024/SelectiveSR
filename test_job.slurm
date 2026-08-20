#!/bin/bash
#SBATCH --partition=markov_gpu
#SBATCH --account=csds465
#SBATCH --time=24:00:00
#SBATCH -o test_con_sgs.log
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu4090
#SBATCH --mem=64gb
#SBATCH --requeue

# Load Modules
module load Python/3.9.6-GCCcore-11.2.0
module load CUDA/12.6.0

# Activate Virtual Environment
source /home/gxk384/csds451pa2/bin/activate

# Export environment variables
# H100 is Compute Capability 9.0. Previous value "8.9" was for RTX 4090/L40.
export TORCH_CUDA_ARCH_LIST="8.9"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the training script
python test.py --config ./configs/test/test-set5-4.yaml --model ./checkpoints/continuous_sgs_sr_epoch_100.pth
