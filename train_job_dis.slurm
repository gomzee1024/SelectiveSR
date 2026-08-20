#!/bin/bash
#SBATCH --partition=markov_gpu
#SBATCH --account=csds451
#SBATCH --time=01:00:00
#SBATCH --output=train_sgs_manga.log
#SBATCH --constraint=gpu4090
#SBATCH --mem=64gb
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

# ====================================================
# USER INSTRUCTIONS:
#
# 1. To run on 1 GPU with 4 MPI processes:
#    sbatch --ntasks=4 --gres=gpu:1 run_dist.slurm
#
# 2. To run on 2 GPUs with 8 MPI processes:
#    sbatch --ntasks=8 --gres=gpu:2 run_dist.slurm
# ====================================================

# 1. Load Modules
module purge
module load Python/3.9.6-GCCcore-11.2.0
module load CUDA/12.6.0
module load OpenMPI/4.1.1-GCC-11.2.0

# 2. Activate Environment
source /mnt/pan/courses/csds451/gxk384/csds465pa2/bin/activate

# 3. Export Vars
export TORCH_CUDA_ARCH_LIST="8.9"
export UCX_LOG_LEVEL=error

# 4. Debug Info
echo "------------------------------------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Tasks (CPUs): $SLURM_NTASKS"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "------------------------------------------------"

# 5. Run Distributed Training
# --mca pml ob1 --mca btl self,vader,tcp : Fixes UCX/InfiniBand crashes
# -n $SLURM_NTASKS : Uses the number of tasks requested in sbatch

mpirun -n $SLURM_NTASKS \
python -u train_full_distributed.py \
    --dataset_config /mnt/pan/courses/csds451/gxk384/ContinuousSR/configs/train/train-manga109.yaml \
    --epochs 200 \
    --batch_size 1 \
    --lr 5e-5 \
    --num_workers 4 \
    --resume /mnt/pan/courses/csds451/gxk384/ContinuousSR/checkpoints/continuous_sgs_sr_epoch_100.pth