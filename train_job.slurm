#!/bin/bash
#SBATCH --partition=markov_gpu
#SBATCH --account=csds465
#SBATCH --time=96:00:00
#SBATCH -o train_manga.log
#SBATCH --constraint=gpu2h100
#SBATCH --mem=64gb
#SBATCH --requeue

# Load Modules
module purge
module load Python/3.9.6-GCCcore-11.2.0
module load CUDA/12.6.0

python3 -m venv <Path>/<VEnvName>
source /<Path>/<VEnvName>/bin/activate

git clone https://github.com/XingtongGe/gsplat.git
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
cd gsplat
BUILD_NO_CUDA=1 pip install -e .[dev]
pip install --no-cache-dir --no-build-isolation -r requirements_fixed.txt

# Export environment variables
export TORCH_CUDA_ARCH_LIST="9.0"

# Run the training script
python train_full.py --dataset_config ./configs/train/train-manga109.yaml --epochs 200 --batch_size 1 --lr 5e-5 --num_workers 4 --resume ./checkpoints/continuous_sgs_sr_epoch_100.pth

