import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import time
import argparse
from argparse import Namespace
import os
import yaml
import pprint
import numpy as np
from mpi4py import MPI

# --- Import all model and utility files ---
import models
import utils
import datasets

# ==========================================
# 0. MPI & Device Setup
# ==========================================
# Suppress InfiniBand warnings
os.environ["UCX_LOG_LEVEL"] = "error"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    gpu_id = rank % num_gpus
    device = torch.device(f"cuda:{gpu_id}")

    # CRITICAL FIX: Force this process to only see/use its assigned GPU.
    # This prevents the "Ghost Process" issue on GPU 0.
    torch.cuda.set_device(device)
else:
    num_gpus = 0
    device = torch.device("cpu")

# --- HARDCODED MODEL SPECIFICATION (HAT Encoder + Continuous Gaussian) ---
HAT_GAUSSIAN_MODEL_SPEC = {
    'name': 'continuous-gaussian',
    'args': {
        'cnn_spec': {
            'args': {
                'init_range': 0.1
            },
            'name': 'cnn'
        },
        'encoder_spec': {
            'args': {
                'compress_ratio': 3,
                'conv_scale': 0.01,
                'depths': [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                'embed_dim': 180,
                'img_range': 1.0,
                'img_size': 64,
                'in_chans': 3,
                'mlp_ratio': 2,
                'num_heads': [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                'overlap_ratio': 0.5,
                'resi_connection': '1conv',
                'squeeze_factor': 30,
                'upsampler': 'pixelshuffle',
                'upscale': 4,
                'window_size': 16
            },
            'name': 'hat'
        },
        'fc_spec': {
            'args': {
                'hidden_list': [256, 256, 256, 256],
                'out_dim': 3
            },
            'name': 'mlp'
        }
    }
}


# ==========================================
# 1. Helper: All-Reduce Gradient Sync
# ==========================================
def sync_gradients(model):
    """
    All-Reduce Gradients: Move to CPU -> Flatten -> MPI Sum -> Average -> Move to GPU
    """
    grads_list = []
    # 1. Gather grads from GPU to CPU
    for param in model.parameters():
        if param.grad is not None:
            grads_list.append(param.grad.view(-1).cpu().numpy())
        else:
            grads_list.append(np.zeros(param.numel(), dtype=np.float32))

    flat_grads = np.concatenate(grads_list)
    global_grads = np.zeros_like(flat_grads)

    # 2. MPI Communication
    comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)

    # 3. Average & Update
    global_grads /= size

    ptr = 0
    for param in model.parameters():
        if param.grad is not None:
            num_el = param.numel()
            avg_grad_np = global_grads[ptr: ptr + num_el]
            ptr += num_el

            # Move back to GPU
            avg_grad_tensor = torch.from_numpy(avg_grad_np).view(param.shape).to(device)
            param.grad.data = avg_grad_tensor


# --- Training Function ---
def train_model(epochs, batch_size, lr, div2k_hr_path, num_workers, dataset_config_path, resume_path=None):
    if rank == 0:
        print(f"Initialized Training: {size} Processes, {num_gpus} GPUs total.")
        print("\n--- MODEL SPECIFICATION (HAT + Gaussian) ---")
        pprint.pprint(HAT_GAUSSIAN_MODEL_SPEC)
        print("------------------------------------------\n")
        print(f"Loading dataset configuration from {dataset_config_path}...")

    # 2. Load Dataset Specification
    with open(dataset_config_path, 'r') as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)

    dataset_spec = data_config['train_dataset']

    # --- Data Loading ---
    final_root_path = div2k_hr_path
    if not final_root_path:
        final_root_path = dataset_spec['dataset']['args'].get('root_path')

    dataset_spec['dataset']['args']['root_path'] = final_root_path
    dataset_spec['wrapper']['args']['batch_per_gpu'] = batch_size

    if (not final_root_path or not os.path.exists(final_root_path)) and rank == 0:
        print("=" * 50)
        print("ERROR: DIV2K HR training path is not set or does not exist.")
        print(f"Path tried: {final_root_path}")
        print("=" * 50)
        return

    if rank == 0:
        print("Initializing dataset...")

    # 3. Initialize the Dataset
    # Ensure all ranks initialize dataset (assuming read-only access is safe)
    base_dataset = datasets.make(dataset_spec['dataset'])
    wrapped_dataset = datasets.make(dataset_spec['wrapper'], args={'dataset': base_dataset})

    if rank == 0:
        print(f"Dataset '{final_root_path}' loaded successfully.")

    # 4. Distributed Sampler & DataLoader
    # CRITICAL: Sampler partitions data among ranks
    sampler = DistributedSampler(wrapped_dataset, num_replicas=size, rank=rank, shuffle=True)

    dataloader = DataLoader(
        wrapped_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use sampler instead of shuffle=True
        num_workers=num_workers,
        pin_memory=True
    )

    if rank == 0:
        print("Initializing model...")

    # 5. Initialize the Model
    model = models.make(HAT_GAUSSIAN_MODEL_SPEC).to(device)

    # Broadcast Initial Weights so everyone starts equal
    for param in model.parameters():
        data = param.data.view(-1).cpu().numpy()
        comm.Bcast(data, root=0)
        param.data = torch.from_numpy(data).view(param.shape).to(device)

    if rank == 0:
        print(f"Model created. Total parameters: {utils.compute_num_params(model, text=True)}")

    # 6. Initialize Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    # --- RESUME TRAINING LOGIC ---
    start_epoch = 0
    if resume_path:
        if os.path.isfile(resume_path):
            if rank == 0:
                print(f"Loading checkpoint from '{resume_path}'")

            # Map storage to current device to avoid OOM or loading to wrong GPU
            checkpoint = torch.load(resume_path, map_location=device)

            if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
                state_dict = checkpoint['model'].get('sd')
                if state_dict is None:
                    state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            try:
                model.load_state_dict(state_dict)
                if rank == 0:
                    print("Successfully loaded model weights.")

                try:
                    filename = os.path.basename(resume_path)
                    if "epoch_" in filename:
                        epoch_str = filename.split("epoch_")[1].split(".")[0]
                        start_epoch = int(epoch_str)
                        if rank == 0:
                            print(f"Resuming from epoch {start_epoch}")
                except Exception:
                    pass

            except Exception as e:
                if rank == 0:
                    print(f"Error loading state_dict: {e}")
                return
        else:
            if rank == 0:
                print(f"Error: No checkpoint found at '{resume_path}'")
            return
    # -----------------------------

    if rank == 0:
        print(f"Starting training from epoch {start_epoch + 1} to {epochs}...")
        print(f"Batch size: {batch_size}, Learning rate: {lr}")

    # --- Training Loop ---
    model.train()

    save_dir = "checkpoints"
    if rank == 0 and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(start_epoch, epochs):
        # CRITICAL: Shuffle data differently per epoch
        sampler.set_epoch(epoch)

        epoch_start_time = time.time()
        running_loss = 0.0

        for i, batch in enumerate(dataloader):
            step_start_time = time.time()

            lr_batch = batch['inp'].to(device)
            hr_batch = batch['gt'].to(device)
            s = batch['scale'][0].item()
            scale_tensor = torch.tensor([[s, s]], device=device)

            optimizer.zero_grad()

            # Forward
            pred_batch = model(lr_batch, scale_tensor)

            if pred_batch.shape != hr_batch.shape:
                pred_batch = F.interpolate(pred_batch, size=hr_batch.shape[-2:], mode='bicubic', align_corners=False)

            loss = criterion(pred_batch, hr_batch)

            # Backward
            loss.backward()

            # --- SYNC GRADIENTS ---
            sync_gradients(model)

            # Update
            optimizer.step()

            running_loss += loss.item()

            # Logging only on Rank 0
            if rank == 0 and (i + 1) % 20 == 0:
                step_time = time.time() - step_start_time
                steps_per_sec = 20.0 / step_time
                print(f"[Epoch {epoch + 1}/{epochs}] [Step {i + 1}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f} | Steps/sec: {steps_per_sec:.2f}", flush=True)

        epoch_loss = running_loss / len(dataloader)
        epoch_time = time.time() - epoch_start_time

        if rank == 0:
            print("-" * 50)
            print(f"Epoch {epoch + 1} Complete. Avg Loss: {epoch_loss:.4f} | Time: {epoch_time:.2f}s")
            print("-" * 50)

            if (epoch + 1) % 10 == 0:
                save_path = os.path.join(save_dir, f"continuous_sgs_sr_epoch_{epoch + 1}.pth")
                temp_model_spec = HAT_GAUSSIAN_MODEL_SPEC.copy()
                temp_model_spec['sd'] = model.state_dict()
                checkpoint = {'model': temp_model_spec}
                torch.save(checkpoint, save_path)
                print(f"Model checkpoint saved to {save_path}")

    if rank == 0:
        print("Training complete.")


# --- Main execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ContinuousSR Model")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--path', type=str, required=False)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--dataset_config', type=str, default='train-div2k.yaml')
    parser.add_argument('--resume', type=str, default=None, required=False)

    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        div2k_hr_path=args.path,
        num_workers=args.num_workers,
        dataset_config_path=args.dataset_config,
        resume_path=args.resume
    )