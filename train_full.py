import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import argparse 
from argparse import Namespace
import os
import yaml 
import pprint 

# --- Import all model and utility files ---
import models
import utils
import datasets

# --- HARDCODED MODEL SPECIFICATION (HAT Encoder + Continuous Gaussian) ---
# NOTE: This specification uses the HAT encoder with high-dimensional settings,
# which is why this model is high-performing.
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
# --- END HARDCODED MODEL SPECIFICATION ---

# --- Training Function ---
def train_model(epochs, batch_size, lr, div2k_hr_path, num_workers, dataset_config_path, resume_path=None):
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This model requires a GPU.")
        return
    device = torch.device("cuda")

    # 1. Use Hardcoded Model Specification
    model_spec = HAT_GAUSSIAN_MODEL_SPEC
    
    print("\n--- MODEL SPECIFICATION (HAT + Gaussian) ---")
    pprint.pprint(model_spec)
    print("------------------------------------------\n")
    
    # 2. Load Dataset Specification
    print(f"Loading dataset configuration from {dataset_config_path}...")
    with open(dataset_config_path, 'r') as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    
    dataset_spec = data_config['train_dataset']
    
    # --- Data Loading ---
    
    # Override root_path and batch_per_gpu args using command-line input
    final_root_path = div2k_hr_path
    if not final_root_path:
        final_root_path = dataset_spec['dataset']['args'].get('root_path')

    dataset_spec['dataset']['args']['root_path'] = final_root_path
    dataset_spec['wrapper']['args']['batch_per_gpu'] = batch_size

    if not final_root_path or not os.path.exists(final_root_path):
        print("="*50)
        print("ERROR: DIV2K HR training path is not set or does not exist.")
        print(f"Path tried: {final_root_path}")
        print("Please ensure '--path' is set correctly or 'root_path' in the YAML file is valid.")
        print("="*50)
        return

    print("Initializing dataset...")
    
    # 3. Initialize the Dataset using the nested structure
    base_dataset = datasets.make(dataset_spec['dataset'])
    wrapped_dataset = datasets.make(dataset_spec['wrapper'], args={'dataset': base_dataset})
    
    print(f"Dataset '{final_root_path}' loaded successfully.")
    
    # 4. Initialize DataLoader
    dataloader = DataLoader(
        wrapped_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    print("Initializing model...")
    
    # 5. Initialize the Model using models.make and the hardcoded spec
    model = models.make(model_spec).to(device)
    print(f"Model created. Total parameters: {utils.compute_num_params(model, text=True)}")

    # 6. Initialize Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    # --- RESUME TRAINING LOGIC ---
    start_epoch = 0
    if resume_path:
        if os.path.isfile(resume_path):
            print(f"Loading checkpoint from '{resume_path}'")
            checkpoint = torch.load(resume_path)
            
            # Check if the checkpoint has the structure we expect (dict with 'model')
            # or if it's a direct state_dict save (older versions/different scripts)
            if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
                 # This matches the structure we save in this script: {'model': {'sd': state_dict, ...}}
                 state_dict = checkpoint['model'].get('sd')
                 if state_dict is None:
                     # Maybe it's just the spec without weights? Try loading 'model' directly if it looks like weights
                     # But based on our save logic, 'sd' should be there.
                     print("Warning: 'sd' key not found in checkpoint['model']. Attempting to load 'model' directly.")
                     state_dict = checkpoint['model']
            else:
                # Assume it's a raw state_dict
                state_dict = checkpoint

            try:
                model.load_state_dict(state_dict)
                print("Successfully loaded model weights.")
                
                # Try to parse epoch number from filename (e.g., continuous_sr_epoch_10.pth)
                # This is a simple heuristic.
                try:
                    filename = os.path.basename(resume_path)
                    if "epoch_" in filename:
                        epoch_str = filename.split("epoch_")[1].split(".")[0]
                        start_epoch = int(epoch_str)
                        print(f"Resuming from epoch {start_epoch}")
                except Exception:
                    print("Could not determine epoch from filename. Starting from epoch 0.")

            except Exception as e:
                print(f"Error loading state_dict: {e}")
                print("Please check if the model architecture matches the checkpoint.")
                return
        else:
            print(f"Error: No checkpoint found at '{resume_path}'")
            return
    # -----------------------------

    print(f"Starting training from epoch {start_epoch+1} to {epochs}...")
    print(f"Batch size: {batch_size}, Learning rate: {lr}")

    # --- Training Loop ---
    model.train()
    total_steps = 0
    
    # Create directory for saving checkpoints if it doesn't exist
    save_dir = "checkpoints"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        running_loss = 0.0

        for i, batch in enumerate(dataloader):
            step_start_time = time.time()
            
            # Unpack the batch dictionary from the dataset wrapper
            lr_batch = batch['inp'].to(device)
            hr_batch = batch['gt'].to(device)

            # The scale value is randomly chosen per batch
            # FIX: Extract the scalar value correctly
            s = batch['scale'][0].item() 
            scale_tensor = torch.tensor([[s, s]], device=device) 

            # 7. Forward Pass
            optimizer.zero_grad()
            pred_batch = model(lr_batch, scale_tensor)
            
            # 8. Calculate Loss (Shape alignment check is critical for variable scales)
            if pred_batch.shape != hr_batch.shape:
                 pred_batch = F.interpolate(pred_batch, size=hr_batch.shape[-2:], mode='bicubic', align_corners=False)

            loss = criterion(pred_batch, hr_batch)

            # 9. Backward Pass and Optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_steps += 1
            
            if (i + 1) % 20 == 0: # Print log every 20 steps
                step_time = time.time() - step_start_time
                steps_per_sec = 20.0 / step_time
                print(f"[Epoch {epoch+1}/{epochs}] [Step {i+1}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f} | Steps/sec: {steps_per_sec:.2f}")

        epoch_loss = running_loss / len(dataloader)
        epoch_time = time.time() - epoch_start_time
        print("-" * 50)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {epoch_loss:.4f} | Time: {epoch_time:.2f}s")
        print("-" * 50)

        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(save_dir, f"continuous_sgs_sr_epoch_{epoch+1}.pth")
            
            # Save checkpoint in the format demo.py expects: { 'model': { model_spec, 'sd': weights } }
            temp_model_spec = HAT_GAUSSIAN_MODEL_SPEC.copy()
            temp_model_spec['sd'] = model.state_dict()
            checkpoint = {'model': temp_model_spec}
            torch.save(checkpoint, save_path)
            
            print(f"Model checkpoint saved to {save_path}")

    print("Training complete.")

# --- Main execution ---
if __name__ == "__main__":
    
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(description="Train ContinuousSR Model")
    
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of training epochs. (Default: 100)')
                        
    parser.add_argument('--batch_size', type=int, default=4, 
                        help='Batch size per GPU. (Default: 4, Reduce if you get CUDA OOM)')
                        
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Learning rate. (Default: 1e-4)')
                        
    parser.add_argument('--path', type=str, required=False, 
                        help='Path to the DIV2K HR training images folder (e.g., /path/to/DIV2K_train_HR). If not provided, path from YAML is used.')
                        
    parser.add_argument('--num_workers', type=int, default=8, 
                        help='Number of dataloader workers. (Default: 8)')
    
    parser.add_argument('--dataset_config', type=str, default='train-div2k.yaml',
                        help='Path to the dataset configuration YAML file (e.g., train-div2k.yaml).')

    # Added argument for resuming training
    parser.add_argument('--resume', type=str, default=None, required=False,
                        help='Path to a checkpoint file to resume training from.')
                        
    args = parser.parse_args()
    
    # 2. Call the training function with parsed arguments
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        div2k_hr_path=args.path,
        num_workers=args.num_workers,
        dataset_config_path=args.dataset_config,
        resume_path=args.resume
    )
