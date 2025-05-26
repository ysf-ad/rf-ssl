#!/usr/bin/env python3
"""
Launch script for RF Signal MAE Pre-training

This script launches the MAE pretraining with parameters similar to the original command.
Adapted for RF signal data instead of ImageNet.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    # Set up paths
    output_dir = "./rf_mae_output"
    log_dir = "./rf_mae_logs"
    data_path = "preprocessed-data"
    
    # Create directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if data exists
    if not Path(data_path).exists():
        print(f"Error: Data path {data_path} does not exist!")
        print("Please make sure you have run the preprocessing script first.")
        sys.exit(1)
    
    # Build command similar to the original submitit command
    cmd = [
        "python", "rf_mae_pretrain.py",
        "--batch_size", "32",  # Reduced from 64 for memory efficiency
        "--model", "mae_vit_large_patch16",
        "--norm_pix_loss",
        "--mask_ratio", "0.75",
        "--epochs", "400",  # Reduced from 800 for faster training
        "--warmup_epochs", "40",
        "--blr", "1.5e-4",
        "--weight_decay", "0.05",
        "--data_path", data_path,
        "--output_dir", output_dir,
        "--log_dir", log_dir,
        "--num_workers", "4",
        "--pin_mem"
    ]
    
    print("Starting RF Signal MAE Pre-training...")
    print("Command:", " ".join(cmd))
    print(f"Output directory: {output_dir}")
    print(f"Log directory: {log_dir}")
    print(f"Data path: {data_path}")
    print("-" * 50)
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        print("Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main() 