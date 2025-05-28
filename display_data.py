import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path

def find_all_processed_files():
    """Find all processed .pt files in the dataset"""
    processed_dir = Path("preprocessed-thz-data")
    
    if not processed_dir.exists():
        print("Error: preprocessed-thz-data directory not found!")
        return []
    
    all_files = []
    
    # Walk through all bandwidth directories
    for bandwidth_dir in processed_dir.iterdir():
        if bandwidth_dir.is_dir() and "GHz" in bandwidth_dir.name:
            # Walk through all modulation directories
            for modulation_dir in bandwidth_dir.iterdir():
                if modulation_dir.is_dir():
                    # Find all .pt files
                    pt_files = list(modulation_dir.glob("*.pt"))
                    for pt_file in pt_files:
                        file_info = {
                            'path': pt_file,
                            'bandwidth': bandwidth_dir.name,
                            'modulation': modulation_dir.name,
                            'filename': pt_file.name
                        }
                        all_files.append(file_info)
    
    return all_files

def load_spectrogram(file_path):
    """Load a processed spectrogram from a .pt file"""
    try:
        spectrogram = torch.load(file_path, map_location='cpu')
        if isinstance(spectrogram, torch.Tensor):
            return spectrogram.squeeze().numpy()
        return spectrogram
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def display_random_grid():
    """Display a 4x4 grid of 16 random spectrograms"""
    
    print("Finding all processed files...")
    all_files = find_all_processed_files()
    
    if len(all_files) == 0:
        print("No processed files found!")
        return
    
    print(f"Found {len(all_files)} processed files")
    
    # Randomly sample 16 files
    selected_files = random.sample(all_files, min(16, len(all_files)))
    
    # Create 4x4 grid
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Random THz Signal Spectrograms (160 GHz sampling)', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i in range(16):
        ax = axes[i]
        
        if i < len(selected_files):
            file_info = selected_files[i]
            
            # Load and display spectrogram
            spectrogram = load_spectrogram(file_info['path'])
            
            if spectrogram is not None:
                im = ax.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
                title = f"{file_info['modulation']}\n{file_info['bandwidth']}"
                ax.set_title(title, fontsize=8)
                
                # Add frequency and time axes
                # Assuming 224x224 spectrogram from the processing
                height, width = spectrogram.shape
                
                # Time axis (x-axis) - THz signals are 16ms duration
                time_ticks = np.linspace(0, width-1, 5)
                time_labels = [f"{t:.2f}" for t in np.linspace(0, 16, 5)]  # 0 to 16 ms
                ax.set_xticks(time_ticks)
                ax.set_xticklabels(time_labels, fontsize=6)
                ax.set_xlabel('Time (ms)', fontsize=7)
                
                # Frequency axis (y-axis) - based on bandwidth around 10 GHz IF
                freq_ticks = np.linspace(0, height-1, 5)
                if "5 GHz" in file_info['bandwidth']:
                    freq_labels = [f"{f:.1f}" for f in np.linspace(7.5, 12.5, 5)]  # 5 GHz BW around 10 GHz IF
                elif "10 GHz" in file_info['bandwidth']:
                    freq_labels = [f"{f:.1f}" for f in np.linspace(5.0, 15.0, 5)]  # 10 GHz BW around 10 GHz IF
                elif "20 GHz" in file_info['bandwidth']:
                    freq_labels = [f"{f:.1f}" for f in np.linspace(0.0, 20.0, 5)]  # 20 GHz BW around 10 GHz IF
                else:
                    freq_labels = [f"{f:.1f}" for f in np.linspace(5.0, 15.0, 5)]  # default 10 GHz BW
                
                ax.set_yticks(freq_ticks)
                ax.set_yticklabels(freq_labels, fontsize=6)
                ax.set_ylabel('Frequency (GHz)', fontsize=7)
                
            else:
                ax.text(0.5, 0.5, 'Failed to load', ha='center', va='center', transform=ax.transAxes)
                ax.set_xlabel('Time (ms)', fontsize=7)
                ax.set_ylabel('Frequency (GHz)', fontsize=7)
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    display_random_grid() 