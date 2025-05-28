import numpy as np
import torch
import torchaudio
import time
import matplotlib.pyplot as plt
from pathlib import Path

def process_rf32_file(path):
    data = np.fromfile(path, dtype=np.float32)
    tensor = torch.tensor(data).unsqueeze(0) 
    
    # Generate Spectrogram
    start = time.time()
    spec = torchaudio.transforms.Spectrogram(
    n_fft=1024,
    win_length=512,
    hop_length=512,
    window_fn=torch.hann_window
)(tensor)

    log_spec = torch.log1p(spec)

    # Resize + normalize
    resized = torch.nn.functional.interpolate(log_spec.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
    normalized = (resized - resized.mean()) / (resized.std() + 1e-6)
    end = time.time()

    print(f"Processing time: {end - start:.3f} seconds")
    return normalized  # shape: [1, 224, 224]

def display_spectrogram(spectrogram, file_path):
    """Display the processed spectrogram"""
    plt.figure(figsize=(10, 8))
    
    # Convert to numpy and squeeze to 2D
    spec_np = spectrogram.squeeze().numpy()
    
    # Display spectrogram
    plt.imshow(spec_np, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Normalized Amplitude')
    
    # Add labels and title
    plt.title(f'Processed RF Signal Spectrogram\n{file_path.name}', fontsize=14, fontweight='bold')
    plt.xlabel('Time Bins', fontsize=12)
    plt.ylabel('Frequency Bins', fontsize=12)
    
    # Add some info text
    plt.text(0.02, 0.98, f'Shape: {spec_np.shape}\nMin: {spec_np.min():.3f}\nMax: {spec_np.max():.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# Test
file_path = Path(r"C:\Users\yousi\Downloads\10 GHz Bandwidth\8PSK_5GSyms_75mV\frame_15.sigmf-data")
spec = process_rf32_file(file_path)

# Display the processed spectrogram
display_spectrogram(spec, file_path)
