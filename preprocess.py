import numpy as np
import torch
import torchaudio
import time
from pathlib import Path

def process_rf32_file(path):
    data = np.fromfile(path, dtype=np.float32)
    tensor = torch.tensor(data).unsqueeze(0) 
    
    # Generate Spectrogram
    start = time.time()
    spec = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512)(tensor)
    log_spec = torch.log1p(spec)

    # Resize + normalize
    resized = torch.nn.functional.interpolate(log_spec.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
    normalized = (resized - resized.mean()) / (resized.std() + 1e-6)
    end = time.time()

    print(f"Processing time: {end - start:.3f} seconds")
    return normalized  # shape: [1, 224, 224]

# Test
file_path = Path(r"C:\Users\yousi\Downloads\20 GHz Bandwidth\4PSK_10GSyms_75mV\frame_1.sigmf-data")
spec = process_rf32_file(file_path)
