import os
import numpy as np
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

def process_thz_signal(file_path, duration_ms=16):
    """
    Process THz signal using exact RRD parameters
    STFT: FFT=1024, Window=512, Hop=512, Hanning, 16ms slices
    """
    try:
        # Load signal data (160 GHz sample rate)
        sr = 160e9 
        data = np.fromfile(file_path, dtype=np.float32)
        
        if len(data) == 0:
            return []
        
        signal = torch.tensor(data)
        
        # Calculate samples per 16ms slice
        samples_per_slice = int(sr * duration_ms / 1000)
        
        if len(signal) < 1024:
            return []
        
        # If signal is shorter than 16ms, use entire signal
        if len(signal) < samples_per_slice:
            samples_per_slice = len(signal)
        
        # Slice signal into 16ms chunks
        spectrograms = []
        for start in range(0, len(signal) - samples_per_slice + 1, samples_per_slice):
            segment = signal[start:start + samples_per_slice]
            
            # RRD Dataset Parameters 
            spec = torchaudio.transforms.Spectrogram(
                n_fft=1024,          
                win_length=512,      
                hop_length=512,     
                window_fn=torch.hann_window,  
                power=2.0
            )(segment.unsqueeze(0))
            
            # Log scale and resize to (224, 224)
            log_spec = torch.log1p(spec)
            resized = torch.nn.functional.interpolate(
                log_spec.unsqueeze(0), 
                size=(224, 224), 
                mode='bilinear', 
                align_corners=False
            ).squeeze()
            
            spectrograms.append(resized)
        
        return spectrograms

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def compute_global_stats(specs):
    """Compute global normalization statistics"""
    if not specs:
        raise ValueError("No spectrograms collected!")
    
    all_specs = torch.stack(specs)
    return all_specs.mean(), all_specs.std()

def save_normalized_spectrogram(spec, mean, std, output_path):
    """Normalize and save spectrogram"""
    normalized = (spec - mean) / (std + 1e-8)
    torch.save(normalized, output_path)

def process_thz_dataset(input_root, output_root):
    """Process THz dataset with 160 GHz sample rate"""
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print("=== Processing THz Dataset (160 GHz) ===")
    print(f"Input: {input_root}")
    print(f"Output: {output_root}")
    
    # Find all .sigmf-data files
    sigmf_files = list(input_root.rglob("*.sigmf-data"))
    print(f"Found {len(sigmf_files)} .sigmf-data files")
    
    if len(sigmf_files) == 0:
        print("No .sigmf-data files found!")
        return

    # Pass 1: Collect all spectrograms
    print("\n=== Pass 1: Computing spectrograms ===")
    all_specs = []
    file_spec_pairs = []
    
    for file_path in tqdm(sigmf_files, desc="Processing files"):
        specs = process_thz_signal(file_path)
        if specs:
            all_specs.extend(specs)
            file_spec_pairs.append((file_path, specs))
    
    if len(all_specs) == 0:
        print("Error: No spectrograms generated!")
        return
        
    print(f"Generated {len(all_specs)} spectrograms from {len(file_spec_pairs)} files")
    
    # Compute normalization stats
    print("\n=== Computing normalization statistics ===")
    mean, std = compute_global_stats(all_specs)
    print(f"Global mean: {mean:.6f}, std: {std:.6f}")
    
    # Pass 2: Save normalized spectrograms
    print("\n=== Pass 2: Saving spectrograms ===")
    saved_count = 0
    
    for file_path, specs in tqdm(file_spec_pairs, desc="Saving"):
        # Maintain directory structure
        rel_path = file_path.relative_to(input_root)
        output_dir = output_root / rel_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = file_path.stem.replace(".sigmf-data", "")
        for i, spec in enumerate(specs):
            if len(specs) == 1:
                output_path = output_dir / f"{base_name}.pt"
            else:
                output_path = output_dir / f"{base_name}_slice_{i:03d}.pt"
            
            save_normalized_spectrogram(spec, mean, std, output_path)
            saved_count += 1
    
    print(f"\n=== Complete ===")
    print(f"Saved {saved_count} spectrograms to: {output_root}")
    
    # Save stats for later use
    torch.save({"mean": mean, "std": std}, output_root / "normalization_stats.pt")

if __name__ == "__main__":
    process_thz_dataset(
        input_root=r"C:/Users/yousi/Downloads", 
        output_root="preprocessed-thz-data"
    )
