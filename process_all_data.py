import numpy as np
import torch
import torchaudio
import time
import os
import pickle
from pathlib import Path
from tqdm import tqdm
import json

def process_rf32_file(path):
    """Process a single RF32 file and return normalized spectrogram"""
    try:
        data = np.fromfile(path, dtype=np.float32)
        if len(data) == 0:
            print(f"Warning: Empty file {path}")
            return None
            
        tensor = torch.tensor(data).unsqueeze(0) 
        
        # High-quality spectrogram
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
        
        return normalized  # shape: [1, 224, 224]
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None

def find_all_sigmf_data_files():
    """Find all .sigmf-data files in all bandwidth datasets"""
    base_path = Path(r"C:\Users\yousi\Downloads")
    bandwidth_dirs = ["5 GHz Bandwidth", "10 GHz Bandwidth", "20 GHz Bandwidth"]
    
    all_files = []
    dataset_info = {}
    
    for bandwidth_dir in bandwidth_dirs:
        bandwidth_path = base_path / bandwidth_dir
        if not bandwidth_path.exists():
            print(f"Warning: {bandwidth_path} does not exist")
            continue
            
        print(f"Scanning {bandwidth_dir}...")
        dataset_info[bandwidth_dir] = {}
        
        # Get all subdirectories (modulation types)
        for modulation_dir in bandwidth_path.iterdir():
            if modulation_dir.is_dir():
                print(f"  Found modulation type: {modulation_dir.name}")
                dataset_info[bandwidth_dir][modulation_dir.name] = []
                
                # Find all .sigmf-data files in this directory
                sigmf_files = list(modulation_dir.glob("*.sigmf-data"))
                print(f"    Found {len(sigmf_files)} .sigmf-data files")
                
                for file_path in sigmf_files:
                    file_info = {
                        'path': str(file_path),
                        'bandwidth': bandwidth_dir,
                        'modulation': modulation_dir.name,
                        'filename': file_path.name
                    }
                    all_files.append(file_info)
                    dataset_info[bandwidth_dir][modulation_dir.name].append(file_path.name)
    
    return all_files, dataset_info

def create_output_structure(output_dir):
    """Create the output directory structure"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create subdirectories for each bandwidth and modulation type
    bandwidth_dirs = ["5 GHz Bandwidth", "10 GHz Bandwidth", "20 GHz Bandwidth"]
    
    for bandwidth_dir in bandwidth_dirs:
        bandwidth_path = output_path / bandwidth_dir
        bandwidth_path.mkdir(exist_ok=True)
        
        # Create modulation subdirectories
        base_path = Path(r"C:\Users\yousi\Downloads") / bandwidth_dir
        if base_path.exists():
            for modulation_dir in base_path.iterdir():
                if modulation_dir.is_dir():
                    mod_path = bandwidth_path / modulation_dir.name
                    mod_path.mkdir(exist_ok=True)

def process_all_datasets():
    """Main function to process all datasets"""
    print("=== RF Signal Processing Pipeline ===")
    print("Finding all .sigmf-data files...")
    
    # Find all files
    all_files, dataset_info = find_all_sigmf_data_files()
    
    print(f"\nFound {len(all_files)} total .sigmf-data files")
    print("\nDataset Summary:")
    for bandwidth, modulations in dataset_info.items():
        print(f"  {bandwidth}:")
        for mod_type, files in modulations.items():
            print(f"    {mod_type}: {len(files)} files")
    
    # Create output directory structure
    output_dir = "preprocessed-data"
    print(f"\nCreating output directory: {output_dir}")
    create_output_structure(output_dir)
    
    # Save dataset info
    with open(Path(output_dir) / "dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # Process all files
    print(f"\nProcessing {len(all_files)} files...")
    processed_count = 0
    failed_count = 0
    
    start_time = time.time()
    
    for i, file_info in enumerate(tqdm(all_files, desc="Processing files")):
        try:
            # Process the file
            processed_data = process_rf32_file(file_info['path'])
            
            if processed_data is not None:
                # Create output path
                output_path = Path(output_dir) / file_info['bandwidth'] / file_info['modulation']
                output_filename = file_info['filename'].replace('.sigmf-data', '_processed.pt')
                output_file = output_path / output_filename
                
                # Save processed data
                torch.save(processed_data, output_file)
                processed_count += 1
            else:
                failed_count += 1
                
        except Exception as e:
            print(f"Failed to process {file_info['path']}: {e}")
            failed_count += 1
        
        # Progress update every 100 files
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(all_files) - i - 1) / rate if rate > 0 else 0
            print(f"Processed {i + 1}/{len(all_files)} files. Rate: {rate:.1f} files/sec. ETA: {remaining/60:.1f} min")
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n=== Processing Complete ===")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successfully processed: {processed_count} files")
    print(f"Failed: {failed_count} files")
    print(f"Average processing rate: {len(all_files)/total_time:.1f} files/sec")
    print(f"Output directory: {Path(output_dir).absolute()}")
    
    # Save processing summary
    summary = {
        'total_files': len(all_files),
        'processed_successfully': processed_count,
        'failed': failed_count,
        'processing_time_minutes': total_time/60,
        'average_rate_files_per_sec': len(all_files)/total_time,
        'dataset_info': dataset_info
    }
    
    with open(Path(output_dir) / "processing_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Processing summary saved to: {Path(output_dir) / 'processing_summary.json'}")

if __name__ == "__main__":
    process_all_datasets() 