import torch
import torch.utils.data as data
from pathlib import Path
import os
import random
from typing import List, Tuple, Dict

class RFSignalDataset(data.Dataset):
    """
    Custom dataset for loading preprocessed RF signal spectrograms.
    Each sample is a 224x224 spectrogram tensor saved as .pt file.
    """
    
    def __init__(self, data_path: str, transform=None, split='train', train_ratio=0.8):
        """
        Args:
            data_path: Path to preprocessed-data directory
            transform: Optional transform to be applied on a sample
            split: 'train' or 'val' 
            train_ratio: Ratio of data to use for training
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.split = split
        
        # Find all .pt files across all bandwidth/modulation combinations
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        self._load_samples()
        self._create_split(train_ratio)
        
    def _load_samples(self):
        """Load all .pt files and create class mappings"""
        class_idx = 0
        
        # Iterate through bandwidth directories
        for bandwidth_dir in self.data_path.iterdir():
            if not bandwidth_dir.is_dir():
                continue
                
            # Iterate through modulation directories
            for modulation_dir in bandwidth_dir.iterdir():
                if not modulation_dir.is_dir():
                    continue
                    
                # Create class name from bandwidth and modulation
                class_name = f"{bandwidth_dir.name}_{modulation_dir.name}"
                
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = class_idx
                    self.idx_to_class[class_idx] = class_name
                    class_idx += 1
                
                # Find all .pt files in this directory
                pt_files = list(modulation_dir.glob("*.pt"))
                
                for pt_file in pt_files:
                    self.samples.append({
                        'path': pt_file,
                        'class_name': class_name,
                        'class_idx': self.class_to_idx[class_name],
                        'bandwidth': bandwidth_dir.name,
                        'modulation': modulation_dir.name
                    })
        
        print(f"Found {len(self.samples)} samples across {len(self.class_to_idx)} classes")
        print(f"Classes: {list(self.class_to_idx.keys())}")
        
    def _create_split(self, train_ratio: float):
        """Split data into train/val based on ratio"""
        # Group samples by class for stratified split
        class_samples = {}
        for sample in self.samples:
            class_name = sample['class_name']
            if class_name not in class_samples:
                class_samples[class_name] = []
            class_samples[class_name].append(sample)
        
        # Create stratified split
        train_samples = []
        val_samples = []
        
        for class_name, samples in class_samples.items():
            # Shuffle samples for this class
            random.shuffle(samples)
            
            # Split based on ratio
            n_train = int(len(samples) * train_ratio)
            train_samples.extend(samples[:n_train])
            val_samples.extend(samples[n_train:])
        
        if self.split == 'train':
            self.samples = train_samples
        else:
            self.samples = val_samples
            
        print(f"{self.split} split: {len(self.samples)} samples")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            tensor: Preprocessed spectrogram tensor of shape [1, 224, 224]
            class_idx: Class index for classification (optional)
        """
        sample_info = self.samples[idx]
        
        # Load the preprocessed tensor
        try:
            tensor = torch.load(sample_info['path'], map_location='cpu')
            
            # Ensure tensor is the right shape [1, 224, 224]
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)  # Add channel dimension
            elif tensor.dim() == 3 and tensor.shape[0] != 1:
                tensor = tensor[:1]  # Take first channel if multiple
                
            # Convert single channel to 3 channels for MAE (RGB-like)
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)  # [3, 224, 224]
                
        except Exception as e:
            print(f"Error loading {sample_info['path']}: {e}")
            # Return a zero tensor as fallback
            tensor = torch.zeros(3, 224, 224)
        
        # Apply transforms if provided
        if self.transform:
            tensor = self.transform(tensor)
            
        return tensor, sample_info['class_idx']
    
    def get_class_info(self) -> Dict:
        """Return class mapping information"""
        return {
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class,
            'num_classes': len(self.class_to_idx)
        }
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get detailed information about a specific sample"""
        return self.samples[idx]


def create_rf_dataloaders(data_path: str, batch_size: int = 64, num_workers: int = 4, 
                         train_ratio: float = 0.8, pin_memory: bool = True):
    """
    Create train and validation dataloaders for RF signal data
    
    Args:
        data_path: Path to preprocessed-data directory
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        train_ratio: Ratio of data to use for training
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        train_loader, val_loader, class_info
    """
    
    # For MAE pretraining, we typically don't need heavy augmentation
    # since the model learns from masked reconstruction
    from torchvision import transforms
    
    # Minimal transforms - data is already preprocessed
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = RFSignalDataset(
        data_path=data_path,
        transform=transform,
        split='train',
        train_ratio=train_ratio
    )
    
    val_dataset = RFSignalDataset(
        data_path=data_path,
        transform=transform,
        split='val',
        train_ratio=train_ratio
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    class_info = train_dataset.get_class_info()
    
    print(f"Created dataloaders:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Classes: {class_info['num_classes']}")
    
    return train_loader, val_loader, class_info


if __name__ == "__main__":
    # Test the dataset
    data_path = "preprocessed-data"
    train_loader, val_loader, class_info = create_rf_dataloaders(data_path, batch_size=8)
    
    # Test loading a batch
    for batch_idx, (data, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}: data shape {data.shape}, targets shape {targets.shape}")
        print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
        if batch_idx >= 2:  # Just test a few batches
            break 