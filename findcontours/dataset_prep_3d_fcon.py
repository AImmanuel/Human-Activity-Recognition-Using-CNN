import os
import numpy as np
import torch
from torch.utils.data import Dataset

class OpticalFlow3DDataset(Dataset):
    def __init__(self, base_folder):
        self.base_folder = base_folder
        self.labels = [] 
        self.file_paths = [] 
        
        for root, _, files in os.walk(base_folder):
            for file in files:
                if file.endswith(".npy"):
                    file_path = os.path.join(root, file)
                    self.file_paths.append(file_path)
                    data = np.load(file_path, allow_pickle=True).item()
                    self.labels.append(data['label'])
                    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = np.load(file_path, allow_pickle=True).item()
        if data['array'].ndim == 0:
            raise ValueError(f"Encountered zero-dimensional array in file: {file_path}")
        stacked_frames = data['array']  # Assuming the array is already stacked
        if stacked_frames.ndim == 3:  # Adding channel dimension if not present
            stacked_frames = np.expand_dims(stacked_frames, axis=0)
 
        label = int(data['label'])
        if label in range(1, 6):
            label = 1
        else:
            label = 0
 
        return torch.tensor(stacked_frames, dtype=torch.float32), torch.tensor(label, dtype=torch.long)