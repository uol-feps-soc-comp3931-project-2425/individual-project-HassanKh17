import os
import cv2
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from .preprocess import preprocess_dataset

class PoseDataset(Dataset):
    """Dataset for loading RGB images, masks and 6D pose annotations"""
    def __init__(self, image_dir, annotation_file, mask_dir=None, transform=None):
        """
        Args:
            image_dir: Directory with RGB images
            annotation_file: JSON file with pose annotations
            mask_dir: Directory with mask images (optional)
            transform: Torchvision transforms
        """
        preprocess_dataset(dataset_root)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        
        # Load pose annotations
        with open(annotation_file, "r") as f:
            self.annotations = json.load(f)
            
        # Default transform converts to tensor
        self.transform = transform or transforms.Lambda(lambda x: TF.to_tensor(x))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get annotation for this sample
        ann = self.annotations[idx]
        
        # Load and resize RGB image
        img_path = os.path.join(self.image_dir, f"{ann['image_id']}.png")
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (640, 480))
        
        # Load mask if available
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, f"{ann['image_id']}.png")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found: {mask_path}")
                
            mask = cv2.resize(mask, (640, 480)) / 255.0  # Normalize to [0,1]
            
            # Stack RGB + mask (4 channels total)
            image = np.dstack((image, mask)).astype(np.float32)
        
        # Apply transforms
        image = self.transform(image)
        
        # Get pose targets
        translation = torch.tensor(ann["translation"], dtype=torch.float32) / 1000  # Convert mm to meters
        rotation = torch.tensor(ann["rotation"], dtype=torch.float32)
        rotation = rotation / torch.norm(rotation)  # Normalize quaternion
        
        return image, translation, rotation

def create_dataloader(image_dir, annotation_file, mask_dir=None, batch_size=16, shuffle=True):
    """Helper function to create a DataLoader"""
    dataset = PoseDataset(image_dir, annotation_file, mask_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )
