import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class StructuralDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str, transforms: A.Compose = None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms
        
        if self.images_dir.exists():
            self.image_files = sorted([f for f in self.images_dir.iterdir() if f.suffix in ['.png', '.jpg', '.jpeg']])
        else:
            self.image_files = []
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        bboxes = []
        class_labels = []
        
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_c, y_c, w, h = map(float, parts[1:])
                        # Convert YOLO to Pascal VOC for albumentations (xmin, ymin, xmax, ymax)
                        x1 = max(0.0, x_c - w/2.0)
                        y1 = max(0.0, y_c - h/2.0)
                        x2 = min(1.0, x_c + w/2.0)
                        y2 = min(1.0, y_c + h/2.0)
                        bboxes.append([x1, y1, x2, y2])
                        class_labels.append(class_id)
                        
        if self.transforms:
            try:
                transformed = self.transforms(image=image, bboxes=bboxes, class_labels=class_labels)
                image = transformed['image']
                bboxes = transformed['bboxes']
                class_labels = transformed['class_labels']
            except Exception as e:
                # Fallback
                image = ToTensorV2()(image=image)["image"]

        target = {
            "boxes": torch.tensor(bboxes, dtype=torch.float32) if len(bboxes) > 0 else torch.zeros((0, 4)),
            "labels": torch.tensor(class_labels, dtype=torch.int64) if len(class_labels) > 0 else torch.zeros((0,), dtype=torch.int64)
        }
        
        return image, target

def get_train_transforms():
    return A.Compose([
        A.RandomResizedCrop(height=640, width=640, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))

def get_val_transforms():
    return A.Compose([
        A.Resize(height=640, width=640),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))
