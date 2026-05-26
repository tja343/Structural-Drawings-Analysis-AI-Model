import cv2
import numpy as np
import random
from typing import List, Tuple, Dict
from app.synthetic.components import SyntheticBeam, SyntheticText, BoundingBox

try:
    import albumentations as A
except ImportError:
    A = None

class DrawingGenerator:
    def __init__(self, width=1024, height=1024):
        self.width = width
        self.height = height
        
        # Albumentations pipeline for domain randomization
        self.transform = None
        if A is not None:
            self.transform = A.Compose([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.MotionBlur(blur_limit=5, p=0.3),
                A.Perspective(scale=(0.01, 0.05), p=0.2),
                A.RandomBrightnessContrast(p=0.5),
                A.JpegCompression(quality_lower=50, quality_upper=100, p=0.4)
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def generate_random_drawing(self) -> Tuple[np.ndarray, List[BoundingBox]]:
        # White background
        image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        bboxes = []
        
        # Generate 1-3 beams
        for _ in range(random.randint(1, 3)):
            bx = random.randint(50, self.width - 400)
            by = random.randint(50, self.height - 200)
            blen = random.randint(200, 300)
            bht = random.randint(30, 80)
            beam = SyntheticBeam(bx, by, blen, bht)
            bboxes.extend(beam.draw(image))
            
            # Generate text for this beam
            texts = ["H10@300", "T12@150", "2Y16", "Y20 TOP", "R10-200"]
            t_str = random.choice(texts)
            tx = bx + 10
            ty = by - 10 # Place above beam
            text_el = SyntheticText(tx, ty, t_str, font_scale=0.8, thickness=2)
            bboxes.extend(text_el.draw(image))
            
        return image, bboxes

    def apply_augmentations(self, image: np.ndarray, bboxes: List[BoundingBox]):
        if self.transform is None:
            return image, bboxes

        # Convert bboxes to albumentations format [x_min, y_min, x_max, y_max]
        # Albumentations expects coordinates to be within [0, width] and [0, height] strictly
        height, width = image.shape[:2]
        alb_bboxes = []
        class_labels = []
        valid_bboxes = []
        
        for b in bboxes:
            x1 = max(0, min(width - 1, b.x1))
            y1 = max(0, min(height - 1, b.y1))
            x2 = max(0, min(width - 1, b.x2))
            y2 = max(0, min(height - 1, b.y2))
            if x2 > x1 and y2 > y1:
                alb_bboxes.append([x1, y1, x2, y2])
                class_labels.append(b.class_id)
                valid_bboxes.append(b)
        
        try:
            transformed = self.transform(image=image, bboxes=alb_bboxes, class_labels=class_labels)
            trans_image = transformed['image']
            trans_bboxes = transformed['bboxes']
            
            new_bboxes = []
            for i, tb in enumerate(trans_bboxes):
                orig_b = valid_bboxes[i]
                new_bboxes.append(BoundingBox(int(tb[0]), int(tb[1]), int(tb[2]), int(tb[3]), 
                                              orig_b.class_id, orig_b.text, orig_b.semantic))
            return trans_image, new_bboxes
        except Exception as e:
            # Fallback if bbox transformation fails
            return image, bboxes
