import os
import json
import cv2
from pathlib import Path
from typing import List, Dict, Any
from app.synthetic.components import BoundingBox

class Exporter:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.yolo_dir = self.output_dir / "labels"
        self.json_dir = self.output_dir / "semantics"
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.yolo_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir.mkdir(parents=True, exist_ok=True)

    def export(self, image_id: str, image, bboxes: List[BoundingBox]):
        # 1. Export Image
        img_path = self.images_dir / f"{image_id}.png"
        cv2.imwrite(str(img_path), image)
        
        height, width, _ = image.shape
        
        # 2. Export YOLO labels
        yolo_lines = []
        for b in bboxes:
            # YOLO format: class x_center y_center width height (normalized)
            bw = b.x2 - b.x1
            bh = b.y2 - b.y1
            x_center = (b.x1 + bw / 2.0) / width
            y_center = (b.y1 + bh / 2.0) / height
            norm_w = bw / width
            norm_h = bh / height
            
            # Clip between 0 and 1
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))
            
            yolo_lines.append(f"{b.class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
            
        with open(self.yolo_dir / f"{image_id}.txt", "w") as f:
            f.write("\n".join(yolo_lines))
            
        # 3. Export JSON Metadata (OCR & Semantics)
        metadata = {
            "image_id": image_id,
            "annotations": []
        }
        for b in bboxes:
            if b.class_id == 0: # Text
                metadata["annotations"].append({
                    "bbox": [b.x1, b.y1, b.x2, b.y2],
                    "text": b.text,
                    "semantic": b.semantic
                })
                
        with open(self.json_dir / f"{image_id}.json", "w") as f:
            json.dump(metadata, f, indent=2)
