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
        
        # 2. Export YOLO labels (Segmentation format)
        yolo_lines = []
        for b in bboxes:
            # YOLO Seg format: class x1 y1 x2 y2 ... xn yn (normalized)
            seg_points = b.segmentation_points
            norm_points = []
            for p in seg_points:
                nx = max(0.0, min(1.0, p[0] / width))
                ny = max(0.0, min(1.0, p[1] / height))
                norm_points.append(f"{nx:.6f} {ny:.6f}")
            
            points_str = " ".join(norm_points)
            yolo_lines.append(f"{b.class_id} {points_str}")
            
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
