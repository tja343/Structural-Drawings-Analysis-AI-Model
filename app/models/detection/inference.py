import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import List, Dict, Any
from app.core.logger import logger

class DetectionInference:
    def __init__(self, weights_path: str, conf_threshold: float = 0.5):
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            logger.warning(f"Weights not found at {weights_path}. Using base yolov8n.pt")
            self.model = YOLO("yolov8n.pt")
        else:
            self.model = YOLO(str(self.weights_path))
        self.conf_threshold = conf_threshold
        
    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLO inference on a single OpenCV image."""
        # ultralytics expects BGR for inference typically
        results = self.model.predict(
            source=image, 
            conf=self.conf_threshold,
            save=False,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            masks = getattr(result, 'masks', None)
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                
                det = {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(conf),
                    "class_id": cls_id,
                    "class_name": result.names[cls_id]
                }
                
                if masks is not None and masks.xy and len(masks.xy) > i:
                    det["segmentation"] = masks.xy[i].tolist()
                
                detections.append(det)
                
        return detections
        
    def draw_predictions(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Visualize predictions on the image, including segmentation masks if available."""
        out_img = image.copy()
        overlay = image.copy()
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)]
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_id = det["class_id"]
            name = det["class_name"]
            conf = det["confidence"]
            color = colors[cls_id % len(colors)]
            
            # Draw segmentation polygon if available
            if "segmentation" in det and det["segmentation"]:
                pts = np.array(det["segmentation"], dtype=np.int32)
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(out_img, [pts], isClosed=True, color=color, thickness=2)
            else:
                cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 2)
            
            label = f"{name} {conf:.2f}"
            cv2.putText(out_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Blend the filled overlay with the original at 30% opacity
        out_img = cv2.addWeighted(overlay, 0.3, out_img, 0.7, 0)
            
        return out_img
