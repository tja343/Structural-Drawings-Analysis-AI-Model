import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import List, Dict, Any
from app.core.logger import logger


def bbox_corners(x1: float, y1: float, x2: float, y2: float) -> List[List[int]]:
    return [
        [int(x1), int(y1)],
        [int(x2), int(y1)],
        [int(x2), int(y2)],
        [int(x1), int(y2)],
    ]


def rotate_to_top_left(corners: List[List[int]]) -> List[List[int]]:
    if not corners:
        return corners
    start_index = min(range(len(corners)), key=lambda idx: (corners[idx][1], corners[idx][0]))
    return corners[start_index:] + corners[:start_index]


def approximate_corners(seg_points: np.ndarray) -> List[List[int]]:
    contour = np.array(seg_points, dtype=np.int32).reshape(-1, 1, 2)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return []

    for epsilon_ratio in (0.01, 0.015, 0.02, 0.03):
        approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
        corners = rotate_to_top_left(approx.reshape(-1, 2).astype(int).tolist())
        if len(corners) >= 4:
            return corners

    return rotate_to_top_left(contour.reshape(-1, 2).astype(int).tolist())


def is_structural_region(class_id: int, class_name: str) -> bool:
    normalized = class_name.strip().lower().replace("_", " ")
    return class_id in {0, 3} or normalized in {"shape", "beam", "rebar region", "rebar"}


def corners_from_image_region(
    image: np.ndarray,
    bbox: List[int],
    padding: int = 4,
    saturation_threshold: int = 35,
    value_threshold: int = 25,
) -> List[List[int]]:
    """Approximate the visible colored shape inside a detected structural bbox."""
    img_h, img_w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop_x1 = max(0, x1 - padding)
    crop_y1 = max(0, y1 - padding)
    crop_x2 = min(img_w - 1, x2 + padding)
    crop_y2 = min(img_h - 1, y2 + padding)
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return []

    crop = image[crop_y1:crop_y2 + 1, crop_x1:crop_x2 + 1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv[:, :, 1], saturation_threshold, 255)
    visible_mask = cv2.inRange(hsv[:, :, 2], value_threshold, 255)
    color_mask = cv2.bitwise_and(color_mask, visible_mask)

    kernel = np.ones((3, 3), dtype=np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    flood_filled = color_mask.copy()
    flood_mask = np.zeros((flood_filled.shape[0] + 2, flood_filled.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood_filled, flood_mask, (0, 0), 255)
    filled_shape = cv2.bitwise_or(color_mask, cv2.bitwise_not(flood_filled))

    contours, _ = cv2.findContours(filled_shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 8:
        return []

    local_corners = approximate_corners(contour.reshape(-1, 2))
    corners = [[x + crop_x1, y + crop_y1] for x, y in local_corners]
    return rotate_to_top_left(corners)


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
                
                width = int(x2 - x1)
                height = int(y2 - y1)
                x_center = int(x1 + width / 2)
                y_center = int(y1 + height / 2)

                det = {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(conf),
                    "class_id": cls_id,
                    "class_name": result.names[cls_id],
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                }
                
                if masks is not None and masks.xy and len(masks.xy) > i:
                    seg_points = masks.xy[i]
                    det["segmentation"] = seg_points.tolist()
                    det["corners"] = approximate_corners(seg_points) or bbox_corners(x1, y1, x2, y2)
                elif is_structural_region(cls_id, result.names[cls_id]):
                    det["corners"] = corners_from_image_region(image, det["bbox"]) or bbox_corners(x1, y1, x2, y2)
                else:
                    # Fallback to bounding box corners if no segmentation mask is available
                    det["corners"] = bbox_corners(x1, y1, x2, y2)
                
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
