import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
import random

class BoundingBox:
    def __init__(self, x1, y1, x2, y2, class_id, text=None, semantic=None):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.class_id = class_id
        self.text = text
        self.semantic = semantic

class SyntheticBeam:
    def __init__(self, x, y, length, height):
        self.x = x
        self.y = y
        self.length = length
        self.height = height

    def draw(self, image: np.ndarray) -> List[BoundingBox]:
        color = (0, 0, 0) # Black
        thickness = 2
        cv2.rectangle(image, (self.x, self.y), (self.x + self.length, self.y + self.height), color, thickness)
        # Class 3 is Beam
        return [BoundingBox(self.x, self.y, self.x + self.length, self.y + self.height, 3)]

class SyntheticText:
    def __init__(self, x, y, text, font_scale=1.0, thickness=2):
        self.x = x
        self.y = y
        self.text = text
        self.font_scale = font_scale
        self.thickness = thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw(self, image: np.ndarray) -> List[BoundingBox]:
        color = (0, 0, 0)
        (text_width, text_height), baseline = cv2.getTextSize(self.text, self.font, self.font_scale, self.thickness)
        
        # Adjust y to be the top-left for bounding box (OpenCV puts text at bottom-left)
        y_bottom = self.y
        y_top = self.y - text_height
        
        cv2.putText(image, self.text, (self.x, self.y), self.font, self.font_scale, color, self.thickness)
        
        # Parse semantics naively for the generator
        semantic = {"raw": self.text}
        if "@" in self.text:
            parts = self.text.split("@")
            semantic = {"bar_type": parts[0], "spacing": int(parts[1]) if parts[1].isdigit() else parts[1]}
            
        # Class 0 is Text
        return [BoundingBox(self.x, y_top, self.x + text_width, y_bottom + baseline, 0, self.text, semantic)]
