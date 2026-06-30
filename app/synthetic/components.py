import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
import re

TEXT_COLOR_PALETTE_BGR = [
    ("red", (0, 0, 255)),
    ("orange", (0, 140, 255)),
    ("blue", (255, 0, 0)),
    ("violet", (211, 0, 148)),
    ("magenta", (255, 0, 255)),
    ("teal", (128, 128, 0)),
]

BAR_COLOR_PALETTE_BGR = [
    ("green", (0, 160, 0)),
    ("violet", (211, 0, 148)),
    ("yellow", (0, 220, 220)),
    ("cyan", (220, 220, 0)),
    ("blue", (255, 0, 0)),
    ("orange", (0, 140, 255)),
]

class BoundingBox:
    def __init__(self, x1, y1, x2, y2, class_id, text=None, semantic=None, segmentation_points=None):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.class_id = class_id
        self.text = text
        self.semantic = semantic
        
        # segmentation_points is a list of [x, y] coordinates defining the polygon
        if segmentation_points is None:
            # Default to the 4 corners of the bounding box if no points are given
            self.segmentation_points = [
                [self.x1, self.y1],
                [self.x2, self.y1],
                [self.x2, self.y2],
                [self.x1, self.y2]
            ]
        else:
            self.segmentation_points = segmentation_points

class SyntheticBeam:
    def __init__(self, x, y, length, height, color_name="blue", color_bgr=(255, 0, 0)):
        self.x = x
        self.y = y
        self.length = length
        self.height = height
        self.color_name = color_name
        self.color_bgr = color_bgr

    def draw(self, image: np.ndarray) -> List[BoundingBox]:
        color = self.color_bgr
        thickness = 2
        cv2.rectangle(image, (self.x, self.y), (self.x + self.length, self.y + self.height), color, thickness)
        # Class 3 is Beam
        return [BoundingBox(
            self.x,
            self.y,
            self.x + self.length,
            self.y + self.height,
            3,
            semantic={"color": self.color_name}
        )]
class SyntheticLBeam:
    def __init__(self, x, y, w_horiz, h_horiz, w_vert, h_vert, color_name="blue", color_bgr=(255, 0, 0)):
        self.x = x
        self.y = y
        self.w_horiz = w_horiz
        self.h_horiz = h_horiz
        self.w_vert = w_vert
        self.h_vert = h_vert
        self.color_name = color_name
        self.color_bgr = color_bgr

    def draw(self, image: np.ndarray) -> List[BoundingBox]:
        thickness = 2
        
        # 6 points defining the L shape
        pts = np.array([
            [self.x, self.y], 
            [self.x + self.w_horiz, self.y], 
            [self.x + self.w_horiz, self.y + self.h_horiz], 
            [self.x + self.w_vert, self.y + self.h_horiz], 
            [self.x + self.w_vert, self.y + self.h_vert], 
            [self.x, self.y + self.h_vert]
        ], np.int32)
        
        cv2.polylines(image, [pts], isClosed=True, color=self.color_bgr, thickness=thickness)
        
        x_coords = pts[:, 0]
        y_coords = pts[:, 1]
        
        return [BoundingBox(
            np.min(x_coords), np.min(y_coords),
            np.max(x_coords), np.max(y_coords),
            3, # Class 3 is Beam
            semantic={"color": self.color_name, "type": "L-Beam"},
            segmentation_points=pts.tolist()
        )]

class SyntheticTBeam:
    def __init__(self, x, y, width, top_height, stem_width, stem_height, color_name="blue", color_bgr=(255, 0, 0)):
        self.x = x
        self.y = y
        self.width = width
        self.top_height = top_height
        self.stem_width = stem_width
        self.stem_height = stem_height
        self.color_name = color_name
        self.color_bgr = color_bgr

    def draw(self, image: np.ndarray) -> List[BoundingBox]:
        thickness = 2
        
        stem_x = self.x + (self.width - self.stem_width) // 2
        
        # 8 points defining the T shape
        pts = np.array([
            [self.x, self.y],
            [self.x + self.width, self.y],
            [self.x + self.width, self.y + self.top_height],
            [stem_x + self.stem_width, self.y + self.top_height],
            [stem_x + self.stem_width, self.y + self.stem_height],
            [stem_x, self.y + self.stem_height],
            [stem_x, self.y + self.top_height],
            [self.x, self.y + self.top_height]
        ], np.int32)
        
        cv2.polylines(image, [pts], isClosed=True, color=self.color_bgr, thickness=thickness)
        
        x_coords = pts[:, 0]
        y_coords = pts[:, 1]
        
        return [BoundingBox(
            np.min(x_coords), np.min(y_coords),
            np.max(x_coords), np.max(y_coords),
            3, # Class 3 is Beam
            semantic={"color": self.color_name, "type": "T-Beam"},
            segmentation_points=pts.tolist()
        )]
class SyntheticText:
    def __init__(self, x, y, text, font_scale=1.0, thickness=2, color_name="red", color_bgr=(0, 0, 255)):
        self.x = x
        self.y = y
        self.text = text
        self.font_scale = font_scale
        self.thickness = thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.color_name = color_name
        self.color_bgr = color_bgr

    def draw(self, image: np.ndarray) -> List[BoundingBox]:
        color = self.color_bgr
        (text_width, text_height), baseline = cv2.getTextSize(self.text, self.font, self.font_scale, self.thickness)
        
        # Adjust y to be the top-left for bounding box (OpenCV puts text at bottom-left)
        y_bottom = self.y
        y_top = self.y - text_height
        
        cv2.putText(image, self.text, (self.x, self.y), self.font, self.font_scale, color, self.thickness)
        
        semantic = self.parse_semantic_label()
        semantic["color"] = self.color_name
            
        # Class 0 is Text
        return [BoundingBox(self.x, y_top, self.x + text_width, y_bottom + baseline, 0, self.text, semantic)]

    def parse_semantic_label(self) -> Dict[str, Any]:
        """Parse labels like 'H12 200 T1' into generator metadata."""
        semantic = {"raw": self.text}
        match = re.match(r"^([A-Z])(\d+)\s+(\d+)\s+([TB])([12])$", self.text.strip(), re.IGNORECASE)
        if match:
            semantic.update({
                "bar_type": match.group(1).upper(),
                "diameter": int(match.group(2)),
                "spacing": int(match.group(3)),
                "layer": match.group(4).upper(),
                "direction": int(match.group(5))
            })
        return semantic
