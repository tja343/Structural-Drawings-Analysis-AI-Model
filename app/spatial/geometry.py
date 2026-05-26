import math
from typing import List

def get_centroid(bbox: List[int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def euclidean_distance(bbox1: List[int], bbox2: List[int]) -> float:
    cx1, cy1 = get_centroid(bbox1)
    cx2, cy2 = get_centroid(bbox2)
    return math.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)

def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bb2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou
