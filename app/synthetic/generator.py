import cv2
import numpy as np
import random
from typing import List, Tuple, Dict
from app.synthetic.components import (
    BAR_COLOR_PALETTE_BGR,
    TEXT_COLOR_PALETTE_BGR,
    SyntheticBeam,
    SyntheticLBeam,
    SyntheticTBeam,
    SyntheticText,
    BoundingBox,
)

try:
    import albumentations as A
except ImportError:
    A = None

def boxes_overlap(b1: BoundingBox, b2: BoundingBox) -> bool:
    return not (b1.x2 <= b2.x1 or b1.x1 >= b2.x2 or b1.y2 <= b2.y1 or b1.y1 >= b2.y2)

class DrawingGenerator:
    def __init__(self, width=1024, height=1024):
        self.width = width
        self.height = height
        
        # Albumentations pipeline for domain randomization
        self.transform = None
        if A is not None:
            self.transform = A.Compose([
                A.Perspective(scale=(0.01, 0.05), p=0.2),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']),
               keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def generate_random_drawing(self, base_image: np.ndarray = None) -> Tuple[np.ndarray, List[BoundingBox]]:
        if base_image is not None:
            image = base_image.copy()
            height, width = image.shape[:2]
        else:
            # White background
            image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
            height, width = self.height, self.width
            
        bboxes = []
        
        num_beams = random.randint(1, 4)
        attempts = 0
        beams_generated = 0
        max_attempts = 50
        
        while beams_generated < num_beams and attempts < max_attempts:
            attempts += 1
            beam_type = random.choice(["straight", "L", "T"])
            bar_color_name, bar_color_bgr = random.choice(BAR_COLOR_PALETTE_BGR)
            
            bx = random.randint(50, max(51, width - 400))
            by = random.randint(50, max(51, height - 400))
            
            if beam_type == "straight":
                blen = random.randint(200, 300)
                bht = random.randint(30, 80)
                beam = SyntheticBeam(bx, by, blen, bht, color_name=bar_color_name, color_bgr=bar_color_bgr)
            elif beam_type == "L":
                w_horiz = random.randint(150, 250)
                h_horiz = random.randint(40, 60)
                w_vert = random.randint(40, 60)
                h_vert = random.randint(150, 250)
                beam = SyntheticLBeam(bx, by, w_horiz, h_horiz, w_vert, h_vert, color_name=bar_color_name, color_bgr=bar_color_bgr)
            else: # T
                bwidth = random.randint(200, 300)
                top_h = random.randint(40, 60)
                stem_w = random.randint(40, 60)
                stem_h = random.randint(100, 200)
                beam = SyntheticTBeam(bx, by, bwidth, top_h, stem_w, stem_h, color_name=bar_color_name, color_bgr=bar_color_bgr)

            # Generate text for this beam
            t_str = self.generate_reinforcement_label()
            tx = bx + 10
            ty = by - 10 # Place above beam
            text_color_name, text_color_bgr = random.choice(TEXT_COLOR_PALETTE_BGR)
            text_el = SyntheticText(
                tx,
                ty,
                t_str,
                font_scale=0.8,
                thickness=2,
                color_name=text_color_name,
                color_bgr=text_color_bgr,
            )
            
            # Check overlap on a dummy image
            dummy_image = np.zeros_like(image)
            temp_bboxes = beam.draw(dummy_image)
            temp_bboxes.extend(text_el.draw(dummy_image))
            
            # Check if any temp bbox overlaps with any existing bbox
            overlap = False
            for tb in temp_bboxes:
                # Add a small margin (e.g. 10 pixels) for aesthetics
                tb_margin = BoundingBox(tb.x1 - 10, tb.y1 - 10, tb.x2 + 10, tb.y2 + 10, tb.class_id)
                for eb in bboxes:
                    if boxes_overlap(tb_margin, eb):
                        overlap = True
                        break
                if overlap:
                    break
                    
            if not overlap:
                # Draw on actual image
                actual_bboxes = beam.draw(image)
                actual_bboxes.extend(text_el.draw(image))
                bboxes.extend(actual_bboxes)
                beams_generated += 1
            
        return image, bboxes

    def generate_reinforcement_label(self) -> str:
        """Generate labels like 'H12 200 T1' for synthetic beam annotations."""
        bar_type = random.choice(["H", "Y", "T", "R"])
        diameter = random.choice([8, 10, 12, 16, 20, 25, 32])
        spacing = random.choice([100, 125, 150, 175, 200, 225, 250, 300])
        layer = random.choice(["T", "B"])
        direction = random.choice([1, 2])
        return f"{bar_type}{diameter} {spacing} {layer}{direction}"

    def apply_augmentations(self, image: np.ndarray, bboxes: List[BoundingBox]):
        if self.transform is None:
            return image, bboxes

        # Convert bboxes to albumentations format [x_min, y_min, x_max, y_max]
        # Albumentations expects coordinates to be within [0, width] and [0, height] strictly
        height, width = image.shape[:2]
        alb_bboxes = []
        class_labels = []
        valid_bboxes = []
        
        all_keypoints = []
        keypoint_counts = []
        
        for b in bboxes:
            x1 = max(0, min(width - 1, b.x1))
            y1 = max(0, min(height - 1, b.y1))
            x2 = max(0, min(width - 1, b.x2))
            y2 = max(0, min(height - 1, b.y2))
            if x2 > x1 and y2 > y1:
                alb_bboxes.append([x1, y1, x2, y2])
                class_labels.append(b.class_id)
                valid_bboxes.append(b)
                
                pts = b.segmentation_points
                # Clip points to be within image bounds just in case
                pts = [[max(0, min(width - 1, p[0])), max(0, min(height - 1, p[1]))] for p in pts]
                all_keypoints.extend(pts)
                keypoint_counts.append(len(pts))
        
        try:
            transformed = self.transform(image=image, bboxes=alb_bboxes, class_labels=class_labels, keypoints=all_keypoints)
            trans_image = transformed['image']
            trans_bboxes = transformed['bboxes']
            trans_keypoints = transformed['keypoints']
            
            new_bboxes = []
            kp_idx = 0
            for i, tb in enumerate(trans_bboxes):
                orig_b = valid_bboxes[i]
                pts_count = keypoint_counts[i]
                new_pts = trans_keypoints[kp_idx:kp_idx + pts_count]
                kp_idx += pts_count
                
                # convert to lists of ints
                new_pts = [[int(x), int(y)] for x, y in new_pts]
                
                new_bboxes.append(BoundingBox(int(tb[0]), int(tb[1]), int(tb[2]), int(tb[3]), 
                                              orig_b.class_id, orig_b.text, orig_b.semantic,
                                              segmentation_points=new_pts))
            return trans_image, new_bboxes
        except Exception as e:
            # Fallback if bbox transformation fails
            return image, bboxes
