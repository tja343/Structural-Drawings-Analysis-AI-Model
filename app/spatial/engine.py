from typing import List, Dict, Any
from app.spatial.geometry import euclidean_distance, calculate_iou
from app.core.logger import logger

class SpatialEngine:
    def __init__(self, distance_threshold: float = 150.0):
        self.distance_threshold = distance_threshold

    def associate_text_to_regions(
        self, 
        texts: List[Dict[str, Any]], 
        regions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Builds a bipartite graph logic to associate Text objects (parsed) 
        to the nearest Structural Region (e.g. Beam, Rebar Region).
        """
        associated_regions = []
        
        # Initialize regions with empty annotations list
        for r in regions:
            r["annotations"] = []
            
        for text_item in texts:
            # Skip empty OCR strings
            if not text_item.get("parsed", False) and text_item.get("text", "") == "":
                continue
                
            text_bbox = text_item["bbox"]
            best_region = None
            min_dist = float('inf')
            
            for region in regions:
                region_bbox = region["bbox"]
                
                # Check for direct overlap first (IoU)
                iou = calculate_iou(text_bbox, region_bbox)
                if iou > 0:
                    best_region = region
                    min_dist = 0 # Overlap takes maximum priority
                    break
                    
                # Otherwise, use nearest neighbor via Centroid Euclidean Distance
                dist = euclidean_distance(text_bbox, region_bbox)
                if dist < min_dist and dist < self.distance_threshold:
                    min_dist = dist
                    best_region = region
            
            if best_region is not None:
                # Calculate an association confidence based on distance
                assoc_confidence = 1.0 if min_dist == 0 else max(0.1, 1.0 - (min_dist / self.distance_threshold))
                
                # Inject association confidence
                text_item["association_confidence"] = round(assoc_confidence, 3)
                best_region["annotations"].append(text_item)
            else:
                logger.warning(f"Orphaned text found: {text_item.get('text', 'UNKNOWN')}. No region within {self.distance_threshold}px.")
                
        # Return regions (which now contain nested annotations)
        return regions
