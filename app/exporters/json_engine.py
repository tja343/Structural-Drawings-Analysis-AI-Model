import json
from typing import List, Dict, Any
from app.schemas.engineering import (
    EngineeringOutputSchema, 
    StructuralElementSchema, 
    AnnotationSchema, 
    ParsedSemanticSchema,
    BoundingBoxSchema
)

class JSONGeneratorEngine:
    def __init__(self):
        # Maps YOLO class IDs to human readable types
        self.class_map = {
            1: "rebar_region",
            2: "arrow",
            3: "beam",
            4: "dimension",
            5: "support"
        }

    def _calculate_overall_confidence(self, elements: List[StructuralElementSchema]) -> float:
        """Aggregate confidences across detection, OCR, and spatial association."""
        if not elements:
            return 0.0
            
        confs = []
        for el in elements:
            confs.append(el.detection_confidence)
            for ann in el.annotations:
                confs.append(ann.ocr_confidence)
                confs.append(ann.association_confidence)
                
        return round(sum(confs) / len(confs), 3)

    def build_output(self, drawing_id: str, regions: List[Dict[str, Any]]) -> EngineeringOutputSchema:
        """
        Converts the raw nested dictionary from the Spatial Engine into 
        a strictly validated Pydantic Schema.
        """
        elements = []
        
        for r in regions:
            class_id = r["class_id"]
            if class_id == 0:
                continue # Skip raw floating text that wasn't associated
                
            element_type = self.class_map.get(class_id, "unknown")
            r_bbox = r["bbox"]
            
            annotations = []
            for ann in r.get("annotations", []):
                ann_bbox = ann["bbox"]
                
                # Build semantic object
                parsed_data = ann.get("parsed_data", {})
                semantic = ParsedSemanticSchema(
                    bar_type=parsed_data.get("bar_type"),
                    diameter=parsed_data.get("diameter"),
                    spacing=parsed_data.get("spacing"),
                    quantity=parsed_data.get("quantity"),
                    layer=parsed_data.get("layer")
                )
                
                annotation = AnnotationSchema(
                    bbox=BoundingBoxSchema(x1=ann_bbox[0], y1=ann_bbox[1], x2=ann_bbox[2], y2=ann_bbox[3]),
                    text=ann.get("text", ""),
                    parsed=semantic,
                    ocr_confidence=ann.get("confidence", 0.0),
                    association_confidence=ann.get("association_confidence", 0.0)
                )
                annotations.append(annotation)
                
            element = StructuralElementSchema(
                type=element_type,
                bbox=BoundingBoxSchema(x1=r_bbox[0], y1=r_bbox[1], x2=r_bbox[2], y2=r_bbox[3]),
                detection_confidence=r.get("confidence", 0.0),
                annotations=annotations
            )
            elements.append(element)
            
        overall_conf = self._calculate_overall_confidence(elements)
        
        output = EngineeringOutputSchema(
            drawing_id=drawing_id,
            overall_confidence=overall_conf,
            elements=elements
        )
        
        return output
        
    def export_to_file(self, output: EngineeringOutputSchema, filepath: str):
        with open(filepath, "w") as f:
            f.write(output.model_dump_json(indent=2))
