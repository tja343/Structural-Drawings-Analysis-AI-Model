import cv2
import numpy as np
from typing import Dict, Any, List
from app.models.detection.inference import DetectionInference
from app.models.ocr.service import OCRService
from app.parsing.regex_parser import EngineeringParser
from app.spatial.engine import SpatialEngine
from app.exporters.json_engine import JSONGeneratorEngine
from app.schemas.engineering import EngineeringOutputSchema
from app.core.config import settings
from app.core.logger import logger
from app.preprocessing.color_isolation import isolate_colored_annotations
from app.spatial.geometry import calculate_iou

class InferenceOrchestrator:
    def __init__(self, yolo_weights: str = None):
        logger.info("Initializing Full Inference Pipeline...")
        yolo_weights = yolo_weights or settings.YOLO_MODEL_PATH
        self.detector = DetectionInference(weights_path=yolo_weights, conf_threshold=0.15)
        self.ocr_service = OCRService()
        self.parser = EngineeringParser()
        self.spatial_engine = SpatialEngine(distance_threshold=150.0)
        self.json_engine = JSONGeneratorEngine()

    def _dedupe_detections(self, detections: List[Dict[str, Any]], iou_threshold: float = 0.35) -> List[Dict[str, Any]]:
        kept = []
        for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
            duplicate = False
            for existing in kept:
                if detection["class_id"] != existing["class_id"]:
                    continue

                if calculate_iou(detection["bbox"], existing["bbox"]) >= iou_threshold:
                    duplicate = True
                    break

            if not duplicate:
                kept.append(detection)

        return kept
        
    def process_image(self, drawing_id: str, image: np.ndarray) -> EngineeringOutputSchema:
        logger.info(f"[{drawing_id}] Removing grayscale floor-plan background")
        preprocessing = isolate_colored_annotations(image)
        logger.info(
            f"[{drawing_id}] Retained {preprocessing.colored_pixel_count} colored pixels "
            f"({preprocessing.retained_ratio:.4f} of image)"
        )
        model_image = preprocessing.cleaned

        logger.info(f"[{drawing_id}] Running object detection")
        detections = self._dedupe_detections(self.detector.predict(model_image))
        
        structural_regions = [d for d in detections if d["class_id"] != 0]
        
        logger.info(f"[{drawing_id}] Running full-image OCR")
        ocr_results = self.ocr_service.process_full_image(model_image)
        
        logger.info(f"[{drawing_id}] Parsing engineering annotations")
        parsed_texts = []
        for ocr_res in ocr_results:
            raw_text = ocr_res["text"]
            parsed_data = self.parser.parse(raw_text)
            
            parsed_texts.append({
                "bbox": ocr_res["bbox"],
                "text": raw_text,
                "confidence": ocr_res["confidence"],
                "parsed_data": parsed_data,
                "parsed": parsed_data["parsed"]
            })
            
        logger.info(f"[{drawing_id}] Associating annotations with structural regions")
        associated_regions = self.spatial_engine.associate_text_to_regions(
            texts=parsed_texts,
            regions=structural_regions
        )
        
        logger.info(f"[{drawing_id}] Building validated JSON output")
        final_output = self.json_engine.build_output(drawing_id, associated_regions)
        
        logger.info(f"[{drawing_id}] Pipeline complete. Overall confidence: {final_output.overall_confidence}")
        return final_output
