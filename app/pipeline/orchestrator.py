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

class InferenceOrchestrator:
    def __init__(self, yolo_weights: str = None):
        logger.info("Initializing Full Inference Pipeline...")
        yolo_weights = yolo_weights or settings.YOLO_MODEL_PATH
        self.detector = DetectionInference(weights_path=yolo_weights, conf_threshold=0.3)
        self.ocr_service = OCRService()
        self.parser = EngineeringParser()
        self.spatial_engine = SpatialEngine(distance_threshold=150.0)
        self.json_engine = JSONGeneratorEngine()
        
    def process_image(self, drawing_id: str, image: np.ndarray) -> EngineeringOutputSchema:
        logger.info(f"[{drawing_id}] Step 1: Running Object Detection")
        detections = self.detector.predict(image)
        
        text_regions = [d for d in detections if d["class_id"] == 0]
        structural_regions = [d for d in detections if d["class_id"] != 0]
        
        logger.info(f"[{drawing_id}] Step 2: Running OCR on {len(text_regions)} text regions")
        text_bboxes = [t["bbox"] for t in text_regions]
        ocr_results = self.ocr_service.process_image(image, text_bboxes)
        
        logger.info(f"[{drawing_id}] Step 3: Parsing Engineering Semantics")
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
            
        logger.info(f"[{drawing_id}] Step 4: Spatial Association Engine")
        associated_regions = self.spatial_engine.associate_text_to_regions(
            texts=parsed_texts,
            regions=structural_regions
        )
        
        logger.info(f"[{drawing_id}] Step 5: Validating JSON Pydantic Schema")
        final_output = self.json_engine.build_output(drawing_id, associated_regions)
        
        logger.info(f"[{drawing_id}] Pipeline Complete. Overall Confidence: {final_output.overall_confidence}")
        return final_output
