import cv2
import numpy as np
from paddleocr import PaddleOCR
from typing import List, Dict, Any, Tuple
from app.core.logger import logger
from app.core.config import settings

class OCRService:
    def __init__(self, lang: str = "en", use_gpu: bool = None):
        self.use_gpu = use_gpu if use_gpu is not None else settings.USE_GPU
        device = "gpu" if self.use_gpu else "cpu"
        logger.info(f"Initializing PaddleOCR (Lang: {lang}, Device: {device})")
        
        self.ocr = PaddleOCR(
            lang=lang,
            device=device,
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )

    def preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """Preprocess image crop to improve OCR accuracy."""
        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Resize if crop is too small (PaddleOCR struggles with tiny text < 32px height)
        h, w = gray.shape
        if h < 32:
            scale = 32.0 / h
            gray = cv2.resize(gray, (int(w * scale), 32), interpolation=cv2.INTER_CUBIC)
            
        # Convert back to 3 channels as PaddleOCR expects RGB/BGR layout
        processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return processed

    def crop_region(self, image: np.ndarray, bbox: List[int], padding: int = 2) -> np.ndarray:
        """Safely crop a region from the image with optional padding."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Apply padding while ensuring we don't go out of image bounds
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(w, int(x2) + padding)
        y2 = min(h, int(y2) + padding)
        
        return image[y1:y2, x1:x2]

    def process_image(self, image: np.ndarray, text_bboxes: List[List[int]]) -> List[Dict[str, Any]]:
        """
        Takes the full image and a list of text bounding boxes (from YOLO).
        Returns OCR text and confidence for each box.
        """
        results = []
        for bbox in text_bboxes:
            # 1. Crop
            crop = self.crop_region(image, bbox)
            
            # Skip invalid crops
            if crop.size == 0:
                continue
                
            # 2. Preprocess
            processed_crop = self.preprocess_crop(crop)
            
            # 3. OCR Inference
            ocr_result = self.ocr.predict(processed_crop)
            
            # 4. Postprocessing
            text = ""
            conf = 0.0
            
            if ocr_result:
                page_result = ocr_result[0]
                extracted_texts = list(page_result.get("rec_texts", []))
                confidences = list(page_result.get("rec_scores", []))

                text = " ".join(extracted_texts)
                conf = sum(confidences) / len(confidences) if confidences else 0.0
                
            results.append({
                "bbox": bbox,
                "text": text,
                "confidence": float(conf)
            })
            
        return results
