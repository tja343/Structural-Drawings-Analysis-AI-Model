import cv2
import numpy as np
import re
from typing import List, Dict, Any, Tuple
from app.core.logger import logger
from app.core.config import settings

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

class OCRService:
    def __init__(self, lang: str = "en", use_gpu: bool = None):
        self.use_gpu = use_gpu if use_gpu is not None else settings.USE_GPU
        self.backend = None
        self.reader = None
        self.ocr = None

        if easyocr is not None:
            logger.info(f"Initializing EasyOCR (Lang: {lang}, GPU: {self.use_gpu})")
            self.reader = easyocr.Reader([lang], gpu=self.use_gpu)
            self.backend = "easyocr"
        elif PaddleOCR is not None:
            device = self._resolve_paddle_device()
            logger.info(f"Initializing PaddleOCR (Lang: {lang}, Device: {device})")
            self.ocr = PaddleOCR(
                lang=lang,
                device=device,
                enable_mkldnn=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
            )
            self.backend = "paddleocr"
        else:
            raise ImportError("No OCR backend available. Install easyocr or paddleocr.")

    def _resolve_paddle_device(self) -> str:
        if not self.use_gpu:
            return "cpu"

        try:
            import paddle

            has_cuda = paddle.is_compiled_with_cuda()
            device_count = paddle.device.cuda.device_count() if has_cuda else 0
        except Exception as exc:
            logger.warning(f"Could not inspect Paddle GPU support; using CPU for OCR: {exc}")
            self.use_gpu = False
            return "cpu"

        if not has_cuda or device_count < 1:
            logger.warning("Paddle GPU requested but unavailable; using CPU for OCR")
            self.use_gpu = False
            return "cpu"

        return "gpu"

    def _extract_easyocr_results(self, ocr_result, scale: float = 1.0) -> List[Dict[str, Any]]:
        """Convert EasyOCR results to the standard format used by the pipeline."""
        if not ocr_result:
            return []

        results = []
        for detection in ocr_result:
            # EasyOCR returns: (bbox_points, text, confidence)
            # bbox_points is a list of 4 corner points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            bbox_points, text, confidence = detection

            # Convert polygon points to axis-aligned bounding box
            xs = [p[0] for p in bbox_points]
            ys = [p[1] for p in bbox_points]
            x1, y1 = int(min(xs) / scale), int(min(ys) / scale)
            x2, y2 = int(max(xs) / scale), int(max(ys) / scale)

            results.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "text": self.normalize_text(text),
                    "confidence": float(confidence),
                }
            )

        return results

    def _extract_paddle_results(self, ocr_result, scale: float = 1.0) -> List[Dict[str, Any]]:
        """Convert PaddleOCR results to the standard format used by the pipeline."""
        if not ocr_result:
            return []

        page_result = ocr_result[0]
        texts = list(page_result.get("rec_texts", []))
        confidences = list(page_result.get("rec_scores", []))
        boxes = page_result.get("rec_boxes", [])

        results = []
        for idx, text in enumerate(texts):
            if idx < len(boxes):
                box = boxes[idx]
                if hasattr(box, "tolist"):
                    box = box.tolist()
                if len(box) == 4 and not isinstance(box[0], (list, tuple)):
                    x1, y1, x2, y2 = [int(v / scale) for v in box]
                else:
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    x1, y1 = int(min(xs) / scale), int(min(ys) / scale)
                    x2, y2 = int(max(xs) / scale), int(max(ys) / scale)
            else:
                x1 = y1 = x2 = y2 = 0

            results.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "text": self.normalize_text(text),
                    "confidence": float(confidences[idx]) if idx < len(confidences) else 0.0,
                }
            )

        return results

    def _run_ocr(self, image: np.ndarray):
        if self.backend == "easyocr":
            return self.reader.readtext(
                image,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@- ",
                paragraph=False,
                detail=1,
            )
        return self.ocr.predict(image)

    def _extract_results(self, ocr_result, scale: float = 1.0) -> List[Dict[str, Any]]:
        if self.backend == "easyocr":
            return self._extract_easyocr_results(ocr_result, scale=scale)
        return self._extract_paddle_results(ocr_result, scale=scale)

    def normalize_text(self, text: str) -> str:
        """Normalize OCR spacing without changing engineering meaning."""
        text = (text or "").upper()
        text = re.sub(r"[^A-Z0-9@\-\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\b([A-Z])\s+(\d+)\b", r"\1\2", text)
        text = re.sub(r"\b([TB])\s+([123])\b", r"\1\2", text)
        return text

    def preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """Preprocess image crop to improve OCR accuracy."""
        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Resize if crop is too small (OCR struggles with tiny text < 32px height)
        h, w = gray.shape
        if h < 32:
            scale = 32.0 / h
            gray = cv2.resize(gray, (int(w * scale), 32), interpolation=cv2.INTER_CUBIC)

        # Convert back to 3 channels as EasyOCR expects RGB/BGR layout
        processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return processed

    def crop_region(self, image: np.ndarray, bbox: List[int], padding: int = 8) -> np.ndarray:
        """Safely crop a region from the image with optional padding."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox

        # Apply padding while ensuring we don't go out of image bounds
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(w, int(x2) + padding)
        y2 = min(h, int(y2) + padding)

        return image[y1:y2, x1:x2]

    def _ocr_variants(self, image: np.ndarray):
        """Yield image variants that help EasyOCR with colored CAD annotations."""
        yield image, 1.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        scale = 2.0 if min(h, w) < 900 else 1.5
        if scale > 1:
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        yield cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), scale

        _, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
        thresh = cv2.dilate(thresh, np.ones((2, 2), np.uint8), iterations=1)
        yield cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), scale

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
            ocr_result = []
            for variant, scale in self._ocr_variants(processed_crop):
                ocr_result = self._run_ocr(variant)
                if ocr_result:
                    break

            # 4. Postprocessing
            text = ""
            conf = 0.0

            extracted_results = self._extract_results(ocr_result, scale=scale if ocr_result else 1.0)
            if extracted_results:
                extracted_texts = [item["text"] for item in extracted_results]
                confidences = [item["confidence"] for item in extracted_results]

                text = self.normalize_text(" ".join(extracted_texts))
                conf = sum(confidences) / len(confidences) if confidences else 0.0

            results.append({
                "bbox": bbox,
                "text": text,
                "confidence": float(conf)
            })

        return results

    def process_full_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run OCR over the cleaned drawing and return text with image-space boxes."""
        collected = []
        for variant, scale in self._ocr_variants(image):
            ocr_result = self._run_ocr(variant)
            collected.extend(self._extract_results(ocr_result, scale=scale))

        return self._dedupe_results(collected)

    def _dedupe_results(self, results: List[Dict[str, Any]], iou_threshold: float = 0.45) -> List[Dict[str, Any]]:
        kept = []
        for item in sorted(results, key=lambda r: r["confidence"], reverse=True):
            duplicate = False
            for existing in kept:
                if self._iou(item["bbox"], existing["bbox"]) >= iou_threshold:
                    duplicate = True
                    break
            if not duplicate and item["text"]:
                kept.append(item)
        return kept

    def _iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        return intersection / float(area1 + area2 - intersection)
