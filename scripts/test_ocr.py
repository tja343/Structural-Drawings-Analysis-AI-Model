import sys
import cv2
from pathlib import Path
import json
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.models.ocr.service import OCRService
from app.models.detection.inference import DetectionInference

def main():
    # 1. Initialize models
    print("Initializing Models...")
    detector = DetectionInference("models/yolov8n.pt", conf_threshold=0.3)
    ocr_service = OCRService()
    
    # 2. Get a sample image (from synthetic dataset)
    sample_img_path = Path("data/synthetic/images/synthetic_draft_0000.png")
    if not sample_img_path.exists():
        print(f"Error: {sample_img_path} not found. Run Phase 2 generation first.")
        sys.exit(1)
        
    print(f"Running inference on {sample_img_path.name}")
    image = cv2.imread(str(sample_img_path))
    
    # 3. Run Detection
    print("Detecting regions...")
    detections = detector.predict(image)
    
    # Filter only Text boxes (class_id == 0)
    text_bboxes = [d["bbox"] for d in detections if d["class_id"] == 0]
    print(f"Found {len(text_bboxes)} text regions.")
    
    # 4. Run OCR on cropped regions
    print("Running OCR...")
    ocr_results = ocr_service.process_image(image, text_bboxes)
    
    # 5. Output results
    print("\n--- OCR Results ---")
    for res in ocr_results:
        print(f"BBox: {res['bbox']} | Text: '{res['text']}' | Conf: {res['confidence']:.3f}")
        
    # Visualize
    vis_img = image.copy()
    for res in ocr_results:
        x1, y1, x2, y2 = res['bbox']
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis_img, res['text'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    out_path = "data/processed/ocr_test_output.png"
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, vis_img)
    print(f"\nVisualization saved to {out_path}")

if __name__ == "__main__":
    main()
