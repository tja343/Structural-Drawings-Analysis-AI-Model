from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from app.schemas.api import InferenceResponse, BatchInferenceResponse
from app.core.logger import logger
import tempfile
import os

router = APIRouter()
_orchestrator = None
_pdf_processor = None


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from app.pipeline.orchestrator import InferenceOrchestrator

        _orchestrator = InferenceOrchestrator()
    return _orchestrator


def get_pdf_processor():
    global _pdf_processor
    if _pdf_processor is None:
        from app.dataset.pdf_processor import PDFProcessor

        _pdf_processor = PDFProcessor(dpi=300)
    return _pdf_processor

@router.post("/inference/image", response_model=InferenceResponse)
async def process_image_upload(file: UploadFile = File(...)):
    """Upload a structural drawing image for end-to-end processing."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    import cv2
    import numpy as np
        
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
        
    drawing_id = file.filename or "uploaded_image"
    
    try:
        orchestrator = get_orchestrator()
        result = orchestrator.process_image(drawing_id, image)
        return InferenceResponse(status="success", message="Inference complete", data=result)
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/inference/pdf", response_model=BatchInferenceResponse)
async def process_pdf_upload(file: UploadFile = File(...)):
    """Upload a structural PDF, extract high-res pages, and process each."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")

    import cv2
        
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, file.filename)
        with open(pdf_path, "wb") as f:
            f.write(await file.read())
            
        try:
            orchestrator = get_orchestrator()
            pdf_processor = get_pdf_processor()

            # 1. Convert PDF to PNGs
            image_paths = pdf_processor.pdf_to_images(pdf_path, tmpdir)
            
            # 2. Process each page through orchestrator
            results = []
            for img_path in image_paths:
                page_img = cv2.imread(img_path)
                page_id = os.path.basename(img_path)
                page_res = orchestrator.process_image(page_id, page_img)
                results.append(page_res)
                
            return BatchInferenceResponse(
                status="success", 
                message="PDF batch processing complete", 
                processed_count=len(results),
                data=results
            )
        except Exception as e:
            logger.error(f"PDF Inference failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
