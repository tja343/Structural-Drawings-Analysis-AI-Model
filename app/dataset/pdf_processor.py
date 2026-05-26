import fitz  # PyMuPDF
import cv2
import numpy as np
from pathlib import Path
from typing import List
from app.core.logger import logger

class PDFProcessor:
    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        self.zoom = dpi / 72.0  # 72 is standard PDF DPI
        self.mat = fitz.Matrix(self.zoom, self.zoom)

    def pdf_to_images(self, pdf_path: str, output_dir: str) -> List[str]:
        """Converts a PDF into high-res PNG images for dataset processing."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
        logger.info(f"Processing PDF: {pdf_path.name}")
        doc = fitz.open(pdf_path)
        output_files = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=self.mat, alpha=False)
            
            # Convert PyMuPDF pixmap to OpenCV image format
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
            out_filename = f"{pdf_path.stem}_page_{page_num+1:03d}.png"
            out_path = output_dir / out_filename
            cv2.imwrite(str(out_path), img)
            output_files.append(str(out_path))
            
            logger.info(f"Saved {out_filename}")
            
        doc.close()
        return output_files
