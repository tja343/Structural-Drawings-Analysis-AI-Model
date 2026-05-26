import sys
import shutil
from pathlib import Path
import random
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.core.config import yaml_config
from app.dataset.pdf_processor import PDFProcessor

def prepare_splits(source_dir: Path, out_dir: Path, train_ratio=0.8, val_ratio=0.1):
    img_dir = source_dir / "images"
    lbl_dir = source_dir / "labels"
    
    if not img_dir.exists():
        print(f"Source dir {img_dir} does not exist. Run synthetic generator first.")
        return
        
    images = list(img_dir.glob("*.png"))
    random.shuffle(images)
    
    train_split = int(len(images) * train_ratio)
    val_split = train_split + int(len(images) * val_ratio)
    
    splits = {
        "train": images[:train_split],
        "val": images[train_split:val_split],
        "test": images[val_split:]
    }
    
    for split, files in splits.items():
        split_img_dir = out_dir / split / "images"
        split_lbl_dir = out_dir / split / "labels"
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for img_file in files:
            lbl_file = lbl_dir / f"{img_file.stem}.txt"
            
            shutil.copy(img_file, split_img_dir / img_file.name)
            if lbl_file.exists():
                shutil.copy(lbl_file, split_lbl_dir / lbl_file.name)
                
    print(f"Dataset split complete in {out_dir}")

def process_pdfs():
    raw_dir = Path(yaml_config.get("paths", {}).get("data_raw", "data/raw"))
    proc_dir = Path(yaml_config.get("paths", {}).get("data_processed", "data/processed"))
    pdf_processor = PDFProcessor(dpi=300)
    
    if not raw_dir.exists():
        raw_dir.mkdir(parents=True)
        print(f"Created raw directory {raw_dir}. Drop real PDF drawings here for processing.")
        return
        
    pdfs = list(raw_dir.glob("*.pdf"))
    for pdf in pdfs:
        print(f"Converting PDF to Images: {pdf.name}")
        pdf_processor.pdf_to_images(str(pdf), str(proc_dir / "images"))

def main():
    print("Step 1: Processing Real PDFs into High-DPI PNGs...")
    process_pdfs()
    
    print("\nStep 2: Generating Train/Val/Test Splits for Synthetic Data...")
    synth_dir = Path(yaml_config.get("paths", {}).get("data_synthetic", "data/synthetic"))
    yolo_dir = Path("data/yolo")
    prepare_splits(synth_dir, yolo_dir)

if __name__ == "__main__":
    main()
