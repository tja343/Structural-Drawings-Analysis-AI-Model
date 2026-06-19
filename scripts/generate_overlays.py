import cv2
import glob
import os
import random
from app.synthetic.generator import DrawingGenerator
from app.synthetic.export import Exporter

def main():
    input_dir = "data/floorplan_overlays/images"
    output_dir = "data/synthetic_overlays"
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(input_dir, "*.png"))
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
        
    print(f"Found {len(image_paths)} images. Generating overlays...")
    
    generator = DrawingGenerator()
    exporter = Exporter(output_dir)
    
    for i, img_path in enumerate(image_paths):
        base_img = cv2.imread(img_path)
        if base_img is None:
            print(f"Failed to load {img_path}")
            continue
            
        image_id = f"overlay_{i+1:02d}"
        
        if len(base_img.shape) == 2:
            base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
            
        generator.width = base_img.shape[1]
        generator.height = base_img.shape[0]
        
        # Only draw bars, skip applying augmentations to keep the floor plan crisp
        image, bboxes = generator.generate_random_drawing(base_image=base_img)
        
        exporter.export(image_id, image, bboxes)
        print(f"Generated {image_id}")
        
    print(f"Done! Saved to {output_dir}")

if __name__ == "__main__":
    main()
