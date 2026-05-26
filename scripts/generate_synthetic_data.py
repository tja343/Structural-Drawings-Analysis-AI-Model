import sys
import os
from pathlib import Path
# Add project root to PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.synthetic.generator import DrawingGenerator
from app.synthetic.export import Exporter
from app.core.config import yaml_config

def main():
    num_samples = 100
    # Use path from config or default
    output_dir = yaml_config.get("paths", {}).get("data_synthetic", "data/synthetic")
    
    print(f"Generating {num_samples} synthetic structural drawings into {output_dir}...")
    
    generator = DrawingGenerator(width=1024, height=1024)
    exporter = Exporter(output_dir)
    
    for i in range(num_samples):
        image_id = f"synthetic_draft_{i:04d}"
        
        # Generate base image and boxes
        image, bboxes = generator.generate_random_drawing()
        
        # Apply domain randomization (noise, blur, scanner artifacts)
        aug_image, aug_bboxes = generator.apply_augmentations(image, bboxes)
        
        # Export PNG, YOLO txt, JSON
        exporter.export(image_id, aug_image, aug_bboxes)
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_samples}")
            
    print("Synthetic dataset generation complete!")

if __name__ == "__main__":
    main()
