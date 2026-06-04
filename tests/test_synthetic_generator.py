import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.synthetic.generator import DrawingGenerator


def test_synthetic_drawing_uses_random_color_annotations():
    generator = DrawingGenerator(width=512, height=512)
    image, bboxes = generator.generate_random_drawing()

    colored_pixels = np.count_nonzero(
        ((image[:, :, 0] < 245) | (image[:, :, 1] < 245) | (image[:, :, 2] < 245))
        & ~((image[:, :, 0] < 40) & (image[:, :, 1] < 40) & (image[:, :, 2] < 40))
    )
    text_boxes = [b for b in bboxes if b.class_id == 0]
    bar_boxes = [b for b in bboxes if b.class_id == 3]

    assert colored_pixels > 0
    assert text_boxes
    assert bar_boxes
    assert all(b.semantic.get("color") for b in text_boxes)
    assert all(b.semantic.get("color") for b in bar_boxes)
