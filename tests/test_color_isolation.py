import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.preprocessing.color_isolation import isolate_colored_annotations


def test_isolate_colored_annotations_removes_grayscale_floor_plan():
    image = np.full((120, 120, 3), 255, dtype=np.uint8)
    image[20:100, 28:31] = (150, 150, 150)
    image[60:63, 10:110] = (80, 80, 80)
    image[35:42, 35:95] = (255, 0, 0)
    image[48:56, 38:82] = (0, 0, 255)

    result = isolate_colored_annotations(image)

    assert result.cleaned[25, 29].tolist() == [255, 255, 255]
    assert result.cleaned[61, 50].tolist() == [255, 255, 255]
    assert result.cleaned[38, 50].tolist() == [255, 0, 0]
    assert result.cleaned[52, 50].tolist() == [0, 0, 255]
    assert result.colored_pixel_count > 0
