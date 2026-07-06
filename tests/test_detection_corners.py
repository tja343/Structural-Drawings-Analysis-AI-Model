import numpy as np

from app.api.router import bbox_corners_from_bbox, merge_text_detections
from app.models.detection.inference import approximate_corners, bbox_corners, corners_from_image_region, is_structural_region
from app.synthetic.components import SyntheticBeam, SyntheticTBeam


def test_bbox_corners_returns_four_ordered_points():
    assert bbox_corners(10, 20, 40, 60) == [
        [10, 20],
        [40, 20],
        [40, 60],
        [10, 60],
    ]


def test_contour_approximation_preserves_rectangle_corners():
    rectangle = np.array([[10, 10], [60, 10], [60, 40], [10, 40]])

    assert approximate_corners(rectangle) == [
        [10, 10],
        [60, 10],
        [60, 40],
        [10, 40],
    ]


def test_contour_approximation_preserves_t_shape_corners():
    t_shape = np.array([
        [20, 10],
        [50, 10],
        [50, 30],
        [70, 30],
        [70, 50],
        [0, 50],
        [0, 30],
        [20, 30],
    ])

    corners = approximate_corners(t_shape)

    assert len(corners) == 8
    assert sorted(corners) == sorted(t_shape.astype(int).tolist())


def test_image_region_corners_returns_four_for_rectangle_beam():
    image = np.full((120, 160, 3), 255, dtype=np.uint8)
    bbox = SyntheticBeam(30, 20, 80, 25).draw(image)[0]

    corners = corners_from_image_region(image, [bbox.x1, bbox.y1, bbox.x2, bbox.y2])

    assert len(corners) == 4


def test_image_region_corners_returns_eight_for_t_beam():
    image = np.full((150, 160, 3), 255, dtype=np.uint8)
    bbox = SyntheticTBeam(30, 20, 80, 20, 30, 80).draw(image)[0]

    corners = corners_from_image_region(image, [bbox.x1, bbox.y1, bbox.x2, bbox.y2])

    assert len(corners) == 8


def test_shape_class_is_treated_as_structural_region():
    assert is_structural_region(0, "shape")


def test_ocr_detections_get_corner_fallback():
    merged = merge_text_detections(
        [],
        [{"bbox": [5, 6, 25, 36], "confidence": 0.8, "text": "B1"}],
    )

    assert merged[0]["corners"] == bbox_corners_from_bbox([5, 6, 25, 36])
