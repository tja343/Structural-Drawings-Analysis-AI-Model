from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ColorIsolationResult:
    original: np.ndarray
    color_mask: np.ndarray
    cleaned: np.ndarray
    colored_pixel_count: int
    retained_ratio: float


def isolate_colored_annotations(
    image: np.ndarray,
    saturation_threshold: int = 35,
    value_threshold: int = 25,
    close_kernel_size: int = 3,
) -> ColorIsolationResult:
    """Remove low-saturation grayscale floor-plan pixels and keep colored annotations.

    Floor plans are usually gray or black linework, which has low saturation in HSV.
    Reinforcement bars and text are intentionally colored, so they remain after the
    saturation threshold. The output keeps colored pixels on a white canvas.
    """
    if image is None or image.size == 0:
        raise ValueError("image must be a non-empty OpenCV BGR array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a 3-channel OpenCV BGR array")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    color_mask = cv2.inRange(saturation, saturation_threshold, 255)
    visible_mask = cv2.inRange(value, value_threshold, 255)
    color_mask = cv2.bitwise_and(color_mask, visible_mask)

    if close_kernel_size > 1:
        kernel = np.ones((close_kernel_size, close_kernel_size), dtype=np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    cleaned = np.full_like(image, 255)
    cleaned[color_mask > 0] = image[color_mask > 0]

    colored_pixel_count = int(np.count_nonzero(color_mask))
    retained_ratio = colored_pixel_count / float(color_mask.shape[0] * color_mask.shape[1])
    return ColorIsolationResult(
        original=image.copy(),
        color_mask=color_mask,
        cleaned=cleaned,
        colored_pixel_count=colored_pixel_count,
        retained_ratio=retained_ratio,
    )
