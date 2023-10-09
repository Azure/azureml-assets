# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Functions for managing segmentation masks for instance segmentation."""
import numpy
from typing import List, Tuple
from simplification.cutil import simplify_coords
from skimage import measure


class MaskToolsParameters:
    """Default values for mask tool parameters."""

    DEFAULT_MAX_NUMBER_OF_POLYGON_POINTS = 100
    DEFAULT_MAX_NUMBER_OF_POLYGON_SIMPLIFICATIONS = 25
    DEFAULT_MASK_SAFETY_PADDING = 1


def convert_mask_to_polygon(
    mask_array: numpy.ndarray,
    max_polygon_points: int = MaskToolsParameters.DEFAULT_MAX_NUMBER_OF_POLYGON_POINTS,
    max_refinement_iterations: int = MaskToolsParameters.DEFAULT_MAX_NUMBER_OF_POLYGON_SIMPLIFICATIONS,
    edge_safety_padding: int = MaskToolsParameters.DEFAULT_MASK_SAFETY_PADDING,
) -> List[List]:
    """Convert a run length encoded mask to a polygon outline in normalized coordinates.

    :param mask_array: Binary mask
    :type: rle_mask: numpy.ndarray
    :param max_polygon_points: Maximum number of (x, y) coordinate pairs in polygon
    :type: max_polygon_points: int
    :param max_refinement_iterations: Maximum number of times to refine the polygon
    trying to reduce the number of pixels to meet max polygon points.
    :type: max_refinement_iterations: int
    :param edge_safety_padding: Number of pixels to pad the mask with
    :type edge_safety_padding: int
    :return: normalized polygon coordinates
    :rtype: list of list
    """
    # Convert rle mask to numpy bitmask
    image_shape = mask_array.shape

    # Pad the mask to avoid errors at the edge of the mask
    embedded_mask = numpy.zeros(
        (image_shape[0] + 2 * edge_safety_padding, image_shape[1] + 2 * edge_safety_padding), dtype=numpy.uint8
    )
    embedded_mask[
        edge_safety_padding: image_shape[0] + edge_safety_padding,
        edge_safety_padding: image_shape[1] + edge_safety_padding,
    ] = mask_array

    # Find Image Contours
    contours = measure.find_contours(embedded_mask, 0.5)
    simplified_contours = []

    for contour in contours:
        # Iteratively reduce polygon points, if necessary
        if max_polygon_points is not None:
            simplify_factor = 0
            while len(contour) > max_polygon_points and simplify_factor < max_refinement_iterations:
                contour = simplify_coords(contour, simplify_factor)
                simplify_factor += 1

        # Convert to [x, y, x, y, ....] coordinates and correct for padding
        unwrapped_contour = [0] * (2 * len(contour))
        unwrapped_contour[::2] = numpy.ceil(contour[:, 1]) - edge_safety_padding
        unwrapped_contour[1::2] = numpy.ceil(contour[:, 0]) - edge_safety_padding

        simplified_contours.append(unwrapped_contour)

    contours = _normalize_contour(simplified_contours, image_shape)
    return _remove_invalid_contours(contours)


def _normalize_contour(contours: List, image_shape: Tuple[int, int]) -> List:
    """Normalize the contour coordinates to be between 0 and 1.

    :param contours: List of contours
    :type: contours: list
    :param image_shape: Shape of the image Height, Width
    :type: image_shape: tuple
    :return: Normalized contours
    :rtype: list
    Returns:
        _type_: _description_
    """
    height, width = image_shape[0], image_shape[1]

    for contour in contours:
        contour[::2] = [x * 1.0 / width for x in contour[::2]]
        contour[1::2] = [y * 1.0 / height for y in contour[1::2]]

    return contours


def _remove_invalid_contours(contours: List[List[float]]) -> List[List[float]]:
    """Remove invalid contours. Contours is valid if it contains more than or equal to 6 co-ordinates (triangle).
    :param contours: List of contours.
    :return: List of contours.
    """
    return [contour for contour in contours if len(contour) >= 6]
