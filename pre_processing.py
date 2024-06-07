import cv2
import numpy as np
from PIL import Image



depth = MiDaS()


def get_area_ratio(
    image: Image.Image, object_class: str = None, return_mask: bool = False
) -> np.float32:

    if object_class == "floor":
        mask, label = get_objects_mask(image, classes=["floor"])
    else:
        mask = get_room_transition_masks(image)[:, :, 0]

    if return_mask:
        return mask, round(mask.sum() / (255 * mask.shape[0] * mask.shape[1]), 3)
    return round(mask.sum() / (255 * mask.shape[0] * mask.shape[1]), 3)


def get_max_depth(image: Image.Image) -> np.float32:
    depth_mask = round(float(np.array(depth(np.array(image))).max()), 3)
    return depth_mask


def get_vertical_line_count(image: Image.Image) -> int:
    image = get_mlsd_lines(image)

    # Vertical line must be at least as long as 50% of vertical dim of image
    line_length = int(image.size[1] * 0.1)

    image = np.array(image)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform edge detection using Canny
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Use HoughLinesP to detect lines
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=50, minLineLength=line_length, maxLineGap=10
    )

    # Count and draw the lines
    if lines is not None:
        vertical_lines_count = 0

        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if (
                line_length > 100 and np.abs(x1 - x2) < 10
            ):  # Check if the line is vertical
                vertical_lines_count += 1
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # print("Number of vertical lines longer than 100 pixels:", vertical_lines_count)

        return vertical_lines_count
    return 0


def is_img_good(
    image: Image.Image,
    min_acceptable_floor_area: float = 0.19,
    max_acceptable_trans_area: float = 0.11,
    max_acceptable_depth: float = 50,
    min_acceptable_lines: int = 2,
) -> str:

    floor_area = get_area_ratio(image, object_class="floor")
    lines = get_vertical_line_count(image)
    depth_max = get_max_depth(image)
    metrics_str = f"floor_area: {floor_area}, line: {lines}, depth: {depth_max}"
    # print(metrics_str)
    failing_metric_str = "check_passed"

    if floor_area < min_acceptable_floor_area:
        # print(f"floor_area: {floor_area}")
        failing_metric_str = f"Floor area: {floor_area}"
    if depth_max > max_acceptable_depth:
        # print(f"depth_max: {round(depth_max)}")
        failing_metric_str = f"Max depth: {depth_max}"
    if lines < min_acceptable_lines:
        # print(f"lines: {lines}")
        failing_metric_str = f"Vertical lines: {lines}"

    return failing_metric_str
