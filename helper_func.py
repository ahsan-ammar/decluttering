import io
import base64

import cv2
import numpy as np
from PIL import Image

def decode_base64_image(image_string):
    """
    Decode a base64 encoded image string into a PIL image.
    Args:
        image_string (str): Base64 encoded image string.
    Returns:
        PIL.Image: Decoded image.
    """
    image_binary = base64.b64decode(image_string)
    image = Image.open(io.BytesIO(image_binary))
    return image


def image_to_base64(output_image, ext=".jpg"):
    """
    Encode an image to a base64 string.
    Args:
        output_image (np.ndarray): Image to encode.
        ext (str): File extension for the image encoding.
    Returns:
        str: Base64 encoded image string.
    """
    _, encoded_image = cv2.imencode(ext, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    image_string = base64.b64encode(encoded_image).decode("utf-8")
    return image_string