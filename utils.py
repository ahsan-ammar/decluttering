import cv2
from PIL import Image, ImageOps


def set_img_dims(img, max_dim=1024):
    """
    Resizes the image so that its largest dimension is equal to max_dim while 
    maintaining the aspect ratio.

    Args:
        img (PIL.Image): The input image.
        max_dim (int): The maximum dimension size (default is 1024).

    Returns:
        PIL.Image: The resized image.
    """
    w, h = img.size
    scaler = min(w, h) / max_dim
    img = img.resize((int(w / scaler), int(h / scaler)))
    return img


def get_padded_mask(mask, padding):
    padded_mask = ImageOps.expand(mask, border=padding, fill=255)
    return padded_mask