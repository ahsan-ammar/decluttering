import cv2
import numpy as np
from PIL import Image, ImageOps

import torch
from ade import TARGET_CLASSES, ADE_CLASSES
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation

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


class OneFormer:
    def __init__(self, use_cuda: bool = True) -> None:
        """
        Initializes the OneFormer class with an option to use CUDA for processing.

        Args:
            use_cuda (bool): Flag to determine whether to use GPU (default is True).
        """
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self) -> None:
        """
        Loads the OneFormer model and processor for universal segmentation.
        """
        self.processor = OneFormerProcessor.from_pretrained(
            "shi-labs/oneformer_ade20k_swin_large"
        )
        self.model = OneFormerForUniversalSegmentation.from_pretrained(
            "shi-labs/oneformer_ade20k_swin_large"
        ).to(self.device)

    def __call__(self, image):
        """
        Performs semantic segmentation on the input image.

        Args:
            image (PIL.Image or np.ndarray): The input image to be segmented.

        Returns:
            np.ndarray: The predicted semantic segmentation map.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        with torch.no_grad():
            inputs = self.processor(
                images=image, task_inputs=["semantic"], return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**inputs)
            predicted_semantic_map = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[image.size[::-1]]
            )[0]
            return predicted_semantic_map.cpu().numpy()


# Instantiate the OneFormer model
MODEL = OneFormer()

def get_init_mask(class_mask, classes=[], return_labels=False):
    """
    Generates an initial mask based on the class predictions from the segmentation model.

    Args:
        class_mask (PIL.Image or np.ndarray): The input class mask.
        classes (list): List of class names to be included in the mask (default is an empty list).
        return_labels (bool): Whether to return the detected class labels (default is False).

    Returns:
        np.ndarray: The initial mask.
        list (optional): The detected class labels if return_labels is True.
    """
    class_labels = []
    class_idxs = []

    if isinstance(class_mask, Image.Image):
        if class_mask.mode != "RGB":
            class_mask = class_mask.convert("RGB")
        class_mask = np.array(class_mask)

    mask = class_mask.copy()

    if not len(classes):
        classes = ADE_CLASSES

    # Extracting detected labels and ids
    for i, class_label in enumerate(classes):
        class_idx = ADE_CLASSES.index(class_label)
        if np.any(mask == class_idx):
            class_labels.append(class_label)
            class_idxs.append(class_idx)
    print(class_labels)

    # Creating a mask of detected objects
    for idx in class_idxs:
        mask[mask == idx] = 255
    mask[mask != 255] = 0

    mask = mask.astype(np.uint8)

    if return_labels:
        return mask, class_labels

    return mask

@torch.inference_mode()
def get_mask(image, padding_factor=0, classes=TARGET_CLASSES):
    """
    Generates a mask for the input image using the OneFormer model and applies padding.

    Args:
        image (PIL.Image or np.ndarray): The input image.
        padding_factor (int): The padding factor for the mask (default is 0).
        classes (list): List of class names to be included in the mask (default is TARGET_CLASSES).

    Returns:
        PIL.Image: The final mask image.
    """
    class_mask = MODEL(image)
    final = get_init_mask(class_mask, classes=classes)
    final = cv2.dilate(final, np.ones((2, 2)), iterations=5)
    # final = cv2.erode(final, np.ones((5, 5)), iterations=5)

    # Optionally, add room transition masks
    # transition_mask = get_room_transition_masks(image)
    # transition_mask = inverse_mask(transition_mask[:, :, 0])
    # final = merge_mask(final, transition_mask)

    if padding_factor > 0:
        top_factor = int(final.shape[0] * 0.4)
        padding_factor = 10

        final[:top_factor, :] = 0
        final[-padding_factor:, :] = 0
        final[:, 0:padding_factor] = 0
        final[:, -padding_factor:] = 0

    return Image.fromarray(final)
