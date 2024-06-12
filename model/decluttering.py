import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from diffusers import DPMSolverMultistepScheduler
import pdb
from utils import get_padded_mask, get_mask
from control_nets import get_mlsd_image


def set_img_dims(img, min_dim=1024):
    """
    Resizes the image so that its smallest dimension is equal to min_dim while 
    maintaining the aspect ratio.
    
    Args:
        img (PIL.Image): The input image.
        min_dim (int): The minimum dimension size (default is 1024).

    Returns:
        PIL.Image: The resized image.
    """
    w, h = img.size
    # Calculate scaling factor to make the smallest dimension at least min_dim
    if min(w, h) < min_dim:
        scaler = min_dim / min(w, h)
        img = img.resize((int(w * scaler), int(h * scaler)), Image.Resampling.LANCZOS)
    return img


def resize_image(img):
    """
    Resizes the image such that both its dimensions are multiples of 8, which is 
    often required for certain neural network architectures.
    
    Args:
        img (PIL.Image): The input image.

    Returns:
        PIL.Image: The resized image.
    """
    original_width, original_height = img.size

    # Calculate the new dimensions
    new_width = original_width - (original_width % 8)
    new_height = original_height - (original_height % 8)

    # Resize the image while maintaining the aspect ratio
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img



class Decluttering:
    def __init__(self, device="cuda"):
        """
        Initializes the Staging class with a Stable Diffusion inpainting pipeline, 
        scheduler, and random generator.

        Args:
            seed (bool): Whether to set a random seed (default is False).
            roomtype (str): The type of room (default is "bedroom").
            device (str): The device to run the pipeline on (default is "cuda").
        """
        controlnet = ControlNetModel.from_pretrained("checkpoints/controlnet", torch_dtype=torch.float16)
        
        self.pipeline = StableDiffusionXLControlNetInpaintPipeline.from_single_file(
            "checkpoints/juggerxlInpaint_juggerInpaintV8.safetensors",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(device)


        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config, use_karras_sigmas=True
        )
        self.pipeline.safety_checker = None
        
    def load_lora(self, lora_name):
        self.pipeline.unload_lora_weights()

        # self.pipeline.load_lora_weights('checkpoints/add-detail-xl.safetensors', weights=1.2)
        print("Loaded add-detail")


    def preprocess_img(self, mask, blur_factor=5, padding_factor=5):
        """
        Preprocesses the input image by resizing it and generating a mask.

        Args:
            blur_factor (int): The blur factor for the mask (default is 5).

        Returns:
            tuple: The preprocessed image and the corresponding mask.
        """
        mask = get_padded_mask(mask, padding_factor)
        mask = self.pipeline.mask_processor.blur(mask, blur_factor=blur_factor)
        return mask


    def preprocess_img_and_mask(self, img, blur_factor=5, padding_factor=5):
        """
        Preprocesses the input image by resizing it and generating a mask.

        Args:
            img (PIL.Image): The input image.
            blur_factor (int): The blur factor for the mask (default is 5).

        Returns:
            tuple: The preprocessed image and the corresponding mask.
        """
        img = set_img_dims(img)
        img = resize_image(img)
        mask = np.array(get_mask(img, padding_factor))
        mask = mask.astype("uint8")
        mask = self.pipeline.mask_processor.blur(
            Image.fromarray(mask), blur_factor=blur_factor
        )
        return img, mask


    def __call__(self, prompt, negative_prompt, image, mask_image, preprocessed=False, **kwargs):
        """
        Calls the inpainting pipeline with the given parameters.

        Args:
            prompt (str): The text prompt for the inpainting model.
            negative_prompt (str): The negative prompt for the inpainting model.
            image (PIL.Image): The input image.
            preprocessed (bool): Whether the image has already been preprocessed (default is False).
            mask (PIL.Image): The mask image indicating the region to be inpainted (default is None).

        Returns:
            Output: The output from the inpainting pipeline.
        """
        if not preprocessed:
            print("preprocessing image...")
            padding_factor = kwargs["padding_factor"]
            blur_factor = kwargs["blur_factor"]
            
            if not mask_image:
                image, mask_image = self.preprocess_img_and_mask(image, blur_factor, padding_factor)
            else:
                mask_image = self.preprocess_img(mask_image, blur_factor, padding_factor)
        
        # Removing extra keys
        kwargs.pop('padding_factor', None)
        kwargs.pop('blur_factor', None)
        kwargs.pop('blur_factor', None)
        
        # Load appropriate room_type LoRA
        load_lora = kwargs.pop('load_lora')
        if load_lora:
            self.load_lora(load_lora)

        # Check if a controlnet is selected
        if kwargs.get("controlnet"):
            kwargs.pop('controlnet')
            control_image = get_mlsd_image(image)
        
        if (kwargs.get("width") is None) or (kwargs.get("width") == 0):
            kwargs["width"] = image.size[0]
        if (kwargs.get("height") is None) or (kwargs.get("height") == 0):
            kwargs["height"] = image.size[1]
        # pdb.set_trace()
        out_imgs = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask_image,
            control_image=control_image,
            **kwargs
        )
        return out_imgs, mask_image, control_image
