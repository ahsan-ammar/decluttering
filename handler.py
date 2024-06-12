'''
Contains the handler function that will be called by the serverless framework.
'''

import os
import io
import base64
import cv2
import numpy as np
from PIL import Image
import concurrent.futures

import torch
from model.decluttering import Decluttering

from diffusers.utils import load_image
from utils import set_img_dims

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

# from rp_schemas import INPUT_SCHEMA

# Clear any cached memory in the GPU
torch.cuda.empty_cache()

# ------------------------------- Model Handler ------------------------------ #

# Initialize the model from the Staging class
MODEL = Decluttering()

# ---------------------------------- Helper Functions ---------------------------------- #

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


def _save_and_upload_images(images, job_id):
    """
    Save images locally and upload them if a bucket endpoint is configured.
    Args:
        images (list): List of PIL images to save and upload.
        job_id (str): Job ID for naming the image files.
    Returns:
        list: List of URLs or base64 strings of the saved images.
    """
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)

        if os.environ.get('BUCKET_ENDPOINT_URL', False):
            image_url = rp_upload.upload_image(job_id, image_path)
            image_urls.append(image_url)
        else:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                image_urls.append(f"data:image/png;base64,{image_data}")

    rp_cleanup.clean([f"/{job_id}"])
    return image_urls


@torch.inference_mode()
def generate_image(job):
    """
    Generate an image from text using the pre-trained model.
    Args:
        job (dict): Job details containing input parameters for image generation.
    Returns:
        dict: Results containing generated image, mask, and seed used.
    """
    job_input = job["input"]

    # Load starting image from URL or base64 string
    if job_input.get('image_url'):
        starting_image = job_input['image_url']
        image = load_image(starting_image).convert("RGB")
    else:
        starting_image = job_input['image']
        image = decode_base64_image(starting_image)


    # Load starting image from URL or base64 string
    if job_input.get('mask_url'):
        starting_mask_image = job_input['mask_url']
        mask_image = load_image(starting_mask_image).convert("L")
    else:
        starting_mask_image = job_input['mask_image']
        mask_image = decode_base64_image(starting_mask_image).convert("L")

    # Set random seed if not provided
    if job_input['seed'] is None:
        job_input['seed'] = int.from_bytes(os.urandom(2), "big")

    # Create a random generator with the specified seed
    generator = torch.Generator("cuda").manual_seed(job_input['seed'])
    
    # Adjust image dimensions
    image = set_img_dims(image)
    mask_image = set_img_dims(mask_image)
    
    # Extract parameters from job input
    prompt = job_input.get("prompt", None)
    negative_prompt = job_input.get("negative_prompt", None)
    load_lora = job_input.get("load_lora", None)
    num_images_per_prompt = job_input.get("num_images", 1)
    num_inference_steps = job_input.get("num_inference_steps", 35)
    guidance_scale = job_input.get("guidance_scale", 5)
    seed = job_input.get("seed", -1)
    width = job_input.get('width', None)
    height = job_input.get('height', None)
    padding_factor = job_input.get('mask_padding', 0)
    blur_factor = job_input.get('blur_factor', 0)
    controlnet = job_input.get("controlnet", "mlsd")
    controlnet_conditioning_scale = job_input.get('controlnet_conditioning_scale', 1.0)
    
    # Generate image and mask using the model
    output, mask, control_condition_image = MODEL(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask_image,
        controlnet=controlnet,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        load_lora=load_lora,
        num_images_per_prompt=num_images_per_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        seed=seed,
        padding_factor=padding_factor,
        blur_factor=blur_factor
    )

    # Encode output images to base64
    image_strings = []
    for img in output.images:
        if not isinstance(img, np.ndarray):
            output_image = np.array(img)
        image_strings.append(image_to_base64(output_image))

    # Encode mask to base64
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    Image.fromarray(mask).save('mask.jpg')
    mask = image_to_base64(mask)
    
    # Encode mask to base64
    if not isinstance(control_condition_image, np.ndarray):
        control_condition_image = np.array(control_condition_image)    
    Image.fromarray(control_condition_image).save('control.jpg')
    control_condition_image = image_to_base64(control_condition_image)

    # Prepare results dictionary
    results = {
        "result": image_strings[0],
        "mask": mask,
        "control_condition_image": control_condition_image,
        "seed": job_input['seed']
    }

    if starting_image:
        results['refresh_worker'] = True

    return results

# Start the serverless function handler
runpod.serverless.start({"handler": generate_image})
