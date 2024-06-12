import os
import pdb
import random
from PIL import Image

import gc
import torch


# Import the Staging class from the model.staging module
from model.decluttering import Decluttering
from utils import set_img_dims
from prompts import RANDOM_PHRASES

def flush():
    gc.collect()
    torch.cuda.empty_cache()

# Instantiate the Staging model
model = Decluttering()

# Open the input image
# image = Image.open("test_images/1.jpg")
# room_types = ["bedroom", "living room"]

def staging(idx, image):
    # Optionally, you can include pre-processing checks
    # pre_process_check = is_img_good(image)

    # Define the room type and architectural style for the prompt
    # room_type = random.choice(room_types)
    # architecture_style = "modern"

    # Create the text prompt for the model
    prompt = f"An empty room, without any furniture"
    negative_prompt = "blurry, unrealistic, synthatic, window, door, fireplace, out of order, deformed, disfigured, watermark, text, banner, logo, contactinfo, surreal longbody, lowres, bad anatomy, bad hands, jpeg artifacts, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, rug"

    # Set parameters for the model inference
    num_inference_steps = 20
    num_images_per_prompt = 1
    seed = -1
    guidance_scale = 5

    controlnet = "mlsd"
    mask_image = None
    width = None
    height = None
    load_lora = None
    seed = seed
    padding_factor = 3
    blur_factor = 3
    controlnet_conditioning_scale = 1.0
        
    print(prompt)
    # Generate output images using the model
    output_images, mask, control_condition_image = model(
        prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask_image,
        controlnet=controlnet,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_images_per_prompt=num_images_per_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        load_lora=load_lora,
        width=width,
        height=height,
        seed=seed,
        padding_factor=padding_factor,
        blur_factor=blur_factor
    )
    # pdb.set_trace()
    mask.save(f"output/mask_controlnet_mlsd_decluttering_{idx}_image.jpg")
    control_condition_image.save(f"output/controlnet_mlsd_decluttering_{idx}_image.jpg")
    # Save the generated images to the output directory
    for idxx, output in enumerate(output_images.images):
        output.save(f"output/mlsd_decluttering_{idx}_{idxx}_image.jpg")
    
    flush()

img_list = os.listdir("good")

for idx, item in enumerate(img_list):
    # Open the input image
    image = Image.open("good/" + item)
    image = set_img_dims(image)
    staging(idx, image)
