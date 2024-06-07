import os
import random
from PIL import Image

import gc
import torch


# Import the Staging class from the model.staging module
from model.decluttering import Staging
from utils import set_img_dims
from prompts import RANDOM_PHRASES

def flush():
    gc.collect()
    torch.cuda.empty_cache()

# Instantiate the Staging model
model = Staging()

# Open the input image
image = Image.open("test_images/1.jpg")
room_types = ["bedroom", "living room"]
architecture_styles = ["modern", "scandinavian", "boho", "industrial", "contemporary"]

def staging(idx, image, room_type, architecture_style):
    # Optionally, you can include pre-processing checks
    # pre_process_check = is_img_good(image)

    # Define the room type and architectural style for the prompt
    # room_type = random.choice(room_types)
    # architecture_style = "modern"

    # Create the text prompt for the model
    prompt = f"The (({architecture_style} {room_type})), style, ((best quality)),((masterpiece)),((realistic))"
    prompt += random.choice(RANDOM_PHRASES)
    negative_prompt = "blurry, unrealistic, synthatic, window, door, fireplace, out of order, deformed, disfigured, watermark, text, banner, logo, contactinfo, surreal longbody, lowres, bad anatomy, bad hands, jpeg artifacts, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, rug"

    # Set parameters for the model inference
    num_inference_steps = 20
    num_images_per_prompt = 5
    seed = -1
    guidance_scale = 5
    print(prompt)
    # Generate output images using the model
    output_images = model(
        prompt,
        negative_prompt=negative_prompt,
        image=image,
        num_images_per_prompt=num_images_per_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_inference_step=30,
        seed=seed,
    ).images

    # Save the generated images to the output directory
    for idxx, output in enumerate(output_images):
        output.save(f"output/add-detail_{idx}_{idxx}_{room_type}_{architecture_style}_image.jpg")
    
    flush()

img_list = os.listdir("good")

for idx, item in enumerate(img_list):
    # Open the input image
    image = Image.open("good/" + item)
    image = set_img_dims(image)

    for room_type in room_types:
        for architecture_style in architecture_styles:
            print(item)

            staging(idx, image, room_type, architecture_style)

"""
{
  "input": {
    "prompt": "The ((modern bedroom)), style, ((best quality)),((masterpiece)),((realistic))",
    "negative_prompt": "blurry, unrealistic, synthatic, window, door, fireplace, out of order, deformed, disfigured, watermark, text, banner, logo, contactinfo, surreal longbody, lowres, bad anatomy, bad hands, jpeg artifacts, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, rug",
    "image_url": "https://storage.googleapis.com/generative-models-output/empty_room.jpg",
    "num_inference_steps": 25,
    "refiner_inference_steps": 30,
    "guidance_scale": 5,
    "strength": 0.3,
    "seed": -1,
    "num_images": 1
  }
}
'
https://civitai.com/api/download/models/288402
{
  "input": {
    "prompt": "The ((modern bedroom)), style, ((best quality)),((masterpiece)),((realistic))",
    "negative_prompt": "blurry, unrealistic, synthatic, window, door, fireplace, out of order, deformed, disfigured, watermark, text, banner, logo, contactinfo, surreal longbody, lowres, bad anatomy, bad hands, jpeg artifacts, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, rug",
    "image_url": "https://storage.googleapis.com/generative-models-output/empty_room.jpg",
    "num_inference_steps": 25,
    "refiner_inference_steps": 30,
    "guidance_scale": 5,
    "strength": 0.3,
    "seed": -1,
    "num_images": 1
  }
}
"""