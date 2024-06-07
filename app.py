import os
import gc
import json

from typing import Optional

import torch
import numpy as np
from PIL import Image

from pydantic import BaseModel, HttpUrl, Field
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, Response, HTTPException

from model.decluttering import Decluttering
from helper_func import decode_base64_image, image_to_base64

from diffusers.utils import load_image
from utils import set_img_dims

# Initialise model
MODEL = None
MODEL_NAME = "redraft"


# ===============================================================
# App
# ===============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global MODEL
    MODEL = Decluttering()
    
    yield
    # Clean up the ML models and release the resources
    flush()


app = FastAPI(lifespan=lifespan)


class RoomRequest(BaseModel):
    image_url: HttpUrl
    room_type: str


@app.get(f'/health')
def get_helath():
    
    return Response(
        content=json.dumps({'health_status': 'ok'}),
        status_code=200
    )

# ===============================================================
# Helpers
# ===============================================================

def flush():
    gc.collect()
    torch.cuda.empty_cache()

# Define the input data model
class TextRequest(BaseModel):
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    load_lora: Optional[str] = None
    num_images_per_prompt: int = 1
    num_inference_steps: int = 30
    guidance_scale: float = 5.0
    seed: int = -1
    width: Optional[int] = None
    height: Optional[int] = None
    padding_factor: float = 5.0
    blur_factor: float = 5.0

def cast_to(data, cast_type):
    try:
        return cast_type(data)
    except Exception as e:
        raise RuntimeError(f"Could not cast {data} to {str(cast_type)}: {e}")

def _stage(response):
    print(response.keys())

    if response.get('image_url'):
        starting_image = response['image_url']
        image = load_image(starting_image).convert("RGB")
    else:
        starting_image = response['image']
        image = decode_base64_image(starting_image)

    if response.get('mask_url'):
        starting_mask_image = response['mask_url']
        mask_image = load_image(starting_mask_image).convert("L")
    else:
        starting_mask_image = response['mask_image']
        mask_image = decode_base64_image(starting_mask_image).convert("L")

    # Set random seed if not provided
    if response['seed'] is None:
        response['seed'] = int.from_bytes(os.urandom(2), "big")
    
    # Adjust image dimensions
    image = set_img_dims(image)
    mask_image = set_img_dims(mask_image)

    
    # Extract parameters from job input
    prompt = response.get("prompt", None)
    negative_prompt = response.get("negative_prompt", None)
    load_lora = response.get("load_lora", None)
    num_images_per_prompt = response.get("num_images", 1)
    print(response["num_inference_steps"])
    num_inference_steps = response.get("num_inference_steps", 35)
    guidance_scale = response.get("guidance_scale", 5)
    seed = response.get("seed", -1)
    width = response.get('width', None)
    height = response.get('height', None)
    padding_factor = response.get('mask_padding', 0)
    blur_factor = response.get('blur_factor', 0)
    
    # Generate image and mask using the model
    output, mask = MODEL(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask_image,
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
    output.images[0].save('image.jpg')
    for img in output.images:
        if not isinstance(img, np.ndarray):
            output_image = np.array(img)
        image_strings.append(image_to_base64(output_image))

    # Encode mask to base64
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    Image.fromarray(mask).save('mask.jpg')
    mask = image_to_base64(mask)
    
    # Prepare results dictionary
    results = {
        "result": image_strings[0],
        "mask": mask,
        "seed": response['seed']
    }


    return results

@app.post("/process")
async def process_image(room_request: TextRequest,
                        image_url: str = Query(..., title="Image URL", description="URL of the image to process"),
                        mask_url: str = Query(..., title="Load LoRA", description="Load empty room LoRA or not")):
    try:
        # Process the image URL and room type as needed
        response = {
            "image_url": image_url,
            "mask_url": mask_url,
            "load_lora": room_request.load_lora,
            "prompt": room_request.prompt,
            "negative_prompt": room_request.negative_prompt,
            "num_images_per_prompt": room_request.num_images_per_prompt,
            "num_inference_steps": room_request.num_inference_steps,
            "guidance_scale": room_request.guidance_scale,
            "seed": room_request.seed,
            "width": room_request.width,
            "height": room_request.height,
            "padding_factor": room_request.padding_factor,
            "blur_factor": room_request.blur_factor
        }
        
        response = _stage(response)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))