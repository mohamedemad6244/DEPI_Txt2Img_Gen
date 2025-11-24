# kandinsky_gen.py
import torch
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
from diffusers.models import UNet2DConditionModel
from transformers import CLIPVisionModelWithProjection
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if DEVICE.type=="cuda" else torch.float32

NEGATIVE_PROMPT = """
lowres, text, error, cropped, worst quality, low quality, jpeg artifacts,
ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands,
poorly drawn hands/face, blurry, bad anatomy, missing limbs, watermark, signature
"""

def load_prompt(filename="enhanced_prompt.json"):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["enhanced_prompt"].strip()

def enhance_prompt(prompt: str):
    return f"""
{prompt.strip()}
High-quality, photorealistic, cinematic lighting, sharp focus,
realistic shadows, rich textures, professional color grading.
"""

def load_models():
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "kandinsky-community/kandinsky-2-2-prior", subfolder="image_encoder"
    ).half().to(DEVICE)

    unet = UNet2DConditionModel.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder", subfolder="unet"
    ).half().to(DEVICE)

    prior = KandinskyV22PriorPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-prior",
        image_encoder=image_encoder,
        torch_dtype=torch.float16
    ).to(DEVICE)

    decoder = KandinskyV22Pipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder",
        unet=unet,
        torch_dtype=torch.float16
    ).to(DEVICE)

    return prior, decoder

def generate_image(prompt_text, prior, decoder, width=512, height=512):
    full_prompt = enhance_prompt(prompt_text)

    img_emb = prior(prompt=full_prompt, num_inference_steps=25, num_images_per_prompt=1)
    negative_emb = prior(prompt=NEGATIVE_PROMPT, num_inference_steps=25, num_images_per_prompt=1)

    images = decoder(
        image_embeds=img_emb.image_embeds,
        negative_image_embeds=negative_emb.image_embeds,
        num_inference_steps=75,
        width=width,
        height=height
    )

    images.images[0].save("kandinsky-2-2_generated_image.png")
