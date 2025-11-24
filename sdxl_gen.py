# sdxl_gen.py
import json
import torch
from diffusers import DiffusionPipeline

NEGATIVE_PROMPT = """
bad quality, blurry, low quality, distorted, bad anatomy, pixelated,
extra limbs, mangled, watermark, text, lowres, artifacts, noise,
extra fingers, mutated body, overexposed, underexposed, washed out,
poor detail, low detail, deformed face, out of frame
"""

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device=="cuda" else torch.float32

def load_prompt(filename="enhanced_prompt.json"):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["enhanced_prompt"]

def load_models():
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant="fp16"
    ).to(device)

    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant="fp16"
    ).to(device)

    return base, refiner

def generate_image(prompt_text, base, refiner, width=512, height=512):
    generator = torch.Generator(device=device).manual_seed(43)
    n_steps = 50
    high_noise_frac = 0.8

    latent_output = base(
        prompt=prompt_text,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
        generator=generator,
        width=width,
        height=height
    ).images

    final_image = refiner(
        prompt=prompt_text,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=latent_output,
        generator=generator
    ).images[0]

    final_image.save("stabilityai_SDXL_generated_image.png")
    print("Image saved as stabilityai_SDXL_generated_image.png")
