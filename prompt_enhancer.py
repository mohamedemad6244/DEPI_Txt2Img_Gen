# prompt_enhancer.py
from dotenv import load_dotenv
import os
import json
from google import genai
from google.genai import types

load_dotenv()

SYSTEM_INSTRUCTION = (
    "You are an expert prompt engineer for AI image generation using SDXL, FLUX, and diffusion models. "
    "Your job is to transform the user's idea into a dense, keyword-rich cinematic prompt similar to high-end concept art tags. "
    "Output must be a single line of comma-separated descriptors with no sentences. "
    "Follow this structure: subject, environment, artistic style, rendering engines, realism level, camera details, lighting style, mood, quality tags, resolution tags. "
    "Use styles like: cinematic, octane render, 8k, hyper-detailed, trending on ArtStation, photorealistic, volumetric lighting, concept art, Unreal Engine. "
    "Avoid storytelling. Do NOT explain. Only output the final enhanced prompt."
)

def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("ERROR: GEMINI_API_KEY not found in .env")
    return genai.Client(api_key=api_key)

def enhance_prompt(client, rough_prompt: str, model_id: str = "gemini-2.5-flash"):
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.8,
        max_output_tokens=256,
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )
    response = client.models.generate_content(
        model=model_id,
        contents=rough_prompt,
        config=config
    )
    return response.text

def save_prompt(prompt_text: str, filename: str = "enhanced_prompt.json"):
    data = {"enhanced_prompt": prompt_text.strip()}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
