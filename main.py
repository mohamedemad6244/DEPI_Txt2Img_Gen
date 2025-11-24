# gradio_app.py
import gradio as gr
from prompt_enhancer import get_client, enhance_prompt as gemini_enhance, save_prompt
from kandinsky_gen import load_models as load_kandinsky_models, generate_image as generate_kandinsky_image
from sdxl_gen import load_models as load_sdxl_models, generate_image as generate_sdxl_image
import json

def load_prompt(filename="enhanced_prompt.json"):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["enhanced_prompt"]

def generate_image_pipeline(prompt_text, pipeline_choice, image_size):
    width, height = map(int, image_size.split("x"))

    # Gemini Prompt Enhancer
    if pipeline_choice == "Gemini":
        client = get_client()
        enhanced = gemini_enhance(client, prompt_text)
        save_prompt(enhanced)
        return f"Prompt enhanced and saved!\n\n{enhanced}", None

    # Kandinsky
    elif pipeline_choice == "Kandinsky":
        try:
            prior, decoder = load_kandinsky_models()
            generate_kandinsky_image(prompt_text, prior, decoder, width, height)
            return "Image generated using Kandinsky 2.2!", "kandinsky-2-2_generated_image.png"
        except Exception as e:
            return f"Error: {e}", None

    # SDXL
    elif pipeline_choice == "SDXL":
        try:
            base, refiner = load_sdxl_models()
            generate_sdxl_image(prompt_text, base, refiner, width, height)
            return "Image generated using Stable Diffusion XL!", "stabilityai_SDXL_generated_image.png"
        except Exception as e:
            return f"Error: {e}", None

    else:
        return "Invalid pipeline choice.", None

with gr.Blocks() as demo:
    gr.Markdown("# Imagica AI Image Generator 🎨")
    
    prompt_input = gr.Textbox(label="Enter your prompt", placeholder="Write your idea here...", lines=3)
    
    pipeline_choice = gr.Radio(
        choices=["Gemini", "Kandinsky", "SDXL"],
        label="Select Pipeline",
        value="Gemini"
    )
    
    image_size = gr.Radio(
        choices=["512x512", "512x768", "768x512", "768x768"], 
        label="Select Image Size / Ratio", 
        value="512x512"
    )
    
    output_text = gr.Textbox(label="Output", interactive=False)
    output_image = gr.Image(label="Generated Image")

    generate_btn = gr.Button("Generate")
    
    generate_btn.click(
        fn=generate_image_pipeline,
        inputs=[prompt_input, pipeline_choice, image_size],
        outputs=[output_text, output_image]
    )

demo.launch()
