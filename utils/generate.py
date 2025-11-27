# utils/generate.py

import os
from datetime import datetime
from typing import List, Dict, Optional

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont

from utils.prompts import build_full_prompt, build_negative_prompt


def apply_watermark(image: Image.Image, text: str = "AI generated @ Vibha") -> Image.Image:
    """
    Adds a simple text watermark at the bottom-right corner.
    Uses Pillow's textbbox (new method, no textsize error).
    """
    if not text:
        return image

    # Make sure image is RGBA for transparency
    if image.mode != "RGBA":
        base = image.convert("RGBA")
    else:
        base = image

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Try a TTF font, else default
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    # ✅ Use textbbox instead of deprecated textsize
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    width, height = base.size
    x = width - text_w - 10
    y = height - text_h - 10

    # Semi-transparent white text
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 160))

    watermarked = Image.alpha_composite(base, overlay)
    return watermarked.convert(image.mode)


def save_with_metadata(image: Image.Image, base_filename: str, output_dir: str, **metadata) -> Dict:
    """
    Saves image as PNG and JPEG and creates a metadata JSON file.
    Returns paths as a dict.
    """
    import json

    os.makedirs(output_dir, exist_ok=True)

    # Save PNG
    png_path = os.path.join(output_dir, f"{base_filename}.png")
    image.save(png_path)

    # Save JPEG
    jpg_path = os.path.join(output_dir, f"{base_filename}.jpg")
    image.convert("RGB").save(jpg_path, "JPEG", quality=95)

    # Save metadata JSON
    metadata_path = os.path.join(output_dir, f"{base_filename}_metadata.json")
    data = {
        "png_path": png_path,
        "jpg_path": jpg_path,
        "created_at": datetime.now().isoformat(),
        **metadata,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return {"image_path": png_path, "jpg_path": jpg_path, "metadata_path": metadata_path}


class TextToImageGenerator:
    """
    Simple wrapper around a Stable Diffusion pipeline.
    Keeps configuration and device handling in one place.
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device_preference: str = "cuda",
        torch_dtype=torch.float16,
    ):
        self.model_id = model_id
        self.device = self._select_device(device_preference)
        self.torch_dtype = torch_dtype if self.device == "cuda" else torch.float32

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            safety_checker=None,  # using custom filter instead
        )
        self.pipeline.to(self.device)

    @staticmethod
    def _select_device(preference: str) -> str:
        if preference == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def generate_images(
        self,
        base_prompt: str,
        style: str,
        n_images: int = 1,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        negative_prompt_text: Optional[str] = None,
        output_dir: str = "outputs",
    ) -> (List[Dict], float):
        """
        Generates images and returns:
        ( [ { 'image_path', 'jpg_path', 'metadata_path' }, ... ], estimated_time_seconds )
        """
        os.makedirs(output_dir, exist_ok=True)

        full_prompt = build_full_prompt(base_prompt, style)
        negative_prompt = build_negative_prompt(negative_prompt_text)

        # Rough ETA estimate (sirf display ke liye, actual se match zaroori nahi)
        approx_seconds_per_image = 1.0 if self.device == "cuda" else 5.0
        estimated_time = approx_seconds_per_image * n_images

        # Autocast / inference context
        if self.device == "cuda":
            context = torch.autocast("cuda", dtype=self.torch_dtype)
        else:
            context = torch.inference_mode()

        # ✅ IMPORTANT: ek hi prompt, num_images_per_prompt se multiple images
        with context:
            result = self.pipeline(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=n_images,
            )
            images = result.images

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results: List[Dict] = []

        for idx, img in enumerate(images):
            # Add watermark
            img = apply_watermark(img, text="AI generated @ Vibha")

            base_filename = f"sd_{timestamp}_{idx}"
            img_info = save_with_metadata(
                image=img,
                base_filename=base_filename,
                output_dir=output_dir,
                prompt=base_prompt,
                full_prompt=full_prompt,
                negative_prompt=negative_prompt,
                style=style,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                device=self.device,
            )
            results.append(img_info)

        return results, estimated_time
