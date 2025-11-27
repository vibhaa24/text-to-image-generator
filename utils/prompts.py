# utils/prompts.py

def build_full_prompt(base_prompt: str, style: str) -> str:
    """
    Combines base prompt and style into a final prompt.
    """
    style_map = {
    "photorealistic": "highly detailed, 8k, ultra realistic",
    "artistic": "digital painting, concept art, cinematic",
    "cartoon": "cartoon style, clean lines, vibrant colors",
    "anime": "anime style, sharp lines, vibrant, detailed",
    "3D render": "octane render, 3d, realistic lighting, highly detailed"
}

    style_text = style_map.get(style, "")
    full_prompt = f"{base_prompt}, {style_text}"

    return full_prompt


def build_negative_prompt(custom_negative: str = "") -> str:
    """
    Returns negative prompt (bad things to avoid).
    """
    default_negative = "blurry, low quality, distorted face, extra fingers, bad anatomy"

    if custom_negative and custom_negative.strip():
        return default_negative + ", " + custom_negative
    return default_negative
