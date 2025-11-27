🎨 AI Text-to-Image Generator

Hi! 👋
This is my AI based Text-to-Image Generator project created using Stable Diffusion, Python and Streamlit.
The idea behind this project was to convert simple text prompts into beautiful images with different styles.

This project was built as part of my AI Internship Task and I focused on keeping it simple, clean and easy to use.

What does this app do?

You just have to:

Write a text prompt

Select a style (photorealistic, anime, cartoon, etc.)

Click generate

And it will create AI images for you ✨

It also:

Supports generating multiple images

Allows negative prompts to improve quality

Adds a small watermark for ethical usage

Saves images with metadata

Features

Text to Image generation using Stable Diffusion

Style selection
(Photorealistic, Artistic, Cartoon, Anime, 3D Render)

Multiple images per prompt

Negative prompt support

Works on CPU and GPU

Simple and clean Streamlit UI

Image watermark + metadata saving

How to Run the Project
1. Clone the repository
git clone https://github.com/vibhaa24/text-to-image-generator.git
cd text-to-image-generator

2. Create a virtual environment

For Windows:

python -m venv venv
venv\Scripts\activate

3. Install required libraries
pip install torch diffusers transformers accelerate streamlit pillow

4. Run the app
streamlit run app.py


After that, open the link shown in terminal.
The app will open in your browser 🚀

How it Works (Simple Explanation)

The app uses a Stable Diffusion model to generate images.

Behind the scenes:

Your prompt + selected style are merged using prompt engineering

Negative prompts remove unwanted things like blur or distortion

The AI model processes it and generates images

The output images are saved with watermark and metadata

Folder Structure
Text-to-Image-Generator/
│
├── app.py
├── utils/
│   ├── generate.py
│   ├── prompts.py
│   └── __init__.py
├── outputs/
├── README.md
├── .gitignore

Few Notes

If you run it on CPU, it may take some time (5–10 mins per image 😅)

On GPU, it works much faster.

For best quality, write clear prompts and use good styles.

Future Ideas

If I improve this project more in future, I would:

Add image size options

Add more artistic styles

Deploy it on cloud for public use

Improve safety filtering

Project By

Vibha Pandey
💻 GitHub: https://github.com/vibhaa24/text-to-image-generator