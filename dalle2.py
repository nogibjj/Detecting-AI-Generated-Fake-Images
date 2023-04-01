from dotenv import load_dotenv
import os
import requests
from PIL import Image
import openai

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_image(input_image_path, output_folder):
    # Generate the AI image
    response = openai.Image.create_variation(
        image=open(input_image_path, "rb"),
        n=1,
        size="1024x1024"
    )
    image_url = response['data'][0]['url']
    print(f"Generated image URL: {image_url}")

    # Download the image from the URL
    image_data = requests.get(image_url).content

    # Get the base filename (without extension) of the input image
    filename, ext = os.path.splitext(os.path.basename(input_image_path))

    # Save the image to a file with a numbered suffix
    i = 1
    while True:
        output_image_path = os.path.join(output_folder, f"{filename}_ai({i}){ext}")
        if not os.path.exists(output_image_path):
            break
        i += 1

    with open(output_image_path, "wb") as f:
        f.write(image_data)

    print(f"Saved image to {output_image_path}")

    return output_image_path

if __name__ == "__main__":
    input_folder = "/Users/scottlai/Desktop/coding_project/DallE2-ai-generator/input"
    output_folder = "/Users/scottlai/Desktop/coding_project/DallE2-ai-generator/output"

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Generate images for all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            input_image_path = os.path.join(input_folder, filename)
            generate_image(input_image_path, output_folder)
