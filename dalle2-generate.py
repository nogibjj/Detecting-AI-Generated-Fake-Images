from dotenv import load_dotenv
import os
import requests
import openai
from PIL import Image
import random
import json
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Set up the Google Drive API client
credentials = service_account.Credentials.from_service_account_file('dalle2aigenerator-f4cbc4432631.json', scopes=['https://www.googleapis.com/auth/drive'])
drive_service = build('drive', 'v3', credentials=credentials)

df = pd.read_json("ffhq-dataset-v2.json", orient='columns', lines=True)
# Extract the required information
data = []

for column in df.columns:
    entry = df[column][0]
    photo_title = entry['metadata']['photo_title']
    file_url = entry['image']['file_url']
    file_path = entry['image']['file_path']
    
    data.append({
        'photo_title': photo_title,
        'file_url': file_url,
        'file_path': file_path
    })

# Create a new DataFrame with the extracted data
new_df = pd.DataFrame(data)

def generate_image(input_image_path, output_folder, n, photo_title):
    # Generate the AI images
    response = openai.Image.create_variation(
        image=requests.get(input_image_path).content,
        n=n,
        size="1024x1024"
    )

    # Get the URLs of all generated images
    image_urls = [data['url'] for data in response['data']]
    print(f"Generated image URLs: {image_urls}")

    # Download the images from the URLs and save to files
    for i, image_url in enumerate(image_urls):
        # Download the image from the URL
        image_data = requests.get(image_url).content

        # Save the image to a file with a numbered suffix
        j = 1
        while True:
            output_image_path = os.path.join(output_folder, f"{photo_title}_ai({i+1}_{j}).png")
            if not os.path.exists(output_image_path):
                break
            j += 1

        with open(output_image_path, "wb") as f:
            f.write(image_data)

        print(f"Saved image to {output_image_path}")

        # Upload the image to Google Drive
        # folder_id = ""  # Add the folder ID here
        file_metadata = {
            'name': os.path.basename(output_image_path),
            'mimeType': 'image/png',
            'parents': ['15hNp5EiJbOTtRATX_7ugbdKVb-1guQdd']
        }

        media = MediaFileUpload(output_image_path, mimetype='image/png')
        file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        print(f"Uploaded image to Google Drive with File ID: {file.get('id')}")

    return image_urls



def main(output_folder, num_images):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Randomly select the specified number of images from the DataFrame
    selected_images = new_df.sample(num_images)

    for _, row in selected_images.iterrows():
        input_image_url = row['file_url']
        photo_title = row['photo_title']
        print(f"Processing image: {input_image_url}")
        generated_image_urls = generate_image(input_image_url, output_folder, 1, photo_title)


if __name__ == "__main__":
    output_folder = "aigenerate"
    num_images = 1  # Set the number of images you want to process randomly
    main(output_folder, num_images)
