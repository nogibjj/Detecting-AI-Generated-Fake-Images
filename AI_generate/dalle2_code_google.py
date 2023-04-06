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
import io

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Set up the Google Drive API client
# credentials = service_account.Credentials.from_service_account_file('dalle2aigenerator-f4cbc4432631.json', scopes=['https://www.googleapis.com/auth/drive'])
# drive_service = build('drive', 'v3', credentials=credentials)

new_df = pd.read_csv('face.csv')


def save_original_photo_to_drive(input_image_path, folder_id, photo_title):
    # Download the image from the URL
    image_data = requests.get(input_image_path).content

    # Save the image to a temporary file
    temp_image_path = "temp_original_image.png"
    with open(temp_image_path, "wb") as f:
        f.write(image_data)

    # Upload the image to Google Drive
    file_metadata = {
        'name': photo_title + '.png',
        'mimeType': 'image/png',
        'parents': [folder_id]
    }
    media = MediaFileUpload(temp_image_path, mimetype='image/png')
    file = drive_service.files().create(
        body=file_metadata, media_body=media, fields='id').execute()
    print(
        f"Uploaded original image to Google Drive with File ID: {file.get('id')}")

    # Delete the temporary file
    os.remove(temp_image_path)

    return file.get('id')


def generate_image(input_image_path, n, photo_title):
    # Save the original photo to Google Drive
    # Replace with the ID of the "human" folder
    human_folder_id = "1ZjumcbLeUY7KK6nDf2y2CvzJo6CtEprh"
    original_photo_file_id = save_original_photo_to_drive(
        input_image_path, human_folder_id, photo_title)

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

        # Save the image to a temporary file
        temp_image_path = f"temp_generated_image_{i}.png"
        with open(temp_image_path, "wb") as f:
            f.write(image_data)

        # Upload the image to Google Drive
        # Replace with the ID of the folder where you want to save generated images
        folder_id = "15hNp5EiJbOTtRATX_7ugbdKVb-1guQdd"
        file_metadata = {
            'name': f"{photo_title}_ai({i+1}).png",
            'mimeType': 'image/png',
            'parents': [folder_id]
        }

        media = MediaFileUpload(temp_image_path, mimetype='image/png')
        file = drive_service.files().create(
            body=file_metadata, media_body=media, fields='id').execute()
        print(f"Uploaded image to Google Drive with File ID: {file.get('id')}")

        # Update the DataFrame with the original and generated image URLs
        new_df.loc[new_df['file_url'] == input_image_path,
                   'original_photo_file_id'] = original_photo_file_id
        new_df.loc[new_df['file_url'] == input_image_path,
                   'generated_image_url'] = image_url
        new_df.loc[new_df['file_url'] == input_image_path,
                   'generated_image_file_id'] = file.get('id')

        # Delete the temporary file
        os.remove(temp_image_path)

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
        generate_image(input_image_url, 1, photo_title)

    # Save the updated DataFrame to a new CSV file
    new_df.to_csv("updated_ffhq-dataset-v2.csv", index=False)


if __name__ == "__main__":
    output_folder = "aigenerate"
    num_images = 1  # Set the number of images you want to process randomly
    main(output_folder, num_images)
