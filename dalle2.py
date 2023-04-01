from dotenv import load_dotenv
import os

import openai

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def image_to_image():
    response = openai.Image.create_variation(
    image=open("/Users/scottlai/Desktop/coding_project/facerecognition/pic_try/00000.png", "rb"),
    n=1,
    size="1024x1024"
    )
    image_url = response['data'][0]['url']
    print(image_url)
    # return image_url

def edit_image():
    response = openai.Image.create_edit(
    image=open("/Users/scottlai/Desktop/coding_project/facerecognition/pic_try/00001.png", "rb"),
    mask=open("/Users/scottlai/Desktop/coding_project/facerecognition/pic_try/00000.png", "rb"),
    prompt="change the face to a different emotion",
    n=3,
    size="1024x1024"
    )
    image_url = response['data'][0]['url']
    print(image_url)
    
    
if __name__ == "__main__":
    # image_to_image()
    edit_image()