import boto3
import os
import requests
import openai
import pandas as pd


session = boto3.Session(
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    region_name='us-east-1'
)

s3 = session.client('s3')

openai.api_key = os.getenv('OPENAI_API_KEY')


def generate_image(input_image_path, image_name):

    image_content = None
    with open(input_image_path, "rb") as f:
        image_content = f.read()

    # Generate the AI images
    response = openai.Image.create_variation(
        image=image_content,
        n=1,
        size="1024x1024"
    )

    image_url = [data['url'] for data in response['data']][0]
    print(f"Image URL: {image_url}")

    image_data = requests.get(image_url).content
    with open(f"fake_{input_image_path}", "wb") as f:
        f.write(image_data)

    s3.upload_file(f"fake_{input_image_path}",
                   "dalle2images", f"fake/{image_name}")


def main():

    images_df = pd.read_csv('df_final.csv')
    image_names = list(images_df["Name"])

    for image_name in image_names:
        s3.download_file('dalle2images', f'real/{image_name}', image_name)
        generate_image(image_name, image_name)
        if (os.path.exists(image_name)):
            os.remove(image_name)
        if (os.path.exists(f"fake_{image_name}")):
            os.remove(f"fake_{image_name}")
        images_df = images_df.loc[images_df["Name"] != image_name]
        images_df.to_csv("df_final.csv")


if __name__ == "__main__":
    main()
