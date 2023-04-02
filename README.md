# DallE2-ai-generator
This project generates AI images using the DALL-E2 API from OpenAI. It takes as input a folder of images and generates a corresponding AI image for each input image, which is then saved to an output folder.


dalle2_code.py: generate AI pic and save it into Google drive 
localDalle2.py: Read pic from input folder and generate AI pic and save it into local output folder
# install package 
To install the required packages, run the following command:

> make install 
This will install the required packages specified in the requirements.txt file.

# API Key
To use the DALL-E2 API, you will need to obtain an API key from OpenAI. Once you have an API key, create a .env file in the root directory of the project and add the following line:

> touch .env
> OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"


# Usage
## for the localDalle2.py
To use the script, first set the input and output folder paths in the dalle2.py file:

```
if __name__ == "__main__":
    input_folder = "/path/to/input/folder"
    output_folder = "/path/to/output/folder"
```
Replace /path/to/input/folder with the path to the folder containing your input images, and /path/to/output/folder with the path to the folder where you want to save the generated images.

Once you have set the input and output folder paths, run the following command to generate the AI images:

> python dalle2_code.py
The script will generate an AI image for each image file in the input folder, and save the generated images to the output folder with a numbered suffix.


## for the dalle2_code.py
To use the script, first you need to download the "credentials.json" from you google account.
more information in here: [Google Drive API](https://developers.google.com/drive/api/quickstart/python)

Once you have the credentials.json, you need to update the folder_id in dalle2_code with your google drive folder. 
tip: you need to share your folder with the client_email address in your credentials. 

after you set up above, run the code with:
> python dalle2_code.py

# Contributing
Pull requests are welcome! If you would like to contribute to this project, please fork the repository and create a new branch for your changes. Submit a pull request when you are ready to merge your changes back into the main branch.

# License
This project is licensed under the MIT License - see the LICENSE file for details.