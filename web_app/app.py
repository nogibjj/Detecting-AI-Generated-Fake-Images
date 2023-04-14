import os
import tempfile
import requests
# from dotenv import load_dotenv
from PIL import Image
import streamlit as st
import openai
from io import BytesIO

# Load the OpenAI API key
# load_dotenv()


# Streamlit app
st.title("AI Human Face Generator")

content_loaded = False
if "openai_api_key" not in st.session_state:
    st.markdown("### Input your OpenAI API Key here")
    api_key = st.text_input(label="", placeholder="Enter your API Key here", key="API_input_text")

    if st.button("Load API Key") and api_key.strip() != "":
        st.session_state.openai_api_key = api_key.strip()
        st.success("API Key loaded successfully.")
        content_loaded = True
else:
    content_loaded = True

if content_loaded:

    openai.api_key = st.session_state.openai_api_key

    # Function to generate AI image
    def generate_image(input_image_file):
        # Generate the AI image
        response = openai.Image.create_variation(
            image=input_image_file,
            n=1,
            size="1024x1024"
        )
        image_url = response['data'][0]['url']

        # Download the image from the URL
        image_data = requests.get(image_url).content

        return Image.open(BytesIO(image_data))


    model_choice = st.sidebar.radio(
        "Select a model:",
        ("AI Human Face Generator", "Real vs AI Human Face Detection")
    )

    if model_choice == "AI Human Face Generator":
        uploaded_file = st.file_uploader("Choose an image file (png or jpg)", type=["png", "jpg"])

        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)

            st.header("Input Image")
            col1, col2 = st.columns(2)

            col1.image(input_image, use_column_width=True)

            if st.button("Generate AI Image"):
                uploaded_file.seek(0)  # Reset the file pointer to the beginning
                ai_image = generate_image(uploaded_file)
                st.header("AI Generated Image")
                col2.image(ai_image, use_column_width=True)

    elif model_choice == "Real vs AI Human Face Detection":
        st.write("This model is still in development.")
