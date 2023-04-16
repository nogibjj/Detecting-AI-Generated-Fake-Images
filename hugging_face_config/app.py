import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from huggingface_hub import from_pretrained_keras

model = from_pretrained_keras('Emmawang/mobilenet_v2_fake_image_detection')

# Define the Streamlit app
def main():
    st.title("Fake Image Detection")
    st.write("This is a demo of a fake image detection app using a MobileNetV2 model trained on the Fake Image Detection dataset.")
    st.write("Upload an image to see if it's fake or not.")
    st.write("")

    uploaded_file = st.file_uploader("Choose an image...", type="png")
    if uploaded_file is not None:

        img = Image.open(uploaded_file).resize([128, 128])
        img = np.array(img).astype(np.float32)
        img = img/255
        img = img.reshape(-1, 128, 128, 3)
        result = get_prediction(img, model)
        if result > 0.5:
            st.write("This image is fake.")
        else:
            st.write("This image is real.")


def get_prediction(image, model):
    prediction = model.predict(image)
    return np.argmax(prediction)

if __name__ == '__main__':
    main()

