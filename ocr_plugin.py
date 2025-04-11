import io
import os
import tempfile
from google.cloud import vision
from PIL import Image
import streamlit as st
import os


def extract_text_from_image(image):
    client = vision.ImageAnnotatorClient()
    # Save image to a temporary file
    if image:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(image.read())
            temp_path = temp_file.name

        # Read image content
        with io.open(temp_path, "rb") as image_file:
            content = image_file.read()
            image = vision.Image(content=content)


        # Perform OCR with Google Vision
        response = client.text_detection(image=image)
        texts = response.text_annotations

        # Delete the temporary file
        os.remove(temp_path)
    else:
        texts = None

    if texts:
        return texts[0].description  # Extract detected text
    else:
        return None
    
