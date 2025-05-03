import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import load_model

model = load_model()

index_to_letter = [chr(i) for i in range(65, 91) if chr(i) not in ['J', 'Z']]

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

st.set_page_config(page_title="ASL Classifier", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>ASL Clsassifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a handwritten American Sign Language letter (A–Y, excluding J and Z)</p>", unsafe_allow_html=True)
st.markdown("---")


uploaded_file = st.file_uploader("Upload an image (will be resized to 28×28)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(img_tensor)
        pred_index = logits.argmax(dim=1).item()

    with col2:
        st.markdown("Prediction Result")
        if pred_index < len(index_to_letter):
            predicted_letter = index_to_letter[pred_index]
            st.success(f" Predicted Letter: **{predicted_letter}**")
        else:
            st.error(f"Prediction index {pred_index} out of range.")



st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 12px;'>Univeristy at Buffalo(Deep Learning- ASL classifier)</p>",
    unsafe_allow_html=True
)
