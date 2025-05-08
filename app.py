import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load model
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("pneumonia_classifier.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model


model = load_model()

# Define preprocessing
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Prediction function
def predict(image: Image.Image):
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    classes = ["NORMAL", "PNEUMONIA"]
    return classes[predicted.item()]


# Streamlit UI
st.title("Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image to classify it as NORMAL or PNEUMONIA.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        prediction = predict(img)
        st.success(f"Prediction: **{prediction}**")
