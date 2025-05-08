import streamlit as st

# This MUST be the first Streamlit command
st.set_page_config(page_title="AI vs Human Detector", layout="centered")

# Now you can import other libraries
from PIL import Image
import torch
from torchvision import transforms
from timm import create_model

# Constants
IMG_SIZE = 380
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAPPING = {1: "Human-Generated", 0: "AI-Generated"}
MODEL_PATH = "pytorch_model.pth"

# Load Model
@st.cache_resource
def load_model():
    model = create_model('efficientnet_b4', pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# Image Transformation
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE + 20),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Prediction Function
def predict_image(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_class].item()
    return LABEL_MAPPING[predicted_class], confidence

# UI
st.title("🧠 AI vs Human Image Classifier")
st.markdown("Upload an image to detect whether it's **AI-generated** or **Human-made**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "avif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=200)

    with st.spinner("Predicting..."):
        label, confidence = predict_image(image)
        st.success(f"**Prediction**: {label}")
        st.info(f"**Confidence**: {confidence*100:.2f}%")
