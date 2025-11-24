import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

# Mapping classes
class_names = {
    0: "Lung Opacity",
    1: "Normal",
    2: "Pneumonia"
}

# Load Model
@st.cache_resource
def load_model():
    with st.spinner("Loading model..."):
        model = torch.load("full_model.pth", map_location="cpu", weights_only=False)
        model.eval()
    return model

model = load_model()

# ---- UI ----
st.title("🫁 Lung Disease Classifier")
st.write("Upload an X-ray image to get predictions.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Put image and output in columns (better UI)
col1, col2 = st.columns(2)

if uploaded_file:
    # Display uploaded image
    with col1:
        image = Image.open(uploaded_file)
        st.image(
            image, 
            caption="Uploaded Image", 
            use_container_width=True
        )

    # Prediction button
    predict_btn = st.button("🔍 Predict")

    if predict_btn:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            pred_idx = torch.argmax(output, dim=1).item()

        prediction = class_names[pred_idx]

        with col2:
            st.success("### 🩺 Diagnosis Result")
            st.write(f"### **{prediction}**")
