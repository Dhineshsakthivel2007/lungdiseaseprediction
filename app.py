import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from gradcam_utils import GradCAM
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# --- Mapping classes ---
class_names = {
    0: "Lung Opacity",
    1: "Normal",
    2: "Pneumonia"
}

# --- Load Model ---
@st.cache_resource
def load_model():
    with st.spinner("Loading model..."):
        model = torch.load("full_model.pth", map_location="cpu", weights_only=False)
        model.eval()
    return model

model = load_model()

# --- Streamlit UI ---
st.set_page_config(page_title="🫁 Lung Disease Classifier", layout="wide")
st.title("🫁 Lung Disease Classifier")
st.markdown(
    """
    Upload a chest X-ray image and the app will predict whether it's Normal, Pneumonia, or Lung Opacity.
    Grad-CAM visualization will show the regions influencing the prediction.
    """
)

# --- Upload Image ---
uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    # --- Transform image ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0)
    
    # --- Prediction ---
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1).squeeze()
        pred_idx = torch.argmax(output, dim=1).item()
        prediction = class_names[pred_idx]
    
    # --- Grad-CAM ---
    gradcam = GradCAM(model, target_layer="features.16.conv")  # adjust if your model is different
    cam = gradcam.generate(img_tensor, class_idx=pred_idx)
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam = np.uint8(255 * cam)
    cam_resized = np.array(Image.fromarray(cam).resize(image.size))
    
    # --- Display side by side ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded X-ray")
        st.image(image, use_container_width=True)
        st.subheader("🩺 Diagnosis Result")
        st.success(prediction)
        st.write(f"{class_names[0]}: {probs[0]*100:.2f}%")
    
    with col2:
        st.subheader("Grad-CAM Heatmap")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image)
        ax.imshow(cam_resized, cmap="jet", alpha=0.4)
        ax.axis("off")
        st.pyplot(fig)
    
    st.info("Red areas in Grad-CAM indicate regions contributing most to the prediction.")
