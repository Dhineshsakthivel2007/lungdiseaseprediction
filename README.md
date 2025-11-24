Lung Disease Prediction App 🫁
Project Link

Live app

Dataset

“Lung Disease” Kaggle Dataset
 by Fatemeh Mehrparvar

📘 Overview

This project is a web‑application built with Streamlit that predicts lung conditions from chest X‑ray images. The model classifies images into the following categories:

Lung Opacity

Normal

Pneumonia

It uses a pretrained deep learning model (e.g., PyTorch) and provides an interactive interface for users to upload an image and get a prediction in real‑time.

🧰 Features

Upload chest X‑ray image (jpg/png)

Model inference on the uploaded image

Displays prediction result with human‑readable class label

Lightweight UI built for ease of use on the web

Designed for CPU inference (no GPU required)

🛠️ Tech Stack

Frontend / Web: Streamlit

Deep Learning Framework: PyTorch

Dataset: Kaggle “Lung Disease” dataset

Model: Saved full architecture + weights for inference

Deployment: Streamlit Cloud

📂 Dataset Description

The dataset from Kaggle includes labeled chest X‑ray images for various lung conditions. It is used here to train/validate the model behind this app.

Classes:

0 → Lung Opacity

1 → Normal

2 → Pneumonia

📄 Usage

Clone or download this repository.

Place the trained model file (full_model.pth) in the project directory.

Ensure the requirements.txt lists all dependencies, e.g.:

streamlit
torch==2.5.0
torchvision==0.20.0
Pillow
timm
numpy


Create a .streamlit/packages.toml file with:

[tool.streamlit.packages]
python_version = "3.10"


Run the app locally:

streamlit run app.py


Upload a chest X‑ray image and click “Predict” to view the result.

🎯 How It Works

The user uploads an X‑ray image.

The app preprocesses the image (resizing, tensor conversion).

The model makes a prediction: returns a class index.

The index is mapped to a label (as listed above).

The label is displayed in the UI.

🧩 Model & Class Mapping
class_names = {
    0: "Lung Opacity",
    1: "Normal",
    2: "Pneumonia"
}

🚀 Deployment & Hosting

This project is deployed on Streamlit Cloud and accessible via the link above. The deployment uses CPU mode and supports inference without specialized hardware.

⚠️ Notes & Limitations

The model is trained on a specific dataset; performance may vary on new/unseen data.

Always verify predictions with a qualified medical professional.

Web app is meant for demonstration and research purposes only — not a substitute for clinical diagnostics.

Ensure image quality and format are appropriate (chest X‑ray, clear resolution) for best results.

📚 References

Kaggle Dataset: “Lung Disease” by Fatemeh Mehrparvar.

PyTorch documentation.

Streamlit documentation.

🧑‍💻 Contributing

Contributions are welcome! If you have suggestions (e.g., more classes, improved UI, optimization), please open an issue or pull request.