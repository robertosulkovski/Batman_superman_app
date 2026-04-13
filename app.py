import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
from torchvision import models

# ===== CONFIG =====
MODEL_URL = "https://huggingface.co/robertosulkovski/Batman_Superman_model/resolve/main/model.pth"
CLASSES = ["Batman", "Superman"]

st.set_page_config(page_title="Batman vs Superman AI", layout="centered")

# ===== LOAD MODEL =====
@st.cache_resource
def load_model():
    response = requests.get(MODEL_URL, timeout=30)
    with open("model.pth", "wb") as f:
        f.write(response.content)

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ===== TRANSFORM =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ===== UI HEADER =====
st.title("🦇 Batman vs Superman Classifier")
st.markdown("Upload, cole (Ctrl+V) ou use uma URL de imagem.")

# ===== INPUTS =====
col1, col2 = st.columns(2)

uploaded_file = col1.file_uploader("📂 Upload imagem", type=["jpg", "png", "jpeg"])
image_url = col2.text_input("🌐 URL da imagem")

# ===== PASTE IMAGE (Ctrl+V) =====
pasted_image = st.camera_input("📋 Cole imagem (Ctrl+V ou print)")

image = None

# prioridade: upload > url > paste
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

elif image_url:
    try:
        response = requests.get(image_url, timeout=10)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except:
        st.error("Erro ao carregar imagem da URL")

elif pasted_image:
    image = Image.open(pasted_image).convert("RGB")

# ===== PREDICTION =====
if image:
    st.image(image, caption="Imagem carregada", use_column_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

        confidence, predicted = torch.max(probs, 0)

    # ===== RESULT =====
    st.success(f"🧠 Predição: {CLASSES[predicted.item()]}")
    st.info(f"📊 Confiança: {confidence.item()*100:.2f}%")

    # ===== PROB BARS =====
    st.subheader("📈 Probabilidades")
    for i, cls in enumerate(CLASSES):
        st.progress(float(probs[i]))
        st.write(f"{cls}: {probs[i]*100:.2f}%")
