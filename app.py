import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
from torchvision import models
import os

st.set_page_config(layout="centered")

# ===== CSS =====
st.markdown("""
<style>
.stApp {
    background-color: #F9FAFB;
    color: #111827;
}

section[data-testid="stFileUploader"] {
    background: white !important;
    border: 1px dashed #D1D5DB !important;
    border-radius: 12px !important;
    padding: 14px !important;
}

section[data-testid="stFileUploader"]:hover {
    border: 1px dashed #3B82F6 !important;
}

input {
    background: white !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 10px !important;
    padding: 12px !important;
}

.stButton>button {
    background: #3B82F6;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 18px;
}

.result-card {
    background: white;
    border-radius: 14px;
    padding: 20px;
    border: 1px solid #E5E7EB;
}

.progress-bar {
    background: #E5E7EB;
    border-radius: 6px;
    height: 10px;
    margin-bottom: 12px;
}

.progress-fill {
    height: 100%;
    background: #3B82F6;
}

.skeleton {
    animation: pulse 1.2s infinite;
    background: linear-gradient(90deg, #F3F4F6 25%, #E5E7EB 50%, #F3F4F6 75%);
    background-size: 200% 100%;
    border-radius: 12px;
    height: 280px;
}

@keyframes pulse {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
</style>
""", unsafe_allow_html=True)

# ===== CONFIG =====
MODEL_URL = "https://huggingface.co/robertosulkovski/Batman_Superman_model/resolve/main/model.pth"
CLASSES = ["Batman", "Superman"]

st.title("🦇 Batman vs Superman AI")
st.caption("Classificador de imagens com Deep Learning (ResNet18)")

# ===== SESSION =====
if "history" not in st.session_state:
    st.session_state.history = []

if "image" not in st.session_state:
    st.session_state.image = None

# ===== MODEL =====
@st.cache_resource
def load_model():
    if not os.path.exists("model.pth"):
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

# ===== INPUT =====
st.subheader("📥 Entrada de imagem")

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

uploaded_file = st.file_uploader(
    "Upload imagem",
    type=["jpg", "png", "jpeg"],
    key=st.session_state.uploader_key
)

# ===== URL FORM (ENTER FUNCIONA) =====
with st.form("url_form"):
    image_url = st.text_input("Ou cole a URL da imagem")
    submit_url = st.form_submit_button("🔎 Carregar imagem")

# ===== CLEAR =====
if st.button("🧹 Limpar histórico"):
    st.session_state.history = []
    st.session_state.image = None
    st.rerun()

# ===== PRIORIDADE: UPLOAD > URL =====
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.session_state.image = image

elif submit_url and image_url:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(image_url, headers=headers, timeout=10)

        if response.status_code == 200 and "image" in response.headers.get("Content-Type", ""):
            image = Image.open(BytesIO(response.content)).convert("RGB")
            st.session_state.image = image
        else:
            st.error("❌ URL inválida")

    except Exception as e:
        st.error("❌ Erro ao carregar imagem")

# ===== USAR IMAGEM SALVA =====
image = st.session_state.image

# ===== PREDICTION =====
if image:
    img = transform(image).unsqueeze(0)

    col1, col2 = st.columns([1,1])

    with col1:
        st.image(image, use_column_width=True)

    with col2:
        placeholder = st.empty()
        placeholder.markdown('<div class="skeleton"></div>', unsafe_allow_html=True)

        with torch.no_grad():
            outputs = model(img)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probs, 0)

        placeholder.empty()

        result = CLASSES[predicted.item()]
        conf = confidence.item() * 100

        if conf > 80:
            color = "#22C55E"
            status = "Alta confiança"
        elif conf > 60:
            color = "#F59E0B"
            status = "Média confiança"
        else:
            color = "#EF4444"
            status = "Baixa confiança"

        st.markdown(f"""
        <div class="result-card" style="border-left: 5px solid {color}">
            <h3>🧠 {result}</h3>
            <p>📊 {conf:.2f}%</p>
            <p style="color:{color}">{status}</p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("📈 Probabilidades")

        for i, cls in enumerate(CLASSES):
            percent = probs[i] * 100

            st.markdown(f"""
            <b>{cls} — {percent:.2f}%</b>
            <div class="progress-bar">
                <div class="progress-fill" style="width:{percent}%"></div>
            </div>
            """, unsafe_allow_html=True)

    # evitar duplicação no histórico
    if not st.session_state.history or st.session_state.history[-1][0] != image:
        st.session_state.history.append((image, result, conf))

# ===== HISTORY =====
if st.session_state.history:
    st.subheader("🕓 Histórico")

    for img_hist, res, conf in reversed(st.session_state.history[-5:]):
        c1, c2 = st.columns([1,2])
        c1.image(img_hist, width=100)
        c2.write(f"**{res}** ({conf:.1f}%)")
