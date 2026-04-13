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

/* ===== FUNDO ===== */
.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: white;
}

/* ===== SKELETON ===== */
.skeleton {
    animation: pulse 1.2s infinite;
    background: linear-gradient(90deg, #1f1f1f 25%, #2f2f2f 50%, #1f1f1f 75%);
    background-size: 200% 100%;
    border-radius: 16px;
    height: 300px;
}

@keyframes pulse {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* ===== FILE UPLOADER ===== */
section[data-testid="stFileUploader"] {
    background: #0f172a !important;
    border: 1px solid #1f2937 !important;
    border-radius: 14px !important;
    padding: 14px !important;
    transition: 0.3s;
}

section[data-testid="stFileUploader"] * {
    background: transparent !important;
    color: #cbd5e1 !important;
}

section[data-testid="stFileUploader"]:hover {
    border: 1px solid #00FF9C !important;
    box-shadow: 0 0 20px rgba(0,255,156,0.15);
}

/* ===== INPUT ===== */
input {
    background-color: #0f172a !important;
    border: 1px solid #1f2937 !important;
    border-radius: 10px !important;
    color: white !important;
}

/* ===== BOTÃO ===== */
.stButton>button {
    background: linear-gradient(135deg, #00FF9C, #00C2FF);
    color: black;
    font-weight: bold;
    border-radius: 12px;
    border: none;
    padding: 10px 18px;
}

.stButton>button:hover {
    transform: scale(1.05);
}

/* ===== RESULT CARD ===== */
.result-card {
    background: #0f172a;
    border-radius: 16px;
    padding: 20px;
    border: 1px solid #1f2937;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}

/* ===== BARRA CUSTOM ===== */
.progress-bar {
    background: #2a2a2a;
    border-radius: 8px;
    overflow: hidden;
    height: 12px;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #00FF9C, #00CFFF);
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

uploaded_file = st.file_uploader("Upload imagem", type=["jpg", "png", "jpeg"])
image_url = st.text_input("Ou cole a URL")

if st.button("🧹 Limpar histórico"):
    st.session_state.history = []
    st.rerun()

image = None

# upload
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

# url
elif image_url:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(image_url, headers=headers, timeout=10)

        if response.status_code == 200 and "image" in response.headers.get("Content-Type", ""):
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            st.error("❌ URL inválida")

    except:
        st.error("❌ Erro ao carregar imagem")

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

        # cor
        if conf > 80:
            color = "#00FF9C"
            status = "Alta confiança"
        elif conf > 60:
            color = "#FFD166"
            status = "Média confiança"
        else:
            color = "#FF4B4B"
            status = "Baixa confiança"

        # CARD
        st.markdown(f"""
        <div class="result-card" style="border-left: 6px solid {color}">
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

    st.session_state.history.append((image, result, conf))

# ===== HISTORY =====
if st.session_state.history:
    st.subheader("🕓 Histórico")

    for img_hist, res, conf in reversed(st.session_state.history[-5:]):
        c1, c2 = st.columns([1,2])
        c1.image(img_hist, width=100)
        c2.write(f"**{res}** ({conf:.1f}%)")
