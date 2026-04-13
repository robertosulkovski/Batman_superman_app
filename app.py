import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
from torchvision import models
import os

st.set_page_config(layout="centered")

#Aviso
st.markdown("""
<div style="
    background: #FFFBEB;
    padding: 12px 16px;
    border-radius: 10px;
    border: 1px solid #FDE68A;
    color: #92400E;
    font-size: 14px;
    margin-bottom: 20px;
">
⚠️ <b>Aviso:</b> O modelo pode ter menor precisão com desenhos, HQs ou imagens fora do padrão.
</div>
""", unsafe_allow_html=True)


# ===== CSS =====
st.markdown("""
<style>

/* ===== BASE ===== */
.stApp {
    background-color: #F8FAFC;
    color: #0F172A;
}

/* TITULOS */
h1, h2, h3 {
    color: #0F172A;
}

/* ===== UPLOADER ===== */
section[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 1px dashed #CBD5F5 !important;
    border-radius: 12px !important;
    padding: 14px !important;
}

section[data-testid="stFileUploader"]:hover {
    border: 1px dashed #2563EB !important;
}

/* ===== INPUT ===== */
input {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    color: #0F172A !important;
    padding: 12px !important;
}

input:focus {
    border: 1px solid #2563EB !important;
}

/* ===== BOTÃO ===== */
.stButton>button {
    background: #2563EB;
    color: white;
    font-weight: 600;
    border-radius: 10px;
    border: none;
}

.stButton>button:hover {
    background: #1D4ED8;
}

/* ===== CARD ===== */
.result-card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 20px;
    border: 1px solid #E2E8F0;
}

/* ===== PROGRESS ===== */
.progress-bar {
    background: #E2E8F0;
    border-radius: 6px;
    height: 10px;
}

.progress-fill {
    background: #2563EB;
}

/* ===== SKELETON ===== */
.skeleton {
    animation: pulse 1.2s infinite;
    background: linear-gradient(90deg, #E2E8F0 25%, #F1F5F9 50%, #E2E8F0 75%);
    background-size: 200% 100%;
    border-radius: 12px;
    height: 280px;
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
