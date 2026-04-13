import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
from torchvision import models
import os

st.set_page_config(layout="centered")

# ===== AVISO =====
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

/* FUNDO */
.stApp {
    background-color: #F1F5F9;
    color: #0F172A;
}

/* TITULOS */
h1, h2, h3 {
    color: #0F172A;
}

/* TEXTO */
p {
    color: #475569;
}

/* UPLOADER */
section[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 1px dashed #CBD5E1 !important;
    border-radius: 14px !important;
    padding: 16px !important;
}

/* INPUT */
.stTextInput > div {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 12px !important;
    padding: 6px !important;
}

/* FORM */
div[data-testid="stForm"] {
    background: #FFFFFF;
    padding: 16px;
    border-radius: 14px;
    border: 1px solid #E2E8F0;
}

st.markdown("""
<style>

/* TODOS OS BOTÕES */
button[kind="primary"],
button[kind="secondary"],
.stButton > button,
.stForm button {
    background-color: #E2E8F0 !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 10px !important;
    font-weight: 600;
    padding: 10px 18px;
}

/* HOVER */
button:hover {
    background-color: #CBD5E1 !important;
    color: #0F172A !important;
}

</style>
""", unsafe_allow_html=True)

.stButton>button:hover {
    background: #1D4ED8;
}

/* CARD RESULTADO */
.result-card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 20px;
    border: 1px solid #E2E8F0;
}

/* BARRA */
.progress-bar {
    background: #E2E8F0;
    border-radius: 6px;
    height: 10px;
}

.progress-fill {
    background: #2563EB;
    height: 100%;
    border-radius: 6px;
}

/* SKELETON */
.skeleton {
    animation: pulse 1.2s infinite;
    background: linear-gradient(90deg, #E2E8F0 25%, #F1F5F9 50%, #E2E8F0 75%);
    background-size: 200% 100%;
    border-radius: 12px;
    height: 250px;
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

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

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

uploaded_file = st.file_uploader(
    "Upload imagem",
    type=["jpg", "png", "jpeg"],
    key=st.session_state.uploader_key
)

# ===== FORM URL =====
with st.form("url_form"):
    image_url = st.text_input("Ou cole a URL da imagem")
    submit_url = st.form_submit_button("🔎 Carregar imagem")

# ===== CLEAR =====
if st.button("🧹 Limpar histórico"):
    st.session_state.history = []
    st.session_state.image = None
    st.session_state.uploader_key += 1
    st.rerun()

# ===== LOAD IMAGE =====
if uploaded_file:
    st.session_state.image = Image.open(uploaded_file).convert("RGB")

elif submit_url and image_url:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(image_url, headers=headers, timeout=10)

        if response.status_code == 200 and "image" in response.headers.get("Content-Type", ""):
            st.session_state.image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            st.error("❌ URL inválida")

    except:
        st.error("❌ Erro ao carregar imagem")

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

    if not st.session_state.history or st.session_state.history[-1][0] != image:
        st.session_state.history.append((image, result, conf))

# ===== HISTORY =====
if st.session_state.history:
    st.subheader("🕓 Histórico")

    for img_hist, res, conf in reversed(st.session_state.history[-5:]):
        c1, c2 = st.columns([1,2])
        c1.image(img_hist, width=100)
        c2.write(f"**{res}** ({conf:.1f}%)")
