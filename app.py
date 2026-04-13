import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
from torchvision import models

# ===== CSS =====
st.markdown("""
<style>
.skeleton {
    animation: pulse 1.2s infinite;
    background: linear-gradient(90deg, #1f1f1f 25%, #2f2f2f 50%, #1f1f1f 75%);
    background-size: 200% 100%;
    border-radius: 16px;
    height: 320px;
    margin-top: 10px;
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

st.markdown("# 🦇 Batman vs Superman AI")
st.caption("Classificador de imagens com Deep Learning (ResNet18)")

# ===== DARK STYLE =====
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}
.stButton>button {
    border-radius: 10px;
    background-color: #262730;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ===== SESSION STATE =====
if "history" not in st.session_state:
    st.session_state.history = []

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

# ===== HEADER =====
st.title("🦇 Batman vs Superman AI")
st.markdown("Upload ou use uma URL de imagem.")

# ===== INPUTS =====
st.subheader("📥 Entrada de imagem")

uploaded_file = st.file_uploader(
    "Arraste ou selecione uma imagem",
    type=["jpg", "png", "jpeg"]
)

image_url = st.text_input("Ou cole a URL da imagem")

# ===== CLEAR BUTTON =====
if st.button("🧹 Limpar histórico"):
    st.session_state.history = []
    st.rerun()

image = None

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

elif image_url:
    try:
        with st.spinner("🌐 Carregando imagem..."):
            response = requests.get(image_url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
    except:
        st.error("❌ Erro ao carregar imagem da URL")

# ===== PREDICTION =====
if image:
    img = transform(image).unsqueeze(0)

    col1, col2 = st.columns([1, 1])

    # ===== IMAGEM =====
    with col1:
        st.image(image, caption="Imagem carregada", use_column_width=True)

    # ===== RESULTADO =====
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

        # ===== COR DINÂMICA =====
        if conf > 80:
            color = "#00FF9C"
            status = "Alta confiança"
        elif conf > 60:
            color = "#FFD166"
            status = "Média confiança"
        else:
            color = "#FF4B4B"
            status = "Baixa confiança"

        st.markdown(f"""
        <div style="
            background: #1e1e1e;
            padding: 20px;
            border-radius: 12px;
            border-left: 6px solid {color};
            margin-top: 10px;
        ">
        <h3>🧠 Resultado: {result}</h3>
        <p>📊 Confiança: {conf:.2f}%</p>
        <p style="color:{color}; font-weight:bold;">{status}</p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("📈 Probabilidades")
        for i, cls in enumerate(CLASSES):
            st.markdown(f"**{cls} — {probs[i]*100:.2f}%**")
            st.progress(float(probs[i]))

    # ===== SAVE HISTORY =====
    st.session_state.history.append((image, result, conf))

# ===== HISTORY =====
if st.session_state.history:
    st.subheader("🕓 Histórico")

    for i, (img_hist, res, conf) in enumerate(reversed(st.session_state.history[-5:])):
        col1, col2 = st.columns([1, 2])
        col1.image(img_hist, width=100)
        col2.write(f"**{res}** ({conf:.1f}%)")
