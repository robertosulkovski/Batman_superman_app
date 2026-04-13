import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import requests
from io import BytesIO

st.set_page_config(page_title="Batman vs Superman", layout="centered")

classes = ['batman', 'superman']

# carregar modelo
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def prever(img):
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
    return classes[pred.item()]

st.title("🦇 Batman vs Superman Classifier")

opcao = st.radio(
    "Escolha como enviar a imagem:",
    ["Upload do PC", "URL da imagem", "Colar imagem"]
)

img = None

if opcao == "Upload do PC":
    file = st.file_uploader("Envie uma imagem", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file).convert("RGB")

elif opcao == "URL da imagem":
    url = st.text_input("Cole o link da imagem")
    if url:
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        except:
            st.error("Erro ao carregar imagem")

elif opcao == "Colar imagem":
    pasted = st.file_uploader("Cole uma imagem (Ctrl+V funciona aqui)", type=["png", "jpg", "jpeg"])
    if pasted:
        img = Image.open(pasted).convert("RGB")

if img:
    st.image(img, caption="Imagem carregada", use_column_width=True)

    if st.button("🔍 Classificar"):
        resultado = prever(img)
        st.success(f"Predição: {resultado}")