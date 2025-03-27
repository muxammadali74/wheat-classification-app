import gradio as gr
import torch
import kagglehub
import os 
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image

# model_path = kagglehub.model_download("aliochilov/wheat/pyTorch/default")
# model_file = os.path.join(model_path, "model.pth")

model_file = r'D:\Myprojects\Python\For_GitHub\wheat-classification-app\models\redner50_model.pth'

model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.BatchNorm1d(num_features),  # Добавляем Batch Normalization
    nn.Linear(num_features, 512),  # Дополнительный слой (можно убрать)
    nn.ReLU(),                      # Активация
    nn.Linear(512, 2)               # Выходной слой для 2 классов
)

model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))
model.eval()


classNames = ['Kasallangan', 'Sog‘lom',]

def wheat_prediction(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, prediction = output.max(1)

    predicted_class = classNames[prediction.item()]
    confidence = probabilities[0, prediction.item()].item() * 100  

    color = "green" if predicted_class == "Sog‘lom" else "red"
    
    return f"<p style='text-align:center; font-size:20px; color:{color};'><b>{predicted_class} ({confidence:.2f}%)</b></p>"



app = gr.Interface(
    fn=wheat_prediction,
    inputs=gr.Image(type="pil"), 
    outputs=gr.HTML(), 
    title="🌾 Bug‘doy kasalliklarini aniqlash",
    description="🔍 Rasmingizni yuklang va ResNet-50 neyron tarmog‘i bug‘doyning sog‘lom yoki kasallanganligini aniqlaydi.",
    theme="default",
)

# port = int(os.environ.get("PORT", 8080))
# app.launch(server_name="0.0.0.0", server_port=port)
app.launch()