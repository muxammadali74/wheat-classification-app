import gradio as gr
import torch
import kagglehub
import os 
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image

model_file = 'models/resnet50_model.pth'

model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.BatchNorm1d(num_features),
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Linear(512, 2)
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


app.launch()