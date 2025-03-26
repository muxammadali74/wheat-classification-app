import gradio as gr
import torch
import kagglehub
import os 
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image

model_path = kagglehub.model_download("aliochilov/wheat/pyTorch/default")
model_file = os.path.join(model_path, "model.pth")

model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

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

port = int(os.getenv("PORT", 7860))
app.launch(server_name="0.0.0.0", server_port=port)