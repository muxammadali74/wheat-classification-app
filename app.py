import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io

def visualize_heatmap_matplotlib(heatmap, cmap='jet'):
    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)

    cax = ax.imshow(heatmap, cmap=cmap)
    ax.set_title("Grad-CAM Heatmap")
    ax.axis("off")

    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Activity", rotation=270, labelpad=15)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf)







def generate_heatmap(model, image_tensor, target_class, last_conv_layer='layer4'):
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    def forward_hook(module, input, output):
        activations.append(output.detach())

    for name, module in model.named_modules():
        if name == last_conv_layer:
            module.register_forward_hook(forward_hook)
            module.register_backward_hook(backward_hook)
            break

    output = model(image_tensor)
    class_score = output[0, target_class]
    model.zero_grad()
    class_score.backward()

    grads = gradients[0]
    acts = activations[0]

    weights = grads.mean(dim=(2, 3), keepdim=True)
    weighted_acts = weights * acts
    heatmap = weighted_acts.sum(dim=1).squeeze()

    heatmap = heatmap.cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()

    return heatmap




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


classNames = ['Sick', 'Healthy',]

def overlay_heatmap_on_image(heatmap, image_pil):
    image = np.array(image_pil.convert("RGB"))
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(heatmap_color, 0.4, image[:, :, ::-1], 0.6, 0)
    return Image.fromarray(cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB))


def wheat_prediction(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = transform(image).unsqueeze(0)

    output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    _, prediction = output.max(1)
    predicted_class = classNames[prediction.item()]
    confidence = probabilities[0, prediction.item()].item() * 100

    color = "green" if predicted_class == "Healthy" else "red"
    result_html = f"<p style='text-align:center; font-size:20px; color:{color};'><b>{predicted_class} ({confidence:.2f}%)</b></p>"

    heatmap = generate_heatmap(model, image_tensor, prediction.item())
    heatmap_image = visualize_heatmap_matplotlib(heatmap)

    return result_html, heatmap_image



app = gr.Interface(
    fn=wheat_prediction,
    inputs=gr.Image(type="pil"),
    outputs=[gr.HTML(), gr.Image(type="pil")],
    title="🌾 Identifying wheat diseases",
    description="🔍 Upload your image and see a visualization of the areas that influenced the model's decision, along with the model's output (Grad-CAM).",
    theme="default",
)



app.launch()