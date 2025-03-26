import torch
from torchvision import models
from torchvision.models import ResNet50_Weights


model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

torch.save(quantized_model, "model_quantized.pth")