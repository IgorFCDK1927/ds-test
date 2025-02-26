import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
from torchvision import models
import torch.nn as nn

# Loading the model
device = torch.device("cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # Our 10 animals
model.load_state_dict(torch.load("image_classifier.pth", map_location=device))
model.eval()

# Animals classes
class_names = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elep', 'horse', 'sheep', 'spider', 'squirrel']

# Function of preprocessing of animal image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Animal image prediction
def predict_animal(image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

# CMD
if __name__ == "__main__":
    image_path = sys.argv[1]
    predicted_label = predict_animal(image_path)
    print(f"Model prediction: {predicted_label}")
