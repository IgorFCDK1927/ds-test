import spacy
import torch
import sys
from torchvision import models
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

NER_MODEL_PATH = "ner-model/ner_animal_model"
nlp = spacy.load(NER_MODEL_PATH)

# Loading animal classification (RESNET18)
device = torch.device("cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # 10 types of animals

# Load file with training model
CLASSIFIER_MODEL_PATH = "image_classifier/animal_classifier.pth"
model.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device))
model.eval()

# Our types of animals
class_names = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elep', 'horse', 'sheep', 'spider', 'squirrel']

# Function of preprocessing image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Function of animal prediction
def predict_animal(image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

def final_pipeline(text, image_path):
    # NER prediction
    doc = nlp(text)
    extracted_animals = [ent.text.lower() for ent in doc.ents if ent.label_ == "ANIMAL"]

    # Image classifier
    predicted_animal = predict_animal(image_path)

    print(f"Animal in the text: {extracted_animals}")
    print(f"Prediction image: {predicted_animal}")

    match = any(animal in predicted_animal for animal in extracted_animals)
    print(f"Matching: {match}")
    
    return match

if __name__ == "__main__":
    text_input = sys.argv[1]
    image_input = sys.argv[2]
    result = final_pipeline(text_input, image_input)
    print(f"Final result: {result}")




