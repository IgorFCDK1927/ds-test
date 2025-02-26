import spacy
from spacy.training import Example
import json
import os

# Defining dir with training data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "train_data.json") 
MODEL_NAME = os.path.join(BASE_DIR, "ner_animal_model") 
with open(TRAIN_DATA_PATH, "r") as f:
    train_data = json.load(f)

# Creating blank English dataset
nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")

# Label ANIMAL if model will predict animal in a text
ner.add_label("ANIMAL")

# Transform to spacy
formatted_data = [
    (entry["text"], {"entities": [(start, end, label) for start, end, label in entry["entities"]]})
    for entry in train_data
]

# Examples for training
examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in formatted_data]

# Starting training
nlp.begin_training()
for epoch in range(20):  # 20 epochs
    losses = {}
    nlp.update(examples, losses=losses)
    print(f"Epoch {epoch+1}, Loss (NER): {losses.get('ner', 0):.4f}")

# Save trained model
nlp.to_disk(MODEL_NAME)
print(f"NER-model saved here - '{MODEL_NAME}'!")
