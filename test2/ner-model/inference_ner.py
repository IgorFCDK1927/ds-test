import spacy
import sys
import os

# Пути к файлам
MODEL_NAME = "ner_animal_model"

# Проверяем, существует ли обученная модель
if not os.path.exists(MODEL_NAME):
    raise FileNotFoundError(f"❌ Модель '{MODEL_NAME}' не найдена! Запустите сначала 'train_ner.py'.")

# Загружаем обученную NER-модель
nlp = spacy.load(MODEL_NAME)

def predict_ner(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ == "ANIMAL"]
    return entities

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Использование: python inference_ner.py 'your text'")
    else:
        text = sys.argv[1]
        result = predict_ner(text)
        
        if result:
            print(f"✅ Найденные животные: {result}")
        else:
            print("❌ Животные не найдены в тексте.")
