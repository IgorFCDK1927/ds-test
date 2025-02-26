ML Pipeline: Named Entity Recognition (NER) + Image Classification

Functional of this project:

1. Recognition of animal words in sentences.
2. Recognition of the type of animal by image.
3. Comparing string with a sentence and input image, if the sentence contains a type of animal that will match with the input image - the result needs to be True.

Project structure:
Folder Ner_model: contains train and test script with Vocabulary of the animal dataset.
Folder image_classifier: contains train, validation, and test script for predicting the type of animal by image and animal_classifier.pth our training dataset.
Folder an_dataset\processed_224x224: The train folder contains 10 folders with types of animals for training the model and the same way with the test folder (predicting test images by training dataset).

How to use:

In root folder by cmd:

pip install -r requirements.txt

Demo: final_pipeline_demo.ipynb
