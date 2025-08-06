import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class SentimentAnalyzer:
    def __init__(self, model_name="cardiffnlp/twitter-xlm-roberta-base-sentiment"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.labels = ["Negative", "Neutral", "Positive"]

    def predict(self, text: str) -> str:
        encoded_input = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            output = self.model(**encoded_input)
        scores = output.logits[0].detach().numpy()
        scores = np.exp(scores) / np.sum(np.exp(scores)) 
        predicted_class = np.argmax(scores)
        return self.labels[predicted_class]