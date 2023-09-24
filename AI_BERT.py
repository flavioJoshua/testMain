from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Inizializza il tokenizer e il modello
tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-italian-uncased")
model = BertForSequenceClassification.from_pretrained("dbmdz/bert-base-italian-uncased")

# Prepara il testo
text = "Questo prodotto è sufficiente , certo sarebbe  migliorabile!"

# Tokenizza e ottieni l'output del modello
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)

# Ottieni la predizione
logits = outputs.logits
predicted_class_idx = torch.argmax(logits, dim=1).item()

# Stampa la classe prevista (0 per negativo, 1 per positivo)
print(f"Classe prevista: {predicted_class_idx}")