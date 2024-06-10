import numpy as np
import faiss
from transformers import BertModel, BertTokenizer
import torch

# Function to convert text to embeddings using BERT
def text_to_embeddings(texts, model, tokenizer, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            mean_pooling = torch.mean(last_hidden_state, dim=1)
            embeddings.append(mean_pooling.cpu().numpy())
    return np.vstack(embeddings)

# Initialize the BERT model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name).to(device)

# Sample text data
texts = ["Hello, how are you?", "I am fine, thank you!", "What about you?", "I am doing well.", "Good to hear that."]

# Convert text to embeddings
embeddings = text_to_embeddings(texts, model, tokenizer, device).astype('float32')

# Create FAISS index
d = embeddings.shape[1]  # dimension of embeddings
index = faiss.IndexFlatL2(d)  # L2 distance index
index.add(embeddings)  # Add embeddings to the index

# Perform a search
query_texts = ["How are you doing?"]
query_embeddings = text_to_embeddings(query_texts, model, tokenizer, device).astype('float32')
k = 2  # Number of nearest neighbors
D, I = index.search(query_embeddings, k)  # D = distances, I = indices of nearest neighbors

# Output the results
print("Query Texts:", query_texts)
print("Distances to nearest neighbors:\n", D)
print("Indices of nearest neighbors:\n", I)
print("Nearest neighbors:\n", [texts[i] for i in I[0]])
