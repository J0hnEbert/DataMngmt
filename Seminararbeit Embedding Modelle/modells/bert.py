import os
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch

def load_cleaned_data(data_path):
    """Load the cleaned data for embedding generation."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return pd.read_csv(data_path)

def generate_embeddings(texts, model, tokenizer):
    """Generate BERT embeddings for a list of texts."""
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the CLS token representation as the sentence embedding
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_embedding)
    return np.array(embeddings)

def save_embeddings(embeddings, output_path):
    """Save embeddings to a .npy file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)

def main():
    # Paths
    DATA_PATH = "/Users/john-thomas/Desktop/Seminararbeit Embedding Modelle/data/processed/cleaned_reviews.csv"
    VECTOR_PATH = "/Users/john-thomas/Desktop/Seminararbeit Embedding Modelle/data/vectors/bert"

    # Load pre-trained BERT model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # Load cleaned data
    print("Loading cleaned data...")
    data = load_cleaned_data(DATA_PATH)
    texts = data["Cleaned Review"].iloc[:50].tolist()  # Use first 50 rows

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(texts, model, tokenizer)

    # Save embeddings
    print("Saving embeddings...")
    save_embeddings(embeddings, VECTOR_PATH)
    print(f"Embeddings saved to {VECTOR_PATH}")

if __name__ == "__main__":
    main()
