import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def load_cleaned_data(data_path):
    """Load the cleaned data for embedding generation."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return pd.read_csv(data_path)

def generate_embeddings(texts, model):
    """Generate SentenceTransformer embeddings for a list of texts."""
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings)

def save_embeddings(embeddings, output_path):
    """Save embeddings to a .npy file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)

def main():
    # Paths
    DATA_PATH = "/Users/john-thomas/Desktop/Seminararbeit Embedding Modelle/data/processed/cleaned_reviews.csv"
    VECTOR_PATH = "/Users/john-thomas/Desktop/Seminararbeit Embedding Modelle/data/vectors/sentence_transformers/sentence_transformers.npy"
    
    # Load Sentence Transformer model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    # Load cleaned data
    print("Loading cleaned data...")
    data = load_cleaned_data(DATA_PATH)
    texts = data["Cleaned Review"].iloc[:50].tolist()  # Use first 50 rows

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(texts, model)

    # Save embeddings
    print("Saving embeddings...")
    save_embeddings(embeddings, VECTOR_PATH)
    print(f"Embeddings saved to {VECTOR_PATH}")

if __name__ == "__main__":
    main()
