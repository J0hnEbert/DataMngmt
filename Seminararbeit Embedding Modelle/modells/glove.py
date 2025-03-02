import os
import numpy as np
import pandas as pd

# Define paths
DATA_PATH = "/Users/john-thomas/Desktop/Seminararbeit Embedding Modelle/data/processed/cleaned_reviews.csv"
VECTOR_PATH = "/Users/john-thomas/Desktop/Seminararbeit Embedding Modelle/data/vectors/glove/glove.npy"
MODEL_PATH = "/Users/john-thomas/Desktop/Seminararbeit Embedding Modelle/models/glove.6B.300d.txt"

def load_cleaned_data(data_path):
    """Load the cleaned data for embedding generation."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return pd.read_csv(data_path)

def load_glove_model(model_path):
    """Load the pre-trained GloVe model from a text file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"GloVe model not found: {model_path}")
    embeddings_index = {}
    with open(model_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefficients = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefficients
    print(f"Loaded {len(embeddings_index)} word vectors from GloVe model.")
    return embeddings_index

def generate_embeddings(texts, embeddings_index):
    """Generate GloVe embeddings for a list of texts."""
    embedding_dim = len(next(iter(embeddings_index.values())))
    embeddings = []
    for text in texts:
        words = text.split()
        word_vectors = [embeddings_index.get(word, np.zeros(embedding_dim)) for word in words]
        if word_vectors:
            text_embedding = np.mean(word_vectors, axis=0)
        else:
            text_embedding = np.zeros(embedding_dim)
        embeddings.append(text_embedding)
    return np.array(embeddings)

def save_embeddings(embeddings, output_path):
    """Save embeddings to a .npy file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)

def main():
    # Load cleaned data
    print("Loading cleaned data...")
    data = load_cleaned_data(DATA_PATH)
    texts = data["Cleaned Review"].iloc[:50].tolist()  # Use first 50 rows

    # Load GloVe model
    print("Loading GloVe model...")
    embeddings_index = load_glove_model(MODEL_PATH)

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(texts, embeddings_index)

    # Save embeddings
    print("Saving embeddings...")
    save_embeddings(embeddings, VECTOR_PATH)
    print(f"Embeddings saved to {VECTOR_PATH}")

if __name__ == "__main__":
    main()
