import os
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

def load_cleaned_data(data_path):
    """Load the cleaned data for embedding generation."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return pd.read_csv(data_path)

def generate_embeddings(texts, model):
    """Generate Word2Vec embeddings for a list of texts."""
    embeddings = []
    for text in texts:
        words = text.split()  # Split text into words
        word_vectors = [model[word] for word in words if word in model]  # Get vector for each word
        if word_vectors:  # If the review has words with vectors
            review_embedding = np.mean(word_vectors, axis=0)  # Average word vectors to get review embedding
        else:
            review_embedding = np.zeros(model.vector_size)  # Fallback to zero vector if no words match
        embeddings.append(review_embedding)
    return np.array(embeddings)

def save_embeddings(embeddings, output_path):
    """Save embeddings to a .npy file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)

def main():
    # Paths
    data_path = "/Users/john-thomas/Desktop/Seminararbeit Embedding Modelle/data/processed/cleaned_reviews.csv"
    output_path = "/Users/john-thomas/Desktop/Seminararbeit Embedding Modelle/data/vectors/word2vec/word2vec.npy"
    model_path = "/Users/john-thomas/Desktop/Seminararbeit Embedding Modelle/modells/pretrained_models/GoogleNews-vectors-negative300.bin"

    # Load pre-trained Word2Vec model
    print("Loading Word2Vec model...")
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    # Load cleaned data
    print("Loading cleaned data...")
    data = load_cleaned_data(data_path)
    texts = data["Cleaned Review"].iloc[:50].tolist()  # Use first 50 rows

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(texts, model)

    # Save embeddings
    print("Saving embeddings...")
    save_embeddings(embeddings, output_path)
    print(f"Embeddings saved to {output_path}")

if __name__ == "__main__":
    main()
