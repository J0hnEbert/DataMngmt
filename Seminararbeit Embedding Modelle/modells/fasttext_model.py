import fasttext
import pandas as pd
import numpy as np
import os

# Paths
data_path = "/Users/john-thomas/Desktop/Seminararbeit Embedding Modelle/data/processed/cleaned_reviews.csv"
model_path = "/Users/john-thomas/Desktop/Seminararbeit Embedding Modelle/models/cc.en.300.bin"
vector_path = "/Users/john-thomas/Desktop/Seminararbeit Embedding Modelle/data/vectors/fasttext/"

def load_reviews(file_path, n=50):
    """Load the first n cleaned reviews."""
    try:
        reviews = pd.read_csv(file_path)
        if "Cleaned Review" not in reviews.columns:
            raise ValueError("The dataset does not contain the 'Cleaned Review' column.")
        print(f"Loaded {len(reviews)} reviews. Using the first {n}.")
        return reviews["Cleaned Review"].head(n).tolist()
    except Exception as e:
        print(f"Error loading reviews: {e}")
        return []

def generate_embeddings(ft_model, texts):
    """Generate FastText embeddings for a list of texts."""
    embeddings = []
    for text in texts:
        embeddings.append(ft_model.get_sentence_vector(text))
    return np.array(embeddings)

def main():
    # Load the pre-trained FastText model
    print(f"Loading FastText model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"FastText model not found at {model_path}. Please download or provide the model.")

    ft = fasttext.load_model(model_path)
    print("FastText model loaded successfully.")

    # Load reviews
    reviews = load_reviews(data_path, n=50)
    if not reviews:
        print("No reviews to process.")
        return

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(ft, reviews)
    print("Embeddings generated successfully.")

    # Save embeddings
    os.makedirs(os.path.dirname(vector_path), exist_ok=True)
    print(f"Saving embeddings to: {vector_path}")
    np.save(vector_path, embeddings)
    print("Embeddings saved successfully.")

if __name__ == "__main__":
    main()
