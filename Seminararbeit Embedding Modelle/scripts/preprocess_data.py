import pandas as pd
import re

# Paths
raw_data_path = "/Users/john-thomas/Desktop/Seminararbeit Embedding Modelle/data/raw/Amazon_Reviews.csv"
cleaned_data_path = "/Users/john-thomas/Desktop/Seminararbeit Embedding Modelle/data/processed/cleaned_reviews.csv"

# Load the dataset
def load_data(file_path):
    """Load the dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path, quotechar='"', doublequote=True, escapechar='\\', encoding='utf-8')
        print("Dataset loaded successfully!")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Function to clean the text
def clean_text(text):
    """Clean text by converting to lowercase and removing special characters."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
    return text

# Clean and preprocess the review dataset
def clean_reviews(data):
    """Clean and preprocess the review dataset."""
    # Combine 'Review Title' and 'Review Text' if both exist
    if 'Review Title' in data.columns and 'Review Text' in data.columns:
        data['Full Review'] = data['Review Title'].fillna('') + ' ' + data['Review Text'].fillna('')
        data = data[['Full Review']]
    elif 'Review Text' in data.columns:
        data = data[['Review Text']]
        data.rename(columns={'Review Text': 'Full Review'}, inplace=True)
    else:
        print("No review text found in the dataset.")
        return None

    # Drop rows with missing reviews
    data = data.dropna(subset=['Full Review'])

    # Clean the text
    data['Cleaned Review'] = data['Full Review'].apply(clean_text)

    # Remove duplicate reviews
    data = data.drop_duplicates(subset=['Cleaned Review'])

    # Remove short reviews
    data = data[data['Cleaned Review'].str.len() > 10]

    print("Data cleaning completed.")
    return data[['Cleaned Review']]

# Save the cleaned dataset
def save_cleaned_data(data, cleaned_file_path):
    """Save the cleaned dataset to a CSV file."""
    try:
        data.to_csv(cleaned_file_path, index=False)
        print(f"Cleaned dataset saved to {cleaned_file_path}")
    except Exception as e:
        print(f"Error saving cleaned dataset: {e}")

# Main script
if __name__ == "__main__":
    # Load the raw dataset
    raw_data = load_data(raw_data_path)

    if raw_data is not None:
        # Clean the dataset
        cleaned_data = clean_reviews(raw_data)

        if cleaned_data is not None:
            # Save the cleaned dataset
            save_cleaned_data(cleaned_data, cleaned_data_path)
