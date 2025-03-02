import os
import numpy as np
from pinecone import Index

# Pinecone-API-Schlüssel und Umgebungs-Hosts
PINECONE_API_KEY = "pcsk_3FA85r_QdysUGnKdHrRG5gdsjNTuuu7bDxfFSQjUPgFs3VgM7kEhyM1qpf2Tve1zhj7b9T"
PINECONE_HOSTS = {
    "high-dim-index": "https://high-dim-index-tnld61r.svc.aped-4627-b74a.pinecone.io",
    "mid-dim-index": "https://mid-dim-index-tnld61r.svc.aped-4627-b74a.pinecone.io",
    "low-dim-index": "https://low-dim-index-tnld61r.svc.aped-4627-b74a.pinecone.io"
}

# Embedding-Dateien gruppiert nach Dimensionalität
EMBEDDING_FILES = {
    "BERT": {"path": "/Users/john-thomas/Desktop/Seminararbeit Embedding Modelle/data/vectors/bert_embeddings.npy", "index": "high-dim-index"},
    "Sentence Transformers": {"path": "/Users/john-thomas/Desktop/Seminararbeit Embedding Modelle/data/vectors/sentence_transformers_embeddings.npy", "index": "mid-dim-index"},
    "GloVe": {"path": "/Users/john-thomas/Desktop/Seminararbeit Embedding Modelle/data/vectors/glove_embeddings.npy", "index": "low-dim-index"},
    "FastText": {"path": "/Users/john-thomas/Desktop/Seminararbeit Embedding Modelle/data/vectors/fasttext_embeddings.npy", "index": "low-dim-index"},
    "Word2Vec": {"path": "/Users/john-thomas/Desktop/Seminararbeit Embedding Modelle/data/vectors/word2vec_embeddings.npy", "index": "low-dim-index"}
}

def connect_to_pinecone(api_key, host):
    """Stellt eine Verbindung zu einem spezifischen Pinecone-Index her."""
    index_name = host.split("//")[1].split("-")[0]
    index = Index(name=index_name, api_key=api_key, host=host)
    print(f"Verbindung zum Pinecone-Index '{index_name}' hergestellt.")
    return index

def upload_embeddings(index, embeddings_file, model_name):
    """Lädt Embeddings zu Pinecone hoch."""
    embeddings = np.load(embeddings_file)
    ids = [f"{model_name}_{i}" for i in range(len(embeddings))]  # IDs generieren
    vectors = [{"id": id_, "values": embedding.tolist()} for id_, embedding in zip(ids, embeddings)]
    index.upsert(vectors=vectors)
    print(f"{len(vectors)} Vektoren von '{model_name}' erfolgreich hochgeladen.")

def main():
    for model_name, config in EMBEDDING_FILES.items():
        embeddings_path = config["path"]
        index_name = config["index"]
        host = PINECONE_HOSTS[index_name]

        if not os.path.exists(embeddings_path):
            print(f"Embedding-Datei für '{model_name}' nicht gefunden: {embeddings_path}")
            continue

        print(f"Hochladen der Embeddings für: {model_name} in den Index: {index_name}")
        index = connect_to_pinecone(PINECONE_API_KEY, host)
        upload_embeddings(index, embeddings_path, model_name)

    print("Alle Embeddings wurden hochgeladen.")

if __name__ == "__main__":
    main()
