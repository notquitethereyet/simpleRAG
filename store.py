from pinecone import ServerlessSpec
from config import INDEX_NAME

def create_index(pc):
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1024,  # Dimension for multilingual-e5-large
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Index {INDEX_NAME} created.")
    else:
        print(f"Index {INDEX_NAME} already exists.")

def upsert_data(sentences, embeddings, pc):
    index = pc.Index(INDEX_NAME)
    
    # Extract embedding values and prepare records
    records = [
        {
            "id": f"vec{i}",
            "values": embedding['values'],  # Extract 'values' field from each embedding
            "metadata": {"text": sentence}
        }
        for i, (sentence, embedding) in enumerate(zip(sentences, embeddings))  # No .data
    ]
    
    # Upsert records to Pinecone
    index.upsert(vectors=records)
