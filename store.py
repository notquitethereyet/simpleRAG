from pinecone import ServerlessSpec
from config import INDEX_NAME

def create_index(pc):
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1024,  # Dimension for multilingual-e5-large
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

def upsert_data(sentences, embeddings, pc):
    index = pc.Index(INDEX_NAME)
    records = [
        {"id": f"vec{i}", "values": embedding['values'], "metadata": {"text": sentence}}
        for i, (sentence, embedding) in enumerate(zip(sentences, embeddings.data))
    ]
    index.upsert(vectors=records)

