from pinecone import Pinecone, ServerlessSpec

from config import API_KEY, ENVIRONMENT, MODEL_NAME

def initialize_pinecone():
    pc = Pinecone(api_key=API_KEY)
    return pc

def get_embeddings(sentences):
    """
    Generate embeddings using Pinecone's hosted model.
    :param sentences: List of strings to embed
    :return: Embeddings
    """
    pc = initialize_pinecone()
    embeddings = pc.inference.embed(
        model=MODEL_NAME,  # Explicitly set the model
        inputs=sentences,
        parameters={"input_type": "passage", "truncate": "END"}
    )
    return embeddings
