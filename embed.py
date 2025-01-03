from pinecone import Pinecone, ServerlessSpec

from config import API_KEY, ENVIRONMENT, MODEL_NAME

def initialize_pinecone():
    pc = Pinecone(api_key=API_KEY)
    return pc

def chunk_text(sentences, max_length=96):
    """
    Chunk sentences into smaller groups with a maximum length.
    :param sentences: List of sentences.
    :param max_length: Maximum number of sentences per chunk.
    :return: List of chunks.
    """
    for i in range(0, len(sentences), max_length):
        yield sentences[i:i + max_length]

def get_embeddings(sentences):
    """
    Generate embeddings using Pinecone's hosted model.
    :param sentences: List of sentences to embed.
    :return: List of embeddings.
    """
    pc = initialize_pinecone()
    chunks = list(chunk_text(sentences))
    embeddings = []

    for chunk in chunks:
        response = pc.inference.embed(
            model=MODEL_NAME,
            inputs=chunk,
            parameters={"input_type": "passage", "truncate": "END"}
        )
        embeddings.extend(response.data)

    return embeddings

