from config import INDEX_NAME



def search(query, pc, top_k=5):
    """
    Perform a search in Pinecone.
    :param query: Search query string.
    :param pc: Pinecone instance.
    :param top_k: Number of top results to return.
    :return: Search results.
    """
    query_embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={"input_type": "query"}
    ).data[0]
    index = pc.Index(INDEX_NAME)

    results = index.query(
        vector=query_embedding['values'],
        top_k=top_k,
        include_metadata=True
    )

    if not results.get('matches'):
        return "I dunno gang, I couldn't find anything."
    return results


