def chunk_text(
    text,
    chunk_size=500,
    overlap=100
):
    """
    Breaks a large document into smaller overlapping chunks
    for embedding and retrieval.
    """

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Use chunk_size > 0 and 0 <= overlap < chunk_size.")
    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks
