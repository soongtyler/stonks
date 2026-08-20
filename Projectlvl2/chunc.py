def chunk_text(text, chunk_size = 500):

    chunks = []

    for start in range(0, len(text), chunk_size):
        chunk = text[start:start+chunk_size]

        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks