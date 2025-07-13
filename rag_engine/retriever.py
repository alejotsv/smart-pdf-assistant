import faiss

def load_index_and_chunks(index_path, chunks_path):
    index = faiss.read_index(index_path)
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = f.read().split("\n---\n")
    return index, chunks

def embed_query(query, model):
    query_embedding = model.encode([query], convert_to_numpy=True)
    return query_embedding

def retrieve_top_k_chunks(query, index, chunks, model, k=5):
    query_vec = embed_query(query, model)
    distances, indices = index.search(query_vec, k)
    results = [chunks[i] for i in indices[0]]
    return results

