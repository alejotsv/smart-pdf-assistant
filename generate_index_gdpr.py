import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_pdf_text(pdf_path):
    """Load and concatenate text from all pages of a PDF file."""
    doc = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text() for page in doc]) #type: ignore[attr-defined]
    doc.close()
    return full_text

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks

def embed_chunks(chunks, model):
    """Embed text chunks using a pre-trained model."""
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def save_faiss_index(vectors, output_path, metadata_path=None, texts=None):
    """Save FAISS index and optionally the original chunks."""
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    faiss.write_index(index, output_path)

    if metadata_path and texts:
        with open(metadata_path, "w", encoding="utf-8") as f:
            for chunk in texts:
                f.write(chunk.replace("\n", " ") + "\n---\n")

def process_pdf_to_faiss(pdf_filename, index_name):
    model = SentenceTransformer(MODEL_NAME)

    raw_text = load_pdf_text(pdf_filename)
    chunks = chunk_text(raw_text)
    embeddings = embed_chunks(chunks, model)

    output_dir = Path("vectordb")
    output_dir.mkdir(exist_ok=True)
    faiss_path = output_dir / f"{index_name}.faiss"
    metadata_path = output_dir / f"{index_name}_chunks.txt"

    save_faiss_index(embeddings, str(faiss_path), metadata_path, chunks)
    print(f"Saved FAISS index to {faiss_path} with {len(chunks)} chunks.")

if __name__ == "__main__":
    process_pdf_to_faiss("data/GDPR.pdf", "gdpr")
