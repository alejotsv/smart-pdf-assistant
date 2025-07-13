from rag_engine.classifier import gpt_classify_topic
from rag_engine.retriever import load_index_and_chunks, retrieve_top_k_chunks
from pathlib import Path

def get_answer_from_question(question: str, openai_client, model, top_k=5) -> dict:
    """
    Complete RAG loop: classify question, retrieve chunks, send to LLM, return response.
    Returns dict with topic, answer, and retrieved_chunks.
    """
    topic = gpt_classify_topic(question, openai_client)

    # Handle unidentified topics
    if topic == "unidentified":
        return {
            "topic": topic,
            "visible_topic": "Unknown Topic",
            "answer": (
                "❓ I couldn't confidently match your question to one of the supported documents.\n\n"
                "Please try rephrasing your question or focus on one of the following topics:\n"
                "- U.S. Constitution\n"
                "- GDPR Regulation\n"
                "- Attention Is All You Need (Transformer paper)"
            ),
            "retrieved_chunks": []
        }

    index_path = Path("vectordb") / f"{topic}.faiss"
    chunks_path = Path("vectordb") / f"{topic}_chunks.txt"
    index, chunks = load_index_and_chunks(str(index_path), str(chunks_path))

    retrieved_chunks = retrieve_top_k_chunks(question, index, chunks, model, k=top_k)
    context = "\n---\n".join(retrieved_chunks)

    system_prompt = "You are a helpful assistant. Use the provided context to answer the question as accurately as possible. If the answer is not in the context, say you don't know."

    full_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=full_prompt,
        temperature=0.2,
        max_tokens=512,
    )

    answer = response.choices[0].message.content.strip()

    # Map topic to visible name for display
    visible_topic = ""

    if topic == "constitution":
        visible_topic = "U.S. Constitution"
    elif topic == "gdpr":
        visible_topic = "GDPR Regulation"
    elif topic == "attention_is_all_you_need":
        visible_topic = "Attention Is All You Need (Transformer paper)"

    return {
        "topic": topic,
        "visible_topic": visible_topic,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks
    }






