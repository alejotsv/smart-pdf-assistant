def gpt_classify_topic(question: str, openai_client) -> str:
    """
    Uses GPT to classify the question into one of the supported domains.
    Returns one of: 'constitution', 'gdpr', 'attention'
    """
    system_prompt = (
        "You are an intelligent routing agent. "
        "Given a question, classify it into one of the following categories based on its content:\n"
        "- 'constitution': for legal/political questions about U.S. constitutional law or government structure\n"
        "- 'gdpr': for questions about privacy, data rights, or European data regulations\n"
        "- 'attention': for technical or AI-related questions involving models, transformers, or deep learning\n\n"
        "Respond with only the category name."
    )

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        max_tokens=5,
        temperature=0
    )

    topic = response.choices[0].message.content.strip().lower()
    if topic not in {"constitution", "gdpr", "attention"}:
        return "attention"  # fallback
    return topic
