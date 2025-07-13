def gpt_classify_topic(question: str, openai_client) -> str:
    """
    Uses GPT to classify the question into one of the supported domains.
    Returns one of: 'constitution', 'gdpr', 'attention'
    """
    system_prompt = (
        "You are an intelligent routing agent. "
        "Given a question, classify it into one of the following categories:\n\n"
        "- 'constitution': U.S. government structure, amendments, articles, civil rights, laws\n"
        "- 'gdpr': privacy laws, European data regulations, data subjects, controllers, consent\n"
        "- 'attention_is_all_you_need': AI models, machine learning, self-attention, encoder-decoder architectures, transformers in NLP\n\n"
        "Return only the category name.\n\n"
        "Examples:\n"
        "Q: What does the Fourth Amendment guarantee? → constitution\n"
        "Q: What are the rights of a data subject? → gdpr\n"
        "Q: How does self-attention work in transformers? → attention_is_all_you_need\n"
        "Q: What is positional encoding? → attention_is_all_you_need\n"
        "Q: What are transformers in machine learning? → attention_is_all_you_need\n"
        "Q: Who enforces GDPR? → gdpr\n"
        "Q: What does Article I describe? → constitution\n"
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
    if topic not in {"constitution", "gdpr", "attention_is_all_you_need"}:
        return "unidentified"  # fallback
    return topic
