def format_documents(documents):
    # group stays on the tweet, the model only gets the id and the text
    lines = [f"[{doc_id(doc, index)}] {doc['text']}" for index, doc in enumerate(documents, start=1)]
    return "\n".join(lines)

def doc_id(doc, index):
    return str(doc.get("id") or index)
