from langchain_ollama import OllamaLLM
# from src.retriever import HybridRetriever
from retriever import HybridRetriever

TOPICS = ["crispr", "cancer_genomics", "gwas", "protein_folding", "immunotherapy", "neurodegenerative"]

CLASSIFY_PROMPT = """You are a biomedical topic classifier.

Given a question, identify which of the following topics it belongs to: crispr, cancer_genomics, gwas, protein_folding, immunotherapy, neurodegenerative

Rules:
- If the question mentions concepts from TWO or more topics, respond: ambiguous
- If the question applies a technique from one topic to a subject from another topic, respond: ambiguous
- Only respond with a single topic name if the question is EXCLUSIVELY about that one topic
- Do not explain. Do not add punctuation. Output one word only.

Examples:
Question: What CRISPR delivery methods reduce off-target effects?
Topic: crispr

Question: How does CRISPR screening identify cancer driver genes?
Topic: ambiguous

Question: What GWAS loci are associated with Alzheimer's disease risk?
Topic: ambiguous

Question: What is the role of tau protein aggregation in Alzheimer's progression?
Topic: neurodegenerative

Question: {query}
Topic:"""

ANSWER_PROMPT = """You are a biomedical research assistant. Answer the question using ONLY the context provided below.
If the context does not contain enough information to answer, say so clearly.
Be concise and precise. Cite specific mechanisms or findings from the context.

Context:
{context}

Question: {query}

Answer:"""

class QueryPipeline:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.llm = OllamaLLM(model="mistral", temperature=0)

    def classify_topic(self, query) -> str | None:
        prompt = CLASSIFY_PROMPT.format(query=query)
        response = self.llm.invoke(prompt).strip().lower()
        if response in TOPICS:
            return response
        return None  # ambiguous or unrecognized → no filter

    def answer(self, query, top_k: int = 5):
        topic_filter = self.classify_topic(query)
        print(f"  [Call 1] Classified topic: {topic_filter or 'ambiguous (no filter)'}")

        chunks = self.retriever.retrieve(query, top_k=top_k, topic_filter=topic_filter)
        context = "\n\n".join([f"[{i+1}] {c['chunk_text']}" for i, c in enumerate(chunks)])
        sources = [{"pmid": c["pmid"], "topic": c["topic"]} for c in chunks]

        prompt = ANSWER_PROMPT.format(context=context, query=query)
        response = self.llm.invoke(prompt).strip()
        print(f"  [Call 2] Answer generated ({len(response)} chars)")

        return {"query": query, "topic_filter": topic_filter, "answer": response, "sources": sources, "context_chunks": len(chunks)}


if __name__ == "__main__":
    pipeline = QueryPipeline()

    test_queries = [
        "What CRISPR delivery methods reduce off-target effects?", 
        "How does CRISPR screening identify cancer driver genes?", 
        "What is the role of tau protein aggregation in Alzheimer's progression?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        result = pipeline.answer(query)
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources: {result['sources']}")
