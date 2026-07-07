import pickle
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "scientific_abstracts"
EMBED_MODEL = "all-MiniLM-L6-v2"
BM25_PATH = "data/bm25_index.pkl"
RRF_K = 60

class HybridRetriever:
    def __init__(self):
        print("Loading embedding model")
        self.model = SentenceTransformer(EMBED_MODEL)

        print("Loading ChromaDB collection")
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_collection(COLLECTION_NAME)

        print("Loading BM25 index")
        with open(BM25_PATH, "rb") as f:
            index = pickle.load(f)
        self.bm25 = index["bm25"]
        self.bm25_metadata = index["metadata"]

        print("Retriever ready.")

    def retrieve(self, query, top_k: int = 5, topic_filter = None):
        query_embedding = self.model.encode(query).tolist()
        chroma_where = {"topic": topic_filter} if topic_filter else None
        chroma_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,
            where=chroma_where,
            include=["documents", "metadatas", "distances"]
        )
        vector_hits = []
        for i, doc_id in enumerate(chroma_results["ids"][0]):
            vector_hits.append({
                "chunk_id": doc_id,
                "chunk_text": chroma_results["documents"][0][i],
                "topic": chroma_results["metadatas"][0][i]["topic"],
                "pmid": chroma_results["metadatas"][0][i]["pmid"],
            })

        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k * 2]
        bm25_hits = []
        for idx in top_bm25_indices:
            m = self.bm25_metadata[idx]
            if topic_filter and m["topic"] != topic_filter:
                continue
            bm25_hits.append({
                "chunk_id": str(m["chunk_id"]),
                "chunk_text": m["chunk_text"],
                "topic": m["topic"],
                "pmid": m["pmid"],
            })

        rrf_scores = {}
        chunk_data = {}

        for rank, hit in enumerate(vector_hits):
            cid = hit["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
            chunk_data[cid] = hit

        for rank, hit in enumerate(bm25_hits):
            cid = hit["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
            chunk_data[cid] = hit

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [chunk_data[cid] for cid, _ in ranked]


if __name__ == "__main__":
    retriever = HybridRetriever()
    results = retriever.retrieve(
        query="What CRISPR delivery methods reduce off-target effects?",
        top_k=5
    )
    print(f"\nTop {len(results)} results:\n")
    for i, r in enumerate(results):
        print(f"[{i+1}] topic={r['topic']} pmid={r['pmid']}")
        print(f"    {r['chunk_text'][:200]}...")
        print()
        