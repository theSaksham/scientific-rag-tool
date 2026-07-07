import sqlite3
import pickle
from rank_bm25 import BM25Okapi

def build_bm25_index():
    conn = sqlite3.connect("data/abstracts.db")
    cur = conn.cursor()
    cur.execute("SELECT chunk_id, pmid, topic, chunk_text FROM chunks ORDER BY chunk_id")
    rows = cur.fetchall()
    conn.close()

    print(f"Building BM25 index over {len(rows)} chunks...")

    metadata = [{"chunk_id": r[0], "pmid": r[1], "topic": r[2], "chunk_text": r[3]} for r in rows]
    tokenized_corpus = [r[3].lower().split() for r in rows]

    bm25 = BM25Okapi(tokenized_corpus)

    index = {
        "bm25": bm25,
        "metadata": metadata
    }

    with open("data/bm25_index.pkl", "wb") as f:
        pickle.dump(index, f)

    print(f"BM25 index saved to data/bm25_index.pkl")
    print(f"Corpus size: {len(tokenized_corpus)} documents")

if __name__ == "__main__":
    build_bm25_index()
