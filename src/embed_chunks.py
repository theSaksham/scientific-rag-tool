import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer
import time

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "scientific_abstracts"
EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 256

def embed_all_chunks():
    print(f"Loading embedding model: {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    conn = sqlite3.connect("data/abstracts.db")
    cur = conn.cursor()
    cur.execute("SELECT chunk_id, pmid, topic, chunk_text FROM chunks")
    rows = cur.fetchall()
    conn.close()
    print(f"Embedding {len(rows)} chunks in batches of {BATCH_SIZE}...")

    total_added = 0
    start = time.time()
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i+BATCH_SIZE]
        ids = [str(r[0]) for r in batch]
        texts = [r[3] for r in batch]
        metadatas = [{"pmid": r[1], "topic": r[2]} for r in batch]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        total_added += len(batch)
        elapsed = time.time() - start
        print(f"  Batch {i//BATCH_SIZE + 1}/{(len(rows)-1)//BATCH_SIZE + 1} done — {total_added}/{len(rows)} chunks embedded ({elapsed:.1f}s elapsed)")

    print(f"\nDone. {total_added} chunks embedded into ChromaDB collection '{COLLECTION_NAME}'")
    print(f"Total time: {time.time()-start:.1f}s")

if __name__ == "__main__":
    embed_all_chunks()
