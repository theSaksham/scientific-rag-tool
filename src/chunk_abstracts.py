import sqlite3
from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_chunks_table(conn):
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS chunks (chunk_id INTEGER PRIMARY KEY AUTOINCREMENT, pmid TEXT NOT NULL, topic TEXT NOT NULL, chunk_index INTEGER NOT NULL, chunk_text TEXT NOT NULL, FOREIGN KEY (pmid) REFERENCES abstracts(pmid))""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunk_topic ON chunks(topic)")
    conn.commit()
    cur.execute("DELETE FROM chunks")
    conn.commit()
    print("Cleared existing chunks table")

def chunk_all_abstracts():
    conn = sqlite3.connect("data/abstracts.db")
    create_chunks_table(conn)
    cur = conn.cursor()

    cur.execute("SELECT pmid, topic, abstract_text FROM abstracts")
    rows = cur.fetchall()
    print(f"Chunking {len(rows)} abstracts...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    total_chunks = 0
    single_chunk_count = 0
    for pmid, topic, abstract_text in rows:
        chunks = splitter.split_text(abstract_text)
        if len(chunks) == 1:
            single_chunk_count += 1
        for idx, chunk_text in enumerate(chunks):
            cur.execute("""INSERT INTO chunks (pmid, topic, chunk_index, chunk_text) VALUES (?, ?, ?, ?)""", (pmid, topic, idx, chunk_text))
            total_chunks += 1

    conn.commit()
    conn.close()
    print(f"Created {total_chunks} chunks from {len(rows)} abstracts")
    print(f"{single_chunk_count} abstracts remained as a single chunk ({single_chunk_count/len(rows)*100:.1f}%)")

if __name__ == "__main__":
    chunk_all_abstracts()

