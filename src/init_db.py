import sqlite3

conn = sqlite3.connect("data/abstracts.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS abstracts (id INTEGER PRIMARY KEY AUTOINCREMENT, pmid TEXT UNIQUE NOT NULL, topic TEXT NOT NULL, title TEXT, abstract_text TEXT NOT NULL, journal TEXT, pub_date TEXT, ingested_at TEXT DEFAULT CURRENT_TIMESTAMP)""")

cur.execute("CREATE INDEX IF NOT EXISTS idx_topic ON abstracts(topic)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_pmid ON abstracts(pmid)")

conn.commit()
conn.close()
print("Database initialized at data/abstracts.db")
