from Bio import Entrez
import sqlite3
import time
import sys
import socket
socket.setdefaulttimeout(30)

Entrez.email = "EMAIL_ID"  # required by NCBI

def fetch_topic(query, topic, max_results=100):
    conn = sqlite3.connect("data/abstracts.db")
    cur = conn.cursor()
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    handle.close()
    pmids = record["IdList"]
    print(f"Found {len(pmids)} PMIDs for topic '{topic}'")
    inserted = 0
    total_batches = (len(pmids) - 1) // 50 + 1
    for i in range(0, len(pmids), 50):
        batch = pmids[i:i+50]
        print(f"  Fetching batch {i//50 + 1}/{total_batches}...")
        handle = Entrez.efetch(db="pubmed", id=batch, rettype="abstract", retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        for article in records["PubmedArticle"]:
            try:
                pmid = str(article["MedlineCitation"]["PMID"])
                title = str(article["MedlineCitation"]["Article"]["ArticleTitle"])
                abstract_parts = article["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", [])
                abstract_text = " ".join(str(p) for p in abstract_parts)
                journal = str(article["MedlineCitation"]["Article"]["Journal"]["Title"])
                pub_date = article["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["PubDate"]
                pub_date_str = pub_date.get("Year", "") + "-" + pub_date.get("Month", "")
                if not abstract_text:
                    continue
                # cur.execute("""
                #     INSERT OR IGNORE INTO abstracts (pmid, topic, title, abstract_text, journal, pub_date)
                #     VALUES (?, ?, ?, ?, ?, ?)
                # """, (pmid, topic, title, abstract_text, journal, pub_date_str))
                # inserted += 1
                cur.execute("""INSERT OR IGNORE INTO abstracts (pmid, topic, title, abstract_text, journal, pub_date) VALUES (?, ?, ?, ?, ?, ?)""", (pmid, topic, title, abstract_text, journal, pub_date_str))
                if cur.rowcount > 0:
                    inserted += 1

            except (KeyError, IndexError) as e:
                print(f"Skipping malformed record: {e}")
                continue
        conn.commit()
        print(f"    Batch done, {inserted} inserted so far")
        time.sleep(0.4)  # NCBI rate limit: max ~3 req/sec without API key
    print(f"Inserted {inserted} abstracts for topic '{topic}'")
    conn.close()

# if __name__ == "__main__":
#     fetch_topic(
#         query="CRISPR[Title/Abstract] AND gene editing[Title/Abstract]",
#         topic="crispr",
#         max_results=100
#     )
    
TOPICS = [
    {"query": "CRISPR[Title/Abstract] AND gene editing[Title/Abstract]", "topic": "crispr"},
    {"query": '("cancer genomics"[Title/Abstract] OR "tumor genomics"[Title/Abstract]) AND (mutation[Title/Abstract] OR "driver gene"[Title/Abstract])', "topic": "cancer_genomics"},
    {"query": '"genome-wide association study"[Title/Abstract] AND (variant[Title/Abstract] OR locus[Title/Abstract] OR SNP[Title/Abstract])', "topic": "gwas"},
    {"query": '("protein folding"[Title/Abstract] OR "protein structure prediction"[Title/Abstract]) AND (structure[Title/Abstract] OR model[Title/Abstract])', "topic": "protein_folding"},
    {"query": '("cancer immunotherapy"[Title/Abstract] OR "checkpoint inhibitor"[Title/Abstract]) AND (tumor[Title/Abstract] OR "T cell"[Title/Abstract])', "topic": "immunotherapy"},
    {"query": '("neurodegenerative disease"[Title/Abstract] OR "Alzheimer\'s"[Title/Abstract] OR "Parkinson\'s"[Title/Abstract]) AND (genetics[Title/Abstract] OR gene[Title/Abstract] OR variant[Title/Abstract])', "topic": "neurodegenerative"},
]

if __name__ == "__main__":
    for t in TOPICS:
        fetch_topic(query=t["query"], topic=t["topic"], max_results=1000)
        print("---")
