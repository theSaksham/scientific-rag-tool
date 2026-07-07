# Project Journey -- Scientific RAG Tool

This document covers what I built, what broke, what I learned, and the decisions I made along the way. Written for anyone reading the code who wants context beyond what the README covers.

---

## Why I built this

This project came out of wanting to build something that combined retrieval-augmented generation with real scientific data - not a tutorial dataset, actual PubMed literature.

The goal was to have something I could explain end to end in an interview: every architectural decision, every number, every failure.

---

## What I decided upfront

**Six topics:** CRISPR/gene editing, cancer genomics, GWAS, protein folding, immunotherapy, neurodegenerative disease genetics. I picked these because they have enough conceptual overlap to make query routing genuinely hard. For example, "How does CRISPR screening identify cancer driver genes?" spans two topics - a retriever that only searches one topic will miss relevant context.

**18 test questions before writing any code:** 6 clear single-topic, 8 deliberately ambiguous cross-topic, 4 entity-specific. Having these locked in before building anything meant I couldn't tune the system to the test set after seeing results.

**Fully local inference:** Mistral via Ollama, no OpenAI API. This was a deliberate constraint - I wanted the project to work for anyone who clones the repo without needing an API key.

---

## Storage decision -- SQLite instead of PostgreSQL

I initially planned to use PostgreSQL. Midway through setup I switched to SQLite. The reasoning:

- No server process to manage -- SQLite is just a file
- ~5,200 abstracts is well within SQLite's comfort zone
- If I need to re-embed (different model, different chunk size), the raw text lives in SQLite independently of ChromaDB, so I just truncate the chunks table and re-run

The tradeoff is I lose the "used PostgreSQL" line for this specific project, but I already have that from other work. Matching the storage engine to the actual scale of the problem felt like the more defensible engineering choice.

---

## Data ingestion

Used NCBI Entrez via Biopython to pull PubMed abstracts. A few things worth noting:

**The timeout problem:** First run of fetch_abstracts.py hung silently after printing the PMID count. Biopython's Entrez calls have no default timeout -- if NCBI drops a connection mid-batch, you wait forever. Fixed with `socket.setdefaulttimeout(30)` and batch-level progress prints. After that it ran cleanly.

**Abstracts without abstracts:** About 18% of PubMed records returned by the search had no abstract text -- title only, or just metadata. These are useless for RAG so I skipped them. Final yield was roughly 80-82% of requested PMIDs.

**The duplicate chunk bug:** I ran chunk_abstracts.py twice before realizing it had no deduplication. The chunks table ended up with 23,402 rows instead of 11,723. The SQLite `COUNT(*)` query caught it. Fixed by adding a `DELETE FROM chunks` at the start of the chunking script, then re-ran embedding. The printed "Inserted N" counter during ingestion also had a bug -- it incremented on every `INSERT OR IGNORE` attempt, not on actual inserts. Fixed with `if cur.rowcount > 0`. The database was always correct (UNIQUE constraint enforced at the DB level); only the console log was lying.

**Chunk size decision:** I chose chunk_size=800 chars with 100 char overlap expecting most abstracts to pass through as single chunks. That was wrong -- most PubMed abstracts run 800-1500 chars, so 86% of them got split into 2 chunks. The result is 11,723 chunks from 5,262 abstracts, avg ~2.2 chunks per abstract. In retrospect smaller chunks improve retrieval precision (less irrelevant text per match), so this was fine, just not what I predicted.

---

## Retrieval

**Why hybrid BM25 + vector?** Vector search alone misses exact keyword matches -- if a user asks about "APOE4" and no retrieved chunk contains that exact string but several discuss "apolipoprotein E4", pure vector search handles it fine. But the reverse also happens: vector search sometimes retrieves thematically similar but factually different chunks. BM25 anchors retrieval to the actual query terms. RRF fusion handles combining the two rankings without needing score normalization across different scales.

**RRF constant k=60:** This is the standard value from the original RRF paper. I didn't tune it.

**Topic filter:** The two-call pipeline classifies the query first -- single topic or ambiguous. Single-topic queries pass a `where` filter to ChromaDB and a post-score filter to BM25. Ambiguous queries search the full corpus. This required some prompt engineering to get right.

**Classifier prompt iteration:** The first version of the classifier prompt misclassified "How does CRISPR screening identify cancer driver genes?" as `crispr` instead of `ambiguous`. The fix was adding few-shot examples directly in the prompt showing exactly this failure case. After adding examples, 4/4 test classification cases were correct.

---

## Evaluation

I originally planned to use RAGAS for evaluation. It failed to import:

```
ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
```

RAGAS 0.4.3 has a broken import -- it tries to load ChatVertexAI from langchain_community which was removed in the version installed. Rather than trying to fix the dependency, I implemented a custom evaluation harness using Mistral as the judge with structured prompts for faithfulness and answer relevancy. Two calls per question -- one for faithfulness, one for relevancy.

Final scores on 18 questions:
- Avg Faithfulness: 0.814
- Avg Answer Relevancy: 0.858

Two questions scored 0.0 on both metrics:
- "What is the relationship between tumor neoantigens and protein structure prediction?" -- corpus gap, this intersection barely exists in the literature
- "Are there shared genetic variants between cancer and neurodegenerative disease risk?" -- same, genuinely rare research area

These are corpus coverage failures, not retrieval failures. The retriever returned the best available chunks; there just weren't relevant ones. A larger corpus pull or more targeted queries for these topics would likely fix it.

---

## NER module

Fine-tuned `dmis-lab/biobert-base-cased-v1.2` on BC5CDR chemical NER corpus (4,560 train sentences) on Colab T4.

Getting the dataset loaded was painful -- every BC5CDR version on HuggingFace uses old-style loading scripts which newer versions of the `datasets` library no longer support. Eventually downloaded the CoNLL-format data directly from a GitHub mirror and parsed it manually with a standard CoNLL parser.

Getting transformers to import cleanly in Colab was also painful -- multiple version conflicts between the pre-installed huggingface_hub and different transformers versions. The working combination was `transformers==4.44.0` with Colab's existing huggingface_hub.

Training results:

| Epoch | Val F1 |
|-------|--------|
| 1     | 0.920  |
| 2     | 0.926  |
| 3     | 0.926  |
| 4     | 0.925  |

Test F1: 0.906 | Precision: 0.918 | Recall: 0.895

One post-training fix: `aggregation_strategy="simple"` was returning subword tokens as separate entities ("se" and "##legiline" instead of "selegiline"). Changed to `aggregation_strategy="first"` which correctly merges them.

**Scope limitation:** BC5CDR covers chemicals and drugs only. The model does not detect genes, proteins, or diseases. The original project plan described a broader NER scope -- this is an honest narrowing based on what the training data actually covers.

---

## What I would do differently

- Use a larger corpus from the start (10K+ per topic) -- the two corpus-gap failures would likely not appear
- Add a confidence score to the topic classifier output -- "crispr with 0.4 confidence" should probably be treated as ambiguous
- Run evaluation with a non-self-referential judge -- Mistral judging Mistral answers is inherently biased
- Add NCBI API key to increase rate limits -- unauthenticated pulls are capped at 3 req/sec which made ingestion slow

---

## Timeline

- Environment setup + DB schema: ~1 hour
- Data ingestion (6 topics, ~5200 abstracts): ~45 minutes including debugging the timeout hang
- Chunking + embedding: ~8 hours wall clock (embedding 11,723 chunks took 450s on CPU)
- BM25 index: ~5 seconds
- Hybrid retriever + query pipeline: ~2 hours
- Evaluation harness: ~1 hour (after RAGAS failed)
- Evaluation run: ~35 minutes
- NER fine-tuning on Colab T4: ~30 minutes training, ~3 hours debugging environment
- Streamlit UI: ~1 hour
- Total: roughly 3-4 days of actual work spread across a week
