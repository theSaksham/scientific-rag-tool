# Scientific RAG Tool

A hybrid retrieval-augmented generation system for biomedical literature Q&A, built on top of PubMed abstracts. Runs fully locally -- no OpenAI API, no external dependencies at inference time.

## What it does

You ask a biomedical question. The system retrieves the most relevant chunks from a corpus of ~5,200 PubMed abstracts, generates an answer grounded in that context using Mistral, and tags any chemical entities found in the retrieved chunks using a fine-tuned BioBERT model.

Covers six topic areas: CRISPR/gene editing, cancer genomics, GWAS, protein folding, immunotherapy, and neurodegenerative disease genetics.

## Architecture

```
Query
  |
  v
Call 1: Mistral classifies topic (single-topic or ambiguous)
  |
  v
Hybrid Retriever
  |- BM25 keyword search (rank_bm25)
  |- Vector search (ChromaDB + all-MiniLM-L6-v2 embeddings)
  |- RRF fusion (Reciprocal Rank Fusion, k=60)
  |
  v
Call 2: Mistral generates answer from retrieved context
  |
  v
BioBERT NER tags chemical entities in retrieved chunks
  |
  v
Streamlit UI renders answer + sources + entities
```

## Stack

- LLM: Mistral 7B via Ollama (local inference)
- Embeddings: sentence-transformers all-MiniLM-L6-v2
- Vector store: ChromaDB (persistent, cosine similarity)
- Keyword search: BM25Okapi (rank_bm25)
- NER: BioBERT fine-tuned on BC5CDR chemical corpus
- Data: PubMed abstracts via NCBI Entrez (Biopython)
- Storage: SQLite (raw abstracts + chunks), ChromaDB (vectors), pickle (BM25 index)
- UI: Streamlit

## Corpus stats

- 5,262 PubMed abstracts across 6 topics
- 11,723 chunks (chunk size 800 chars, overlap 100)
- Embedded in ~450s on CPU using MiniLM

## Evaluation

Evaluated on 18 questions (6 clear single-topic, 8 cross-topic ambiguous, 4 entity-specific) using Mistral-as-judge:

- Avg Faithfulness: 0.814
- Avg Answer Relevancy: 0.858
- 14/18 questions scored above 0.85 on both metrics
- 2 failures traced to corpus coverage gaps on rare cross-domain topics (neoantigens + protein structure prediction, shared cancer/neurodegenerative variants)

NER model (BioBERT on BC5CDR chemical corpus):

- Test F1: 0.906
- Precision: 0.918
- Recall: 0.895

## Project structure

```
scientific_rag_tool/
  data/
    abstracts.db          -- SQLite database (abstracts + chunks tables)
    chroma_db/            -- ChromaDB persistent storage
    bm25_index.pkl        -- BM25 index (pickled)
    eval_results.csv      -- Per-question evaluation scores
  models/
    biobert_bc5cdr_ner/   -- Fine-tuned BioBERT NER model
  src/
    init_db.py            -- SQLite schema creation
    fetch_abstracts.py    -- NCBI Entrez data ingestion
    chunk_abstracts.py    -- Text chunking (RecursiveCharacterTextSplitter)
    embed_chunks.py       -- Embedding + ChromaDB ingestion
    build_bm25.py         -- BM25 index construction
    retriever.py          -- Hybrid retriever (BM25 + vector + RRF)
    query_pipeline.py     -- Two-call Mistral query pipeline
    ner_tagger.py         -- BioBERT chemical NER inference
    evaluate.py           -- Custom Mistral-as-judge evaluation harness
    app.py                -- Streamlit UI
```

## Setup

**Requirements:**
- Python 3.11
- Ollama with Mistral pulled (`ollama pull mistral`)
- Conda (recommended)

**Install:**

```bash
conda create -n scientific_rag_tool python=3.11 -y
conda activate scientific_rag_tool
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Initialize database and ingest data:**

```bash
python src/init_db.py
python src/fetch_abstracts.py
python src/chunk_abstracts.py
python src/embed_chunks.py
python src/build_bm25.py
```

**Run the UI:**

```bash
streamlit run src/app.py
```

**Run evaluation:**

```bash
python src/evaluate.py
```

## Limitations

- NER model detects chemicals only (BC5CDR scope) -- genes and proteins are out of scope for the current model
- Two test questions scored 0.0 on both metrics due to corpus coverage gaps, not retrieval failure
- Mistral-as-judge evaluation is self-referential; scores should be interpreted as relative, not absolute
- BM25 index is rebuilt from scratch if chunk table changes -- no incremental update
