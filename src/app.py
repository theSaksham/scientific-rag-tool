import streamlit as st
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.query_pipeline import QueryPipeline

st.set_page_config(
    page_title="Scientific RAG Tool",
    page_icon="*",
    layout="wide"
)

@st.cache_resource
def load_pipeline():
    return QueryPipeline()

@st.cache_resource
def load_tagger():
    from src.ner_tagger import NERTagger
    return NERTagger()
st.title("Scientific RAG Tool")
st.caption("Biomedical literature Q&A over PubMed abstracts — CRISPR, Cancer Genomics, GWAS, Protein Folding, Immunotherapy, Neurodegeneration")

st.divider()

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Retrieved chunks (top-k)", min_value=3, max_value=10, value=5)
    st.divider()
    st.markdown("**Corpus**")
    st.markdown("- 5,262 PubMed abstracts\n- 11,723 chunks\n- 6 biomedical topics")
    st.divider()
    st.markdown("**Architecture**")
    st.markdown("- Hybrid BM25 + ChromaDB retrieval\n- RRF fusion\n- Two-call Mistral pipeline\n- Local inference via Ollama")

query = st.text_input(
    "Ask a biomedical question:",
    placeholder="e.g. What is the role of tau protein aggregation in Alzheimer's progression?",
    key="query_input"
)

run = st.button("Search", type="primary")

if run and query.strip():
    pipeline = load_pipeline()

    with st.spinner("Classifying topic..."):
        topic_filter = pipeline.classify_topic(query)

    col1, col2 = st.columns([1, 3])
    with col1:
        if topic_filter:
            st.success(f"Topic: `{topic_filter}`")
        else:
            st.info("Topic: `ambiguous` (searching all topics)")

    st.divider()

    with st.spinner("Retrieving chunks and generating answer..."):
        chunks = pipeline.retriever.retrieve(query, top_k=top_k, topic_filter=topic_filter)
        context = "\n\n".join([f"[{i+1}] {c['chunk_text']}" for i, c in enumerate(chunks)])

        from langchain_ollama import OllamaLLM
        from src.query_pipeline import ANSWER_PROMPT
        llm = OllamaLLM(model="mistral", temperature=0)
        answer = llm.invoke(ANSWER_PROMPT.format(context=context, query=query)).strip()

    st.subheader("Answer")
    st.write(answer)

    st.divider()

    # st.subheader(f"Retrieved Context ({len(chunks)} chunks)")
    # for i, chunk in enumerate(chunks):
    #     with st.expander(f"[{i+1}] PMID: {chunk['pmid']} — Topic: {chunk['topic']}"):
    #         st.write(chunk["chunk_text"])

    tagger = load_tagger()
    st.subheader(f"Retrieved Context ({len(chunks)} chunks)")
    for i, chunk in enumerate(chunks):
        entities = tagger.tag(chunk["chunk_text"])
        entity_words = list(set(e["word"] for e in entities)) if entities else []
        label = f"[{i+1}] PMID: {chunk['pmid']} — Topic: {chunk['topic']}"
        if entity_words:
            label += f" — Chemicals: {', '.join(entity_words)}"
        with st.expander(label):
            st.write(chunk["chunk_text"])
            if entities:
                st.markdown("**Detected chemical entities:**")
                for e in entities:
                    st.markdown(f"- `{e['word']}` (score: {e['score']})")

elif run and not query.strip():
    st.warning("Please enter a question.")

with st.expander("Example questions"):
    st.markdown("""
**Clear single-topic:**
- What CRISPR delivery methods reduce off-target effects?
- What checkpoint inhibitors are most studied for melanoma treatment?
- What is the role of tau protein aggregation in Alzheimer's progression?

**Cross-topic (ambiguous):**
- How does CRISPR screening identify cancer driver genes?
- Can CRISPR-engineered T cells improve immunotherapy outcomes?
- What GWAS findings link genetic risk to neurodegenerative disease?

**Entity-specific:**
- What is the function of the BRCA1 gene in DNA repair?
- How does the APOE4 allele affect Alzheimer's disease risk?
- What is the mechanism of PD-1/PD-L1 interaction in immune evasion?
    """)
