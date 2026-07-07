import json
import csv
from langchain_ollama import OllamaLLM
from src.query_pipeline import QueryPipeline

llm = OllamaLLM(model="mistral", temperature=0)

TEST_QUESTIONS = [

    "What CRISPR delivery methods reduce off-target effects?",
    "How do tumor mutational burden scores relate to immunotherapy response?",
    "What GWAS-identified loci are most replicated for type 2 diabetes risk?",
    "How do AlphaFold-style models predict protein secondary structure?",
    "What checkpoint inhibitors are most studied for melanoma treatment?",
    "What is the role of tau protein aggregation in Alzheimer's progression?",

    "How does CRISPR screening identify cancer driver genes?",
    "What GWAS findings link genetic risk to neurodegenerative disease?",
    "How does protein misfolding contribute to neurodegeneration?",
    "What is the relationship between tumor neoantigens and protein structure prediction?",
    "Can CRISPR-engineered T cells improve immunotherapy outcomes?",
    "How do GWAS-identified cancer susceptibility genes inform genomics research?",
    "What role does antigen structure play in immunotherapy design?",
    "Are there shared genetic variants between cancer and neurodegenerative disease risk?",

    "What is the function of the BRCA1 gene in DNA repair?",
    "How does the APOE4 allele affect Alzheimer's disease risk?",
    "What is the mechanism of PD-1/PD-L1 interaction in immune evasion?",
    "How does Cas9 nuclease activity depend on guide RNA specificity?",
]

FAITHFULNESS_PROMPT = """You are an evaluation judge. Given a question, a context, and an answer, 
score the faithfulness of the answer to the context on a scale of 0.0 to 1.0.

Faithfulness means: every claim in the answer is supported by the context. 
Score 1.0 if all claims are grounded in the context.
Score 0.0 if the answer contains claims not found in the context.

Respond with ONLY a number between 0.0 and 1.0. No explanation.

Question: {question}
Context: {context}
Answer: {answer}
Faithfulness score:"""

RELEVANCY_PROMPT = """You are an evaluation judge. Given a question and an answer, 
score how well the answer addresses the question on a scale of 0.0 to 1.0.

Score 1.0 if the answer directly and completely addresses the question.
Score 0.0 if the answer is irrelevant or completely misses the question.

Respond with ONLY a number between 0.0 and 1.0. No explanation.

Question: {question}
Answer: {answer}
Relevancy score:"""

def score(prompt):
    try:
        raw = llm.invoke(prompt).strip()
        import re
        match = re.search(r"\d+\.\d+|\d+", raw)
        val = float(match.group()) if match else 0.0
        return min(max(val, 0.0), 1.0)
    except Exception:
        return 0.0

def run_eval():
    pipeline = QueryPipeline()
    results = []

    for i, question in enumerate(TEST_QUESTIONS):
        print(f"\n[{i+1}/{len(TEST_QUESTIONS)}] {question[:70]}...")
        result = pipeline.answer(question, top_k=5)
        answer = result["answer"]
        topic_filter = result["topic_filter"]

        chunks = pipeline.retriever.retrieve(question, top_k=5, topic_filter=topic_filter)
        context = "\n\n".join([c["chunk_text"] for c in chunks])

        f_score = score(FAITHFULNESS_PROMPT.format(question=question, context=context, answer=answer))
        r_score = score(RELEVANCY_PROMPT.format(question=question, answer=answer))

        results.append({
            "question": question,
            "topic_filter": topic_filter or "ambiguous",
            "answer": answer,
            "faithfulness": f_score,
            "answer_relevancy": r_score,
        })
        print(f"  Faithfulness={f_score:.2f}  Relevancy={r_score:.2f}  topic={topic_filter or 'ambiguous'}")

    avg_f = sum(r["faithfulness"] for r in results) / len(results)
    avg_r = sum(r["answer_relevancy"] for r in results) / len(results)

    print(f"\n{'='*50}")
    print(f"=== EVALUATION RESULTS ({len(results)} questions) ===")
    print(f"{'='*50}")
    print(f"Avg Faithfulness:     {avg_f:.3f}")
    print(f"Avg Answer Relevancy: {avg_r:.3f}")
    print(f"{'='*50}")

    with open("data/eval_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "topic_filter", "faithfulness", "answer_relevancy", "answer"])
        writer.writeheader()
        writer.writerows(results)
    print("Full results saved to data/eval_results.csv")

if __name__ == "__main__":
    run_eval()
