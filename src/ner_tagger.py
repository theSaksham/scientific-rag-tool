from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

MODEL_PATH = "models/biobert_bc5cdr_ner"

class NERTagger:
    def __init__(self):
        print("Loading NER model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
        self.pipe = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="first",
            device=-1 
        )
        print("NER model ready.")

    def tag(self, text):
        results = self.pipe(text)
        entities = []
        for r in results:
            if r["score"] >= 0.85:
                entities.append({
                    "entity": r["entity_group"],
                    "word": r["word"],
                    "score": round(r["score"], 3),
                    "start": r["start"],
                    "end": r["end"],
                })
        return entities

    def tag_chunks(self, chunks):
        for chunk in chunks:
            chunk["entities"] = self.tag(chunk["chunk_text"])
        return chunks


if __name__ == "__main__":
    tagger = NERTagger()
    test_texts = [
        "Selegiline-induced postural hypotension in Parkinson's disease.",
        "BRCA1 mutations are associated with increased risk of breast cancer.",
        "The APOE4 allele significantly increases Alzheimer's disease risk.",
    ]
    for text in test_texts:
        entities = tagger.tag(text)
        print(f"\nText: {text}")
        print(f"Entities: {entities}")
        