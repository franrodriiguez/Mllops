from fastapi import FastAPI 
from pydantic import BaseModel
from typing import Optional
from transformers import pipeline
from collections import Counter

app = FastAPI()




@app.get('/text-stats')
def text_stats(text: str):
    return {
        "words": len(text.split()),
        "characters": len(text),
        "sentences": len(text.split('.'))
    }

@app.get('/text-cleaner')
def text_cleaner(text: str):
    cleaned_text = " ".join(text.lower().strip().split())
    return {"cleaned_text": cleaned_text}


STOPWORDS = {
    "the","a","an","is","are","of","to",
    "and","in","on","for","with","this","that"
}

@app.get('/keyword-count')
def keyword_count(text: str):

    words = [
        word
        for word in text.lower().split()
        if word not in STOPWORDS
    ]

    most_common = Counter(words).most_common(3)

    return {"top_3_words": most_common}

token_classifier_pipe = pipeline(
    "token-classification",
    model="dslim/bert-base-NER"
)

@app.get('/token_classifier')
def token_classifier(text: str): 
    tokens = token_classifier_pipe(text)

    clean_tokens = []

    for token in tokens:
        clean_tokens.append({
            "entity": token["entity"],
            "score": float(token["score"]),
            "index": int(token["index"]),
            "word": token["word"],
            "start": int(token["start"]),
            "end": int(token["end"])
        })

    return clean_tokens

@app.get('/fill-mask')
def fill_mask(text: str): 
    pipe = pipeline("fill-mask") 
    result = pipe(text)
    return result