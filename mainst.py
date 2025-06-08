from fastapi import FastAPI, Query
from typing import List
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer

app = FastAPI()

# Load dataset once at startup
df = pd.read_csv("cleaned_fashion.csv")

# Inisialisasi stemmer
stemmer = PorterStemmer()

# Weights
weights = {
    "productDisplayName": 5,
    "articleType": 4,
    "masterCategory": 3,
    "subCategory": 2,
    "baseColour": 1
}

def preprocess_word(word: str) -> str:
    # Lowercase dan hapus karakter non-alphanumeric
    word = word.lower()
    word = re.sub(r'[^a-z0-9]', '', word)
    # Stemming
    return stemmer.stem(word)

def preprocess_keywords(keywords: List[str]) -> List[str]:
    return [preprocess_word(kw) for kw in keywords]

def search_products(keywords: list, top_n: int = 10):
    def compute_score(row):
        score = 0
        # Preprocess setiap kolom saat pengecekan kata
        # Proses setiap kata dalam kolom ke bentuk processed yang sederhana (lower+stem)
        processed_cols = {}
        for col in weights.keys():
            val = str(row[col]).lower()
            # Tokenize kolom berdasarkan spasi
            tokens = val.split()
            # Preprocess setiap token di kolom
            processed_cols[col] = [preprocess_word(token) for token in tokens]

        for kw in keywords:
            for col, weight in weights.items():
                if kw in processed_cols[col]:
                    score += weight
        return score

    df['score'] = df.apply(compute_score, axis=1)
    result_df = df[df['score'] > 0].sort_values(by='score', ascending=False).head(top_n)
    return result_df[['id', 'productDisplayName', 'score', 'link']].to_dict(orient='records')

@app.get("/search")
def search(keyword: str = Query(...), top_n: int = 10): 
    keywords = keyword.lower().split()  # Split menjadi kata-kata
    keywords = preprocess_keywords(keywords)  # Preprocess kata-kata
    results = search_products(keywords, top_n)
    return {"results": results}
