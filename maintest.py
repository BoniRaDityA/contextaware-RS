from fastapi import FastAPI, Query
from typing import List
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Middleware untuk CORS agar frontend bisa mengakses
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # or 3000 jika menggunakan CRA
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset sekali saat aplikasi mulai
df = pd.read_csv("cleaned_fashion.csv")

# Inisialisasi stemmer
stemmer = PorterStemmer()

# Bobot skor pencocokan
weights = {
    "productDisplayName": 5,
    "articleType": 4,
    "masterCategory": 3,
    "subCategory": 2,
    "baseColour": 1,
    "gender": 1,
    "season": 5
}

# Daftar pengecualian otomatis untuk kata-kata yang mirip
exclusion_map = {
    "shirt": ["tshirt", "tanktop"],
    "jeans": ["jeggings"],
    "dress": ["jumper", "romper"]
}

# Fungsi preprocessing kata
def preprocess_word(word: str) -> str:
    return stemmer.stem(re.sub(r'[^a-z0-9]', '', word.lower()))  # Bersihkan & lakukan stemming

def search_products(keywords: list, top_n: int = 18):
    keywords = [preprocess_word(kw) for kw in keywords]
    exclusions = sum([exclusion_map.get(kw, []) for kw in keywords], [])

    def compute_score(row):
        score = 0
        processed_cols = {col: [preprocess_word(token) for token in str(row[col]).lower().split()] for col in weights.keys()}
        
        for kw in keywords:
            for col, weight in weights.items():
                if kw in processed_cols[col]:
                    score += weight
        return score

    df['score'] = df.apply(compute_score, axis=1)
    result_df = df[df['score'] > 0].sort_values(by='score', ascending=False)

    # Filter hasil agar tidak mengandung kata dalam exclusions
    result_df = result_df[~result_df['productDisplayName'].str.contains('|'.join(exclusions), case=False, na=False)]
    
    return result_df[['id', 'productDisplayName', 'score', 'link', 'articleType', 'masterCategory', 'baseColour', 'season', 'gender']].head(top_n).to_dict(orient='records')

@app.get("/search")
def search(keyword: str = Query(...), top_n: int = 18):
    if not keyword:
        results = df.sample(n=top_n).to_dict(orient="records")
    else:
        keywords = keyword.lower().split()
        results = search_products(keywords, top_n)
    return {"results": results}
