from fastapi import FastAPI, Query
import pandas as pd
import re
from fastapi.middleware.cors import CORSMiddleware
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # or 3000 if using CRA
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset once at startup
df = pd.read_csv("cleaned_fashion.csv")

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Weights
weights = {
    "productDisplayName": 5,
    "articleType": 4,
    "masterCategory": 3,
    "subCategory": 2,
    "baseColour": 1
}

# Kata-kata yang harus dihindari dalam pencarian
excluded_keywords = {"men", "women", "shirt", "tshirt", "male", "female", "unisex", "clothing"}

def preprocess_text(text):
    """ Membersihkan dan memproses teks sebelum digunakan dalam pencarian. """
    text = text.lower()  # Konversi ke huruf kecil
    text = re.sub(r"[^a-z0-9\s]", "", text)  # Hapus karakter khusus
    words = word_tokenize(text)  # Tokenisasi kata
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word not in excluded_keywords]  # Lemmatization & Stopword removal
    return words

def search_products(keywords: list, top_n: int = 18):
    def compute_score(row):
        score = 0
        for kw in keywords:
            if kw in str(row['productDisplayName']).lower():
                score += weights["productDisplayName"]
            if kw in str(row['articleType']).lower():
                score += weights["articleType"]
            if kw in str(row['masterCategory']).lower():
                score += weights["masterCategory"]
            if kw in str(row['subCategory']).lower():
                score += weights["subCategory"]
            if kw in str(row['baseColour']).lower():
                score += weights["baseColour"]
        return score
    
    df['score'] = df.apply(compute_score, axis=1)
    result_df = df[df['score'] > 0].sort_values(by='score', ascending=False).head(top_n)
    return result_df[['id', 'masterCategory', 'articleType', 'baseColour', 'season', 'productDisplayName', 'score', 'link']].to_dict(orient='records')

@app.get("/")
def root():
    return {"message": "FastAPI Context-Aware Recommender System is running!"}

@app.get("/search")
def search(keyword: str = Query(...), top_n: int = 18):
    keywords = preprocess_text(keyword)  # Preprocessing sebelum pencarian
    results = search_products(keywords, top_n)
    return {"results": results}
