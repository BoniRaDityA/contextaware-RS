from fastapi import FastAPI, Query
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import re

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

# Weights
weights = {
    "productDisplayName": 5,
    "articleType": 4,
    "masterCategory": 3,
    "subCategory": 2,
    "baseColour": 1
}

# Daftar kata yang ingin diabaikan dalam pencarian
excluded_keywords = {"men", "women", "shirt", "tshirt", "male", "female", "unisex", "clothing"}

def search_products(keywords: list, top_n: int = 18):
    def compute_score(row):
        score = 0
        for kw in keywords:
            if kw in excluded_keywords:
                continue  # Lewati kata yang tidak diinginkan
            
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


@app.get("/search")
def search(keyword: str = Query(...), top_n: int = 18):
    keywords = keyword.lower().split()  # Split into list of words
    results = search_products(keywords, top_n)
    return {"results": results}
