from fastapi import FastAPI, Query
from typing import List
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

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

def search_products(keywords: list, top_n: int = 18):
    def compute_score(row):
        score = 0
        for kw in keywords:
            if kw in str(row['productDisplayName']).lower():
                score += 4
            if kw in str(row['articleType']).lower():
                score += 3
            if kw in str(row['season']).lower():
            	score += 5
            if kw in str(row['gender']).lower():
            	score +=5
            if kw in str(row['masterCategory']).lower():
                score += 3
            if kw in str(row['subCategory']).lower():
                score += 2
            if kw in str(row['baseColour']).lower():
                score += 3
        return score
    
    df['score'] = df.apply(compute_score, axis=1)
    result_df = df[df['score'] > 0].sort_values(by='score', ascending=False).head(top_n)
    return result_df[['id', 'masterCategory','articleType', 'baseColour', 'season', 'productDisplayName', 'score', 'link', 'gender']].to_dict(orient='records')



@app.get("/search")
def search(keyword: str = Query(...), top_n: int = 18):
    if keyword == "":
        results = df.sample(n=top_n).to_dict(orient="records")
    else:
        keyword = keyword.lower().split()
        results = search_products(keyword, top_n)
    return {"results": results}


