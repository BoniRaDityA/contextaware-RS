from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import pandas as pd
import re

app = FastAPI()

# Load dataset sekali saat aplikasi mulai
df = pd.read_csv("cleaned_fashion.csv")

# Bobot pencocokan skor
weights = {
    "productDisplayName": 5,
    "articleType": 4,
    "masterCategory": 3,
    "subCategory": 2,
    "baseColour": 1,
    "gender": 1
}

class ProductResult(BaseModel):
    id: int
    name: str
    score: int
    imageUrl: str

class SearchResponse(BaseModel):
    results: List[ProductResult]

# Pengecualian untuk mencegah pencarian yang salah
exclusion_map = {
    "shirt": ["tshirt", "tanktop"],
    "jeans": ["jeggings"],
    "dress": ["jumper", "romper"]
}

def search_products(keywords: List[str], top_n: int = 10):
    keywords = [kw.lower() for kw in keywords]
    exclusions = sum([exclusion_map[kw] for kw in keywords if kw in exclusion_map], [])

    def row_matches(row):
        row_text = " ".join(str(row.get(col, '')).lower() for col in weights.keys())
        tokens = re.findall(r'\b[\w-]+\b', row_text)
        return all(kw in tokens for kw in keywords) and not any(ex in tokens for ex in exclusions)

    filtered_df = df[df.apply(row_matches, axis=1)]

    if filtered_df.empty:
        return []

    filtered_df['score'] = filtered_df.apply(
        lambda row: sum(weights[col] * str(row.get(col, '')).lower().count(kw) for kw in keywords for col in weights.keys()), axis=1
    )

    result_df = filtered_df.sort_values(by='score', ascending=False).head(top_n)

    return [{"id": int(row['id']), "name": row['productDisplayName'], "score": int(row['score']), "imageUrl": row['link']} for _, row in result_df.iterrows()]

@app.get("/search")
def search(keyword: str = Query(...), top_n: int = 10):
    results = search_products(keyword.lower().split(), top_n)
    return {"results": results}
