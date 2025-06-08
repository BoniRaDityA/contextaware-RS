from fastapi import FastAPI, Query
from typing import List
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import orjson
from fastapi.responses import JSONResponse
import json
import re
from nltk.stem import PorterStemmer

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

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

stemmer = PorterStemmer()  # Algoritma stemming

def search_products(keywords: list, top_n: int = 18):
    def compute_score(row):
        score = 0
        product_name = str(row['productDisplayName']).lower()
        article_type = str(row['articleType']).lower()
        master_category = str(row['masterCategory']).lower()
        sub_category = str(row['subCategory']).lower()
        base_colour = str(row['baseColour']).lower()

        for kw in keywords:
            stemmed_kw = stemmer.stem(kw)  # Ubah kata ke bentuk dasar
            pattern = rf'\b{stemmed_kw}\b'  # Regex pencocokan kata persis
            
            if re.search(pattern, product_name): score += 5
            if re.search(pattern, article_type): score += 4
            if re.search(pattern, master_category): score += 3
            if re.search(pattern, sub_category): score += 2
            if re.search(pattern, base_colour): score += 1
        
        return score
    
    df['score'] = df.apply(compute_score, axis=1)
    result_df = df[df['score'] > 0].sort_values(by='score', ascending=False).head(top_n)

    return result_df[['id', 'masterCategory', 'articleType', 'baseColour', 'season', 'productDisplayName', 'score', 'link']].to_dict(orient='records')


# def search_products(keywords: list, top_n: int = 18):
#     def compute_score(row):
#         score = 0
#         for kw in keywords:
#             if kw in str(row['productDisplayName']).lower():
#                 score += 5
#             if kw in str(row['articleType']).lower():
#                 score += 4
#             if kw in str(row['masterCategory']).lower():
#                 score += 3
#             if kw in str(row['subCategory']).lower():
#                 score += 2
#             if kw in str(row['baseColour']).lower():
#                 score += 1
#         return score
    
#     df['score'] = df.apply(compute_score, axis=1)
#     result_df = df[df['score'] > 0].sort_values(by='score', ascending=False).head(top_n)
#     return result_df[['id', 'masterCategory','articleType', 'baseColour', 'season', 'productDisplayName', 'score', 'link']].to_dict(orient='records')



# @app.get("/search")
# def search(keyword: str = Query(...), top_n: int = 18):
#     keywords = keyword.lower().split()  # Split into list of words
#     results = search_products(keywords, top_n)
#     return {"results": results}

# @app.get("/search")
# def search(keyword: str = Query(...), top_n: int = 18):
#     keywords = keyword.lower().split()
#     results = search_products(keywords, top_n)
    
#     # Menggunakan orjson untuk format JSON dengan indentasi
#     return orjson.loads(orjson.dumps({"results": results}, option=orjson.OPT_INDENT_2))

@app.get("/search")
def search(keyword: str = Query(...), top_n: int = 18):
    keywords = keyword.lower().split()
    results = search_products(keywords, top_n)
    return JSONResponse(content=json.loads(json.dumps({"results": results}, indent=4)))



