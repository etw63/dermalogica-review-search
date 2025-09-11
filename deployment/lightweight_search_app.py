import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import os
import re
import logging
from typing import List, Optional
from collections import defaultdict
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Dermalogica Review Search - Lightweight")

# Mount static files and templates
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
templates = Jinja2Templates(directory="templates")

class LightweightSearchEngine:
    def __init__(self):
        self.df = None
        self.products = []
        self.embeddings = None
        self.usage_patterns = None
        self.meta_occasions = None
        
    def load_data(self, csv_file: str):
        """Load preprocessed data"""
        logger.info(f"Loading data from {csv_file}...")
        self.df = pd.read_csv(csv_file)
        self.products = sorted(self.df['product_name'].unique().tolist())
        logger.info(f"Loaded {len(self.df)} reviews for {len(self.products)} products")
        
    def load_embeddings(self, embeddings_file: str):
        """Load pre-computed embeddings"""
        logger.info(f"Loading embeddings from {embeddings_file}...")
        with open(embeddings_file, 'rb') as f:
            self.embeddings = pickle.load(f)
        logger.info(f"Loaded embeddings with shape {self.embeddings.shape}")
        
    def load_usage_patterns(self, csv_file: str):
        """Load preprocessed usage patterns"""
        logger.info(f"Loading usage patterns from {csv_file}...")
        self.usage_patterns = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(self.usage_patterns)} reviews with usage patterns")
        
    def load_meta_occasions(self, json_file: str):
        """Load pre-computed meta occasions"""
        logger.info(f"Loading meta occasions from {json_file}...")
        with open(json_file, 'r') as f:
            self.meta_occasions = json.load(f)
        logger.info(f"Loaded {len(self.meta_occasions)} meta occasions")
        
    def search(self, product: str, query: str, limit: int = 1000):
        """Simple search using pre-computed embeddings"""
        if self.embeddings is None:
            return {"error": "Embeddings not loaded"}
            
        # Filter by product
        product_df = self.df[self.df['product_name'] == product]
        if len(product_df) == 0:
            return {"error": f"No reviews found for product: {product}"}
            
        # Simple keyword search for now
        query_lower = query.lower()
        matches = product_df[
            product_df['clean_text'].str.contains(query_lower, case=False, na=False)
        ]
        
        results = []
        for _, row in matches.head(limit).iterrows():
            results.append({
                'text': row['clean_text'],
                'rating': row.get('rating', 'N/A'),
                'source': row.get('source', 'unknown')
            })
            
        return {
            'query': query,
            'product': product,
            'total_results': len(results),
            'results': results
        }

# Initialize search engine
search_engine = LightweightSearchEngine()

@app.on_event("startup")
async def startup_event():
    """Load pre-computed data on startup"""
    try:
        # Load main data
        search_engine.load_data("dermalogica_aggregated_reviews.csv")
        
        # Load embeddings if available
        if os.path.exists("review_embeddings.pkl"):
            search_engine.load_embeddings("review_embeddings.pkl")
        
        # Load usage patterns if available
        if os.path.exists("dermalogica_aggregated_reviews_with_usage_patterns.csv"):
            search_engine.load_usage_patterns("dermalogica_aggregated_reviews_with_usage_patterns.csv")
            
        # Load meta occasions if available
        if os.path.exists("enhanced_meta_occasions.json"):
            search_engine.load_meta_occasions("enhanced_meta_occasions.json")
            
        logger.info("Application startup complete.")
    except Exception as e:
        logger.error(f"Startup error: {e}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, product: str = "daily microfoliant"):
    """Home page with search interface"""
    return templates.TemplateResponse("search.html", {
        "request": request,
        "products": search_engine.products,
        "selected_product": product
    })

@app.post("/search")
async def search_reviews(product: str = Form(...), query: str = Form(...), limit: int = Form(1000)):
    """Search endpoint"""
    return search_engine.search(product, query, limit)

@app.get("/divergent-dashboard", response_class=HTMLResponse)
async def divergent_dashboard(request: Request):
    """Divergent discovery dashboard"""
    if search_engine.meta_occasions is None:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": "Meta occasions not loaded. Please run the analysis locally first."
        })
    
    return templates.TemplateResponse("divergent_dashboard.html", {
        "request": request,
        "products": search_engine.products,
        "meta_occasions": search_engine.meta_occasions
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
