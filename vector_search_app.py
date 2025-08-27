import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import os
from typing import List, Optional
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Dermalogica Review Search")

# Mount static files and templates
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
templates = Jinja2Templates(directory="templates")

# Initialize the sentence transformer model for embeddings
print("Loading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

class ReviewSearchEngine:
    def __init__(self):
        self.df = None
        self.products = []
        self.embeddings = None
        
    def clean_product_name(self, name):
        """Extract clean product name from messy names"""
        if not name or name == 'Dermalogica Product':
            return None
            
        name_str = str(name).strip()
        
        # If it's already clean (no prices, ratings, etc.), return as is
        if not any(char in name_str for char in ['$', '★', 'out of', 'stars', 'reviews']):
            # Remove "dermalogica" prefix if present
            clean = name_str.lower().replace('dermalogica ', '').strip()
            if clean and len(clean) > 2:
                return clean
        
        # Extract clean name from messy format
        # Remove common prefixes
        clean = name_str
        prefixes_to_remove = [
            'Only at Ulta ', 'ULTA BEAUTY EXCLUSIVE ', 
            'Dermalogica ', '2 sizes ', 'Mini '
        ]
        
        for prefix in prefixes_to_remove:
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
        
        # Find the end of the product name (before price/rating info)
        end_markers = [' $', ' 4.', ' 5.', ' 3.', ' 2.', ' 1.', ' out of', ' Kit Price']
        min_end = len(clean)
        
        for marker in end_markers:
            pos = clean.find(marker)
            if pos > 0 and pos < min_end:
                min_end = pos
        
        clean = clean[:min_end].strip()
        
        # Convert to lowercase and clean up
        clean = clean.lower()
        
        # Remove common suffixes
        suffixes_to_remove = [
            ' moisturizer', ' cleanser', ' serum', ' cream', ' gel', ' oil', 
            ' toner', ' exfoliator', ' mask', ' treatment', ' sunscreen',
            ' spf 30', ' spf 40', ' spf 50', ' kit', ' set'
        ]
        
        original_clean = clean
        for suffix in suffixes_to_remove:
            if clean.endswith(suffix):
                clean = clean[:-len(suffix)].strip()
                break
        
        # If cleaning removed too much, use original
        if len(clean) < 3:
            clean = original_clean
            
        return clean if clean and len(clean) > 2 else None
    
    def load_data(self, csv_file: str):
        """Load the aggregated reviews CSV file"""
        print(f"Loading data from {csv_file}...")
        self.df = pd.read_csv(csv_file)
        
        # Clean and prepare the data
        self.df = self.df.dropna(subset=['review_text', 'product_name'])
        self.df['review_text'] = self.df['review_text'].astype(str)
        self.df['product_name'] = self.df['product_name'].astype(str)
        
        # Create clean product names
        print("Cleaning product names...")
        self.df['clean_product_name'] = self.df['product_name'].apply(self.clean_product_name)
        
        # Filter out rows where we couldn't extract a clean name
        self.df = self.df.dropna(subset=['clean_product_name'])
        
        # Get unique clean products for dropdown
        unique_products = self.df['clean_product_name'].unique()
        self.products = sorted([p for p in unique_products if p and p.strip()])
        
        print(f"Loaded {len(self.df)} reviews for {len(self.products)} products")
        
    def create_vector_database(self):
        """Create embeddings for all reviews"""
        embeddings_file = "review_embeddings.pkl"
        
        # Try to load existing embeddings
        if os.path.exists(embeddings_file):
            print("Loading existing vector database...")
            with open(embeddings_file, 'rb') as f:
                self.embeddings = pickle.load(f)
            print("Loaded existing vector database")
            return
            
        print("Creating new vector database...")
        
        # Create embeddings for all reviews
        review_texts = self.df['review_text'].tolist()
        
        # Process in batches to avoid memory issues
        batch_size = 500
        all_embeddings = []
        
        for i in range(0, len(review_texts), batch_size):
            batch_end = min(i + batch_size, len(review_texts))
            batch_texts = review_texts[i:batch_end]
            
            print(f"Processing batch {i//batch_size + 1}/{(len(review_texts)-1)//batch_size + 1}")
            
            batch_embeddings = model.encode(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        self.embeddings = np.vstack(all_embeddings)
        
        # Save embeddings
        with open(embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
        
        print(f"Vector database created with {len(self.df)} reviews")
    
    def search_reviews(self, product_name: str, query_text: str, limit: int = 10):
        """Search for reviews matching the query text for a specific product"""
        if self.embeddings is None:
            return []
        
        # Create embedding for the query
        query_embedding = model.encode([query_text])
        
        # Filter dataframe by product if specified
        if product_name != "all":
            # Use like match for product names
            mask = self.df['clean_product_name'].str.contains(product_name, case=False, na=False)
            filtered_df = self.df[mask]
            if filtered_df.empty:
                return []
            # Get the positional indices (not DataFrame indices) for embeddings
            filtered_positions = []
            for i, (idx, row) in enumerate(self.df.iterrows()):
                if mask.iloc[i]:
                    filtered_positions.append(i)
            filtered_embeddings = self.embeddings[filtered_positions]
        else:
            filtered_df = self.df
            filtered_positions = list(range(len(self.df)))
            filtered_embeddings = self.embeddings
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, filtered_embeddings)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:limit]
        
        # Format results
        formatted_results = []
        for idx in top_indices:
            # Get the original positional index
            original_position = filtered_positions[idx]
            row = self.df.iloc[original_position]
            similarity_score = similarities[idx]
            
            formatted_results.append({
                'review_text': row['review_text'],
                'product_name': row['clean_product_name'],
                'source': row['source'],
                'rating': str(row.get('rating', '')),
                'title': str(row.get('title', '')),
                'reviewer': str(row.get('reviewer', '')),
                'date': str(row.get('date', '')),
                'similarity_score': float(similarity_score)
            })
        
        return formatted_results

# Initialize the search engine
search_engine = ReviewSearchEngine()

@app.on_event("startup")
async def startup_event():
    """Initialize the database on startup"""
    search_engine.load_data("dermalogica_aggregated_reviews.csv")
    search_engine.create_vector_database()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main search page"""
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "products": search_engine.products
    })

@app.get("/products")
async def get_products():
    """API endpoint to get all available products"""
    return {"products": search_engine.products}

@app.post("/search")
async def search_reviews(
    request: Request,
    product: str = Form(...),
    query: str = Form(...),
    page: int = Form(default=1)
):
    """Search for reviews with pagination"""
    if not query.strip():
        return templates.TemplateResponse("index.html", {
            "request": request,
            "products": search_engine.products,
            "error": "Please enter a search query"
        })
    
    # Get all results first
    all_results = search_engine.search_reviews(product, query, 1000)
    
    # Pagination logic
    per_page = 25
    total_results = len(all_results)
    total_pages = (total_results + per_page - 1) // per_page  # Ceiling division
    
    # Ensure page is within valid range
    page = max(1, min(page, total_pages)) if total_pages > 0 else 1
    
    # Get results for current page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    results = all_results[start_idx:end_idx]
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "products": search_engine.products,
        "results": results,
        "selected_product": product,
        "search_query": query,
        "current_page": page,
        "total_pages": total_pages,
        "total_results": total_results,
        "per_page": per_page,
        "start_result": start_idx + 1 if results else 0,
        "end_result": start_idx + len(results) if results else 0
    })

@app.get("/api/search")
async def api_search(product: str, query: str, limit: int = 1000):
    """API endpoint for programmatic search"""
    results = search_engine.search_reviews(product, query, limit)
    return {"results": results, "count": len(results)}

@app.get("/export")
async def export_reviews(product: str, query: str, limit: int = 1000):
    """Export search results as CSV"""
    import io
    
    # Get search results
    results = search_engine.search_reviews(product, query, limit)
    
    if not results:
        # Return empty CSV with headers
        csv_content = "product_name,source,rating,title,review_text,reviewer,date,similarity_score\n"
    else:
        # Convert results to CSV
        df = pd.DataFrame(results)
        # Reorder columns for better CSV layout
        columns = ['product_name', 'source', 'rating', 'title', 'review_text', 'reviewer', 'date', 'similarity_score']
        df = df[columns]
        
        # Convert to CSV string
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
    
    # Create filename
    product_clean = product.replace(' ', '_').replace(',', '_')
    query_clean = query.replace(' ', '_').replace(',', '_')
    filename = f"dermalogica_reviews_{product_clean}_{query_clean}.csv"
    
    # Return as streaming response
    return StreamingResponse(
        io.StringIO(csv_content),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
