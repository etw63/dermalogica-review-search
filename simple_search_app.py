import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import json
import os
import re
import logging
from typing import List, Optional
from collections import defaultdict
import uvicorn

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Dermalogica Review Search")

# Mount static files and templates
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
templates = Jinja2Templates(directory="templates")

# No heavy ML model loading for deployment version

class ReviewSearchEngine:
    def __init__(self):
        self.df = None
        self.products = []
        self.embeddings = None
        
    def preprocess_text(self, text):
        """Preprocess text for better embedding quality"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\-\.\,\!\?]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def has_exact_keywords(self, text, query):
        """Check if text contains exact keywords from query"""
        text_lower = text.lower()
        query_words = query.lower().split()
        
        # Check for exact phrase first
        if query.lower() in text_lower:
            return True, 1.0  # Exact phrase match
        
        # Check for all keywords present
        words_found = sum(1 for word in query_words if word in text_lower)
        keyword_score = words_found / len(query_words)
        
        return keyword_score > 0.5, keyword_score  # At least half the words must match
    
    def clean_product_name(self, name):
        """Extract clean product name from messy names"""
        if not name or name == 'Dermalogica Product':
            return None
            
        name_str = str(name).strip()
        
        # If it's already clean (no prices, ratings, etc.), return as is
        if not any(char in name_str for char in ['$', 'â˜…', 'out of', 'stars', 'reviews']):
            # Remove "dermalogica" prefix if present
            clean = name_str.lower().replace('dermalogica ', '').strip()
            if clean and len(clean) > 2:
                return clean
        
        # Extract clean name from messy format
        # Remove common prefixes
        clean = name_str
        prefixes_to_remove = [
            'Only at Ulta ', 'ULTA BEAUTY EXCLUSIVE ', 
            'Dermalogica ', '2 sizes ', 'Mini ', 'Clear Start '
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
        
        # Preprocess review text for better embeddings
        print("Preprocessing review text...")
        self.df['processed_review_text'] = self.df['review_text'].apply(self.preprocess_text)
        
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
        
        # Create embeddings for all processed reviews
        review_texts = self.df['processed_review_text'].tolist()
        
        # Process in batches to avoid memory issues
        batch_size = 500
        all_embeddings = []
        
        for i in range(0, len(review_texts), batch_size):
            batch_end = min(i + batch_size, len(review_texts))
            batch_texts = review_texts[i:batch_end]
            
            print(f"Processing batch {i//batch_size + 1}/{(len(review_texts)-1)//batch_size + 1}")
            
            batch_embeddings = get_embeddings(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        self.embeddings = np.vstack(all_embeddings)
        
        # Save embeddings
        with open(embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
        
        print(f"Vector database created with {len(self.df)} reviews")
    
    def search_reviews(self, product_name: str, query_text: str, limit: int = 10):
        """Search for reviews matching the query text for a specific product"""
        logger.info(f"Starting search: product='{product_name}', query='{query_text}', limit={limit}")
        
        if self.embeddings is None:
            logger.error("No embeddings available")
            return []
        
        # Clean the query
        clean_query = self.preprocess_text(query_text)
        if not clean_query:
            logger.warning(f"Query '{query_text}' was empty after preprocessing")
            return []
        
        logger.info(f"Cleaned query: '{clean_query}'")
        
        # Filter dataframe by product if specified
        if product_name != "all":
            mask = self.df['clean_product_name'].str.contains(product_name, case=False, na=False)
            filtered_df = self.df[mask]
            if filtered_df.empty:
                logger.warning(f"No products found matching '{product_name}'")
                return []
            # Get the positional indices for embeddings
            filtered_positions = []
            for i, (idx, row) in enumerate(self.df.iterrows()):
                if mask.iloc[i]:
                    filtered_positions.append(i)
            filtered_embeddings = self.embeddings[filtered_positions]
            logger.info(f"Filtered to {len(filtered_df)} reviews for product '{product_name}'")
        else:
            filtered_df = self.df
            filtered_positions = list(range(len(self.df)))
            filtered_embeddings = self.embeddings
            logger.info(f"Searching all {len(filtered_df)} reviews")
        
        # STEP 1: Keyword pre-filtering
        # Only consider reviews that have some keyword relevance
        keyword_filtered_indices = []
        keyword_scores = []
        
        logger.info(f"Starting keyword pre-filtering for '{query_text}'")
        for idx, (_, row) in enumerate(filtered_df.iterrows()):
            has_keywords, keyword_score = self.has_exact_keywords(row['review_text'], query_text)
            if has_keywords:
                keyword_filtered_indices.append(idx)
                keyword_scores.append(keyword_score)
        
        logger.info(f"Found {len(keyword_filtered_indices)} reviews with keyword matches")
        
        # If no keyword matches found, fall back to semantic search with higher threshold
        if not keyword_filtered_indices:
            logger.info(f"No keyword matches found for '{query_text}', falling back to semantic search")
            # Use only semantic similarity with higher threshold
            query_embedding = get_embeddings([clean_query])
            similarities = cosine_similarity(query_embedding, filtered_embeddings)[0]
            
            # Higher threshold for semantic-only search
            similarity_threshold = 0.5
            valid_indices = np.where(similarities >= similarity_threshold)[0]
            
            logger.info(f"Semantic search found {len(valid_indices)} results above threshold {similarity_threshold}")
            
            if len(valid_indices) == 0:
                logger.warning(f"No semantic matches found above threshold {similarity_threshold}")
                return []
            
            # Sort and limit
            sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
            top_indices = sorted_indices[:limit]
            
            logger.info(f"Returning {len(top_indices)} semantic-only results")
            
            # Format results
            formatted_results = []
            for idx in top_indices:
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
        
        # STEP 2: Semantic similarity on keyword-filtered results
        keyword_filtered_embeddings = filtered_embeddings[keyword_filtered_indices]
        
        logger.info(f"Calculating semantic similarities for {len(keyword_filtered_indices)} keyword-matched reviews")
        
        # Create embedding for the query (no product context)
        query_embedding = get_embeddings([clean_query])
        
        # Calculate cosine similarities
        semantic_similarities = cosine_similarity(query_embedding, keyword_filtered_embeddings)[0]
        
        # STEP 3: Combine keyword and semantic scores
        combined_scores = []
        for i, semantic_sim in enumerate(semantic_similarities):
            keyword_score = keyword_scores[i]
            
            # Weight exact matches heavily
            if keyword_score == 1.0:  # Exact phrase match
                combined_score = 0.8 + 0.2 * semantic_sim  # Minimum 80% for exact matches
            else:
                combined_score = 0.4 * keyword_score + 0.6 * semantic_sim
            
            combined_scores.append(combined_score)
        
        combined_scores = np.array(combined_scores)
        
        # Sort by combined score
        sorted_indices = np.argsort(combined_scores)[::-1]
        top_indices = sorted_indices[:limit]
        
        logger.info(f"Returning {len(top_indices)} hybrid search results")
        logger.info(f"Top score: {combined_scores[top_indices[0]]:.3f}, Bottom score: {combined_scores[top_indices[-1]]:.3f}")
        
        # Format results
        formatted_results = []
        for idx in top_indices:
            original_keyword_idx = keyword_filtered_indices[idx]
            original_position = filtered_positions[original_keyword_idx]
            row = self.df.iloc[original_position]
            combined_score = combined_scores[idx]
            
            formatted_results.append({
                'review_text': row['review_text'],
                'product_name': row['clean_product_name'],
                'source': row['source'],
                'rating': str(row.get('rating', '')),
                'title': str(row.get('title', '')),
                'reviewer': str(row.get('reviewer', '')),
                'date': str(row.get('date', '')),
                'similarity_score': float(combined_score)
            })
        
        logger.info(f"Search completed successfully for '{query_text}'")
        return formatted_results

# Initialize the search engine
search_engine = ReviewSearchEngine()

# Complex usage pattern classes removed for simplified deployment
        self.df_with_usage = None
        self.usage_loaded = False
        
    def load_usage_patterns(self, csv_file_with_patterns):
        """Load the preprocessed dataset with usage patterns"""
        try:
            print(f"Loading preprocessed usage patterns from {csv_file_with_patterns}...")
            self.df_with_usage = pd.read_csv(csv_file_with_patterns)
            self.usage_loaded = True
            print(f"Loaded {len(self.df_with_usage)} reviews with usage patterns")
            return True
        except FileNotFoundError:
            print(f"Usage patterns file not found: {csv_file_with_patterns}")
            print("Run preprocess_usage_patterns.py first to generate usage patterns")
            return False
        except Exception as e:
            print(f"Error loading usage patterns: {e}")
            return False
    
    def get_usage_paradigms(self, product_name, min_size=5):
        """Get major usage paradigms for a product (combinations of context + role)"""
        
        if not self.usage_loaded:
            return {"error": "Usage patterns not loaded. Run preprocessing first."}
        
        # Filter for product
        if product_name != "all":
            # Clean the product name for matching (same logic as main search engine)
            cleaned_product_name = self._clean_product_name_for_matching(product_name)
            product_df = self.df_with_usage[self.df_with_usage['product_name'].str.contains(
                cleaned_product_name, case=False, na=False)]
        else:
            product_df = self.df_with_usage
            
        if len(product_df) < min_size:
            return {"error": f"Not enough reviews for {product_name}"}
        
        # Find context+role combinations
        paradigm_combinations = defaultdict(list)
        
        for idx, row in product_df.iterrows():
            context = row['usage_context']
            role = row['usage_role']
            
            # Skip low-quality assignments
            if (context in ['insufficient_text', 'general_context'] and 
                role in ['insufficient_text', 'general_role']):
                continue
                
            paradigm_key = f"{context}_{role}"
            paradigm_combinations[paradigm_key].append({
                'review_text': row['review_text'],
                'rating': row.get('rating', ''),
                'context_confidence': row.get('context_confidence', 0),
                'role_confidence': row.get('role_confidence', 0)
            })
        
        # Filter paradigms by minimum size and create summaries
        paradigms = {}
        paradigm_id = 1
        
        for paradigm_key, reviews in paradigm_combinations.items():
            if len(reviews) >= min_size:
                context, role = paradigm_key.split('_', 1)
                
                paradigms[f"paradigm_{paradigm_id}"] = {
                    'name': self._generate_paradigm_name(context, role),
                    'context': context,
                    'role': role,
                    'size': len(reviews),
                    'percentage': len(reviews) / len(product_df) * 100,
                    'description': self._generate_paradigm_description(context, role),
                    'avg_rating': np.mean([float(r['rating']) for r in reviews 
                                         if r['rating'] and str(r['rating']).replace('.','').isdigit()]),
                    'sample_reviews': [r['review_text'] for r in reviews[:3]],
                    'all_reviews': [r['review_text'] for r in reviews]
                }
                paradigm_id += 1
        
        # Sort by size
        sorted_paradigms = dict(sorted(paradigms.items(), 
                                     key=lambda x: x[1]['size'], reverse=True))
        
        return {
            'product': product_name,
            'total_reviews': len(product_df),
            'paradigms': sorted_paradigms,
            'summary': self._generate_paradigm_summary(sorted_paradigms, product_name)
        }
    
    def _generate_paradigm_name(self, context, role):
        """Generate human-readable name for context+role combination"""
        
        # Simplified, more intuitive context names
        context_names = {
            'daily_routine': 'Daily Use',
            'shower_routine': 'Shower Time', 
            'pre_makeup': 'Before Makeup',
            'special_events': 'Special Occasions',
            'pre_workout': 'Before Exercise',
            'travel': 'Travel/On-the-Go',
            'sensitive_periods': 'Skin Sensitivity'
        }
        
        # More descriptive role names
        role_names = {
            'replacement': 'Main Product',
            'consolidator': 'Multi-Purpose',
            'prep_enhancer': 'Prep Step',
            'treatment': 'Treatment'
        }
        
        context_display = context_names.get(context, context.replace('_', ' ').title())
        role_display = role_names.get(role, role.replace('_', ' ').title())
        
        return f"{context_display} - {role_display}"
    
    def _generate_paradigm_description(self, context, role):
        """Generate description for context+role combination"""
        
        context_desc = {
            'daily_routine': 'as part of daily skincare routine',
            'shower_routine': 'during shower or bath time',
            'pre_makeup': 'before applying makeup',
            'special_events': 'for special occasions or events',
            'pre_workout': 'before exercise or physical activity',
            'travel': 'during travel or on-the-go situations',
            'sensitive_periods': 'when skin is sensitive or reactive'
        }
        
        role_desc = {
            'replacement': 'as the main product for this purpose',
            'consolidator': 'to handle multiple skincare needs',
            'prep_enhancer': 'to prepare skin for other products',
            'treatment': 'to address specific skin concerns'
        }
        
        context_text = context_desc.get(context, context.replace('_', ' '))
        role_text = role_desc.get(role, role.replace('_', ' '))
        
        return f"Used {context_text} {role_text}"
    
    def _clean_product_name_for_matching(self, product_name):
        """Clean product name for matching (same logic as main search engine)"""
        if not product_name:
            return ""
        
        # Convert to lowercase and remove extra whitespace
        cleaned = str(product_name).lower().strip()
        
        # Remove common prefixes/suffixes
        cleaned = re.sub(r'^dermalogica\s+', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned
    
    def _generate_paradigm_summary(self, paradigms, product_name):
        """Generate executive summary"""
        if not paradigms:
            return f"No major usage paradigms found for {product_name}"
        
        summary_lines = [
            f"Found {len(paradigms)} major usage paradigms for {product_name}:",
            ""
        ]
        
        for i, (paradigm_id, paradigm) in enumerate(paradigms.items(), 1):
            percentage = paradigm['percentage']
            summary_lines.append(
                f"{i}. {paradigm['name']} ({percentage:.0f}% of users)"
            )
            summary_lines.append(f"   → {paradigm['description']}")
            
            if i >= 3:  # Limit to top 3 for summary
                break
        
        return "\n".join(summary_lines)

# Initialize the usage pattern query engine (add after search_engine initialization)
usage_query_engine = UsagePatternQueryEngine()

class DivergentDiscoveryResults:
    def __init__(self):
        self.results = None
        self.meta_occasions = None
        self.loaded = False
    
    def load_results(self, results_file="v5_occasion_discovery_results.json", search_engine=None):
        """Load discovery results from file and clean product names"""
        try:
            if Path(results_file).exists():
                with open(results_file, 'r') as f:
                    self.results = json.load(f)
                
                # Clean product names in outlier cohorts if search_engine is provided
                if search_engine and 'outlier_cohorts' in self.results:
                    for cohort in self.results['outlier_cohorts']:
                        if 'product_name' in cohort and cohort['product_name']:
                            cohort['clean_product_name'] = search_engine.clean_product_name(cohort['product_name'])
                        
                        # Also clean product names in reviews if they exist
                        if 'reviews' in cohort:
                            for review in cohort['reviews']:
                                if 'product_name' in review and review['product_name']:
                                    review['clean_product_name'] = search_engine.clean_product_name(review['product_name'])
                
                self.loaded = True
                
                # Also try to load meta-occasions if available
                self.load_meta_occasions()
                
                return True
            return False
        except Exception as e:
            print(f"Error loading discovery results: {e}")
            return False
    
    def load_meta_occasions(self, meta_file="enhanced_meta_occasions.json"):
        """Load enhanced meta-occasion consolidation results"""
        try:
            # Try enhanced version first, fallback to original
            enhanced_file = "enhanced_meta_occasions.json"
            original_file = "meta_occasions_consolidated.json"
            
            if Path(enhanced_file).exists():
                with open(enhanced_file, 'r') as f:
                    self.meta_occasions = json.load(f)
                print(f"Loaded {len(self.meta_occasions['meta_occasions'])} enhanced meta-occasions")
                return True
            elif Path(original_file).exists():
                with open(original_file, 'r') as f:
                    self.meta_occasions = json.load(f)
                print(f"Loaded {len(self.meta_occasions['meta_occasions'])} meta-occasions")
                return True
            return False
        except Exception as e:
            print(f"Error loading meta-occasions: {e}")
            return False
    
    def get_summary(self):
        """Get summary of discovery results"""
        if not self.loaded or not self.results:
            return None
        return self.results.get('summary', '')
    
    def get_cross_product_patterns(self):
        """Get cross-product patterns"""
        if not self.loaded or not self.results:
            return []
        
        patterns = []
        for cluster_id, info in self.results.get('cluster_interpretations', {}).items():
            if info.get('is_cross_product', False):
                patterns.append({
                    'cluster_id': cluster_id,
                    'interpretation': info.get('interpretation', ''),
                    'size': info.get('cluster_size', 0),
                    'percentage': info.get('percentage', 0),
                    'products': info.get('products', []),
                    'key_terms': info.get('key_terms', []),
                    'sample_reviews': info.get('sample_reviews', [])
                })
        
        return sorted(patterns, key=lambda x: x['size'], reverse=True)
    
    def get_single_product_patterns(self):
        """Get single-product patterns"""
        if not self.loaded or not self.results:
            return []
        
        patterns = []
        for cluster_id, info in self.results.get('cluster_interpretations', {}).items():
            if not info.get('is_cross_product', False):
                patterns.append({
                    'cluster_id': cluster_id,
                    'interpretation': info.get('interpretation', ''),
                    'size': info.get('cluster_size', 0),
                    'percentage': info.get('percentage', 0),
                    'products': info.get('products', []),
                    'key_terms': info.get('key_terms', []),
                    'sample_reviews': info.get('sample_reviews', [])
                })
        
        return sorted(patterns, key=lambda x: x['size'], reverse=True)
    
    def get_outliers(self):
        """Get outlier analysis"""
        if not self.loaded or not self.results:
            return []
        return self.results.get('outlier_cohorts', [])
    
    def get_meta_occasions(self):
        """Get meta-occasion consolidation results"""
        if not self.meta_occasions:
            return []
        return self.meta_occasions.get('meta_occasions', [])
    
    def get_minority_clusters(self):
        """Get minority micro-cluster analysis"""
        if not self.loaded or not self.results:
            return []
        return self.results.get('minority_microclusters', [])
    
    def get_dataset_summary(self):
        """Get dataset summary"""
        if not self.loaded or not self.results:
            return None
        return self.results.get('summary', {})

# Initialize the discovery results loader
discovery_results = DivergentDiscoveryResults()

@app.on_event("startup")
async def startup_event():
    """Initialize the database on startup"""
    search_engine.load_data("dermalogica_aggregated_reviews.csv")
    search_engine.create_vector_database()
    
    # Try to load usage patterns if available
    usage_query_engine.load_usage_patterns("dermalogica_aggregated_reviews_with_usage_patterns.csv")
    # Try to load divergent discovery results
    discovery_results.load_results(search_engine=search_engine)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, product: str = None):
    """Serve the main search page"""
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "products": search_engine.products,
        "selected_product": product
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
    total_pages = (total_results + per_page - 1) // per_page
    
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

# Complex endpoints removed for simplified deployment

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)