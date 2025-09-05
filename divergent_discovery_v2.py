#!/usr/bin/env python3
"""
Divergent Discovery Blueprint Implementation
Finds genuinely unusual usage patterns in Dermalogica reviews using proper outlier detection
and minority micro-cluster analysis.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import umap
import re
import json
import hashlib
from collections import Counter, defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ProperDivergentDiscovery:
    def __init__(self, embeddings_model='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(embeddings_model)
        self.df = None
        self.embeddings = None
        self.umap_embeddings = None
        self.clusterer = None
        self.clusters = None
        self.outlier_scores = None
        self.cluster_summaries = {}
        
    def ingest_and_normalize(self, csv_file, min_tokens=8, max_sample=None):
        """Step 1: Proper data ingestion with deduplication and normalization"""
        print("="*60)
        print("STEP 1: INGESTING & NORMALIZING DATA")
        print("="*60)
        
        # Load data
        print(f"Loading data from {csv_file}...")
        self.df = pd.read_csv(csv_file)
        print(f"Initial dataset: {len(self.df):,} reviews")
        
        # Normalize columns
        required_cols = ['review_text', 'product_name']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Clean and prepare text
        self.df = self.df.dropna(subset=['review_text', 'product_name'])
        self.df['review_text'] = self.df['review_text'].astype(str)
        self.df['clean_text'] = self.df['review_text'].apply(self._clean_text)
        
        # Clean product names using same logic as main application
        self.df['product_name'] = self.df['product_name'].apply(self._clean_product_name)
        
        # Filter by token count
        self.df['token_count'] = self.df['clean_text'].apply(lambda x: len(x.split()))
        initial_count = len(self.df)
        self.df = self.df[self.df['token_count'] >= min_tokens]
        print(f"Filtered out {initial_count - len(self.df):,} reviews with < {min_tokens} tokens")
        
        # Deduplication
        # Check if using pre-deduplicated file
        if "deduplicated" in csv_file:
            print("Using pre-deduplicated dataset - skipping deduplication step")
            # Just clean the text
            self.df['clean_text'] = self.df['review_text'].apply(self._clean_text)
        else:
            print("Deduplicating reviews...")
            self.df = self._deduplicate_reviews()
        
        # Sample if needed
        if max_sample and len(self.df) > max_sample:
            print(f"Sampling {max_sample:,} reviews for analysis...")
            self.df = self._stratified_sample(max_sample)
        
        print(f"Final normalized dataset: {len(self.df):,} reviews")
        print(f"Products: {self.df['product_name'].nunique()}")
        if 'source' in self.df.columns:
            print(f"Sources: {self.df['source'].value_counts().to_dict()}")
        
        return self.df
    
    def _clean_text(self, text):
        """Minimal text cleaning for embeddings"""
        if pd.isna(text):
            return ""
        
        # Remove URLs and email addresses
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\[Link\]\([^)]+\)', '', text)  # Remove Amazon link artifacts
        
        # Basic cleanup
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip().lower()
        
        return text
    
    def _clean_product_name(self, name):
        """Extract clean product name from messy names (same logic as vector_search_app.py)"""
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
        
        return clean if clean and len(clean) > 2 else name_str.lower()
    
    def _deduplicate_reviews(self):
        """Proper deduplication using text hashes"""
        initial_count = len(self.df)
        
        # Create text hash for exact duplicates
        self.df['text_hash'] = self.df['clean_text'].apply(
            lambda x: hashlib.md5(x.encode()).hexdigest()
        )
        
        # Remove exact duplicates
        self.df = self.df.drop_duplicates(subset=['text_hash'])
        exact_dupes_removed = initial_count - len(self.df)
        
        # Near-duplicate removal using Jaccard similarity within products
        near_dupes_removed = 0
        unique_reviews = []
        
        for product in self.df['product_name'].unique():
            product_df = self.df[self.df['product_name'] == product].copy()
            
            if len(product_df) <= 1:
                unique_reviews.append(product_df)
                continue
            
            # Simple near-duplicate detection
            keep_indices = []
            for i, (idx1, row1) in enumerate(product_df.iterrows()):
                is_unique = True
                for j in keep_indices:
                    row2 = product_df.loc[j]
                    similarity = self._jaccard_similarity(row1['clean_text'], row2['clean_text'])
                    if similarity > 0.85:  # 85% similarity threshold
                        is_unique = False
                        break
                
                if is_unique:
                    keep_indices.append(idx1)
            
            unique_reviews.append(product_df.loc[keep_indices])
            near_dupes_removed += len(product_df) - len(keep_indices)
        
        self.df = pd.concat(unique_reviews, ignore_index=True)
        
        print(f"Removed {exact_dupes_removed:,} exact duplicates")
        print(f"Removed {near_dupes_removed:,} near-duplicates")
        
        return self.df
    
    def _jaccard_similarity(self, text1, text2):
        """Calculate Jaccard similarity between two texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0
    
    def _stratified_sample(self, sample_size):
        """Stratified sampling by product to maintain diversity"""
        sampled_dfs = []
        for product in self.df['product_name'].unique():
            product_df = self.df[self.df['product_name'] == product]
            sample_n = max(1, int(sample_size * len(product_df) / len(self.df)))
            sample_n = min(sample_n, len(product_df))
            sampled_dfs.append(product_df.sample(sample_n, random_state=42))
        
        return pd.concat(sampled_dfs, ignore_index=True)
    
    def create_semantic_embeddings(self):
        """Step 2: Generate semantic embeddings"""
        print("\n" + "="*60)
        print("STEP 2: CREATING SEMANTIC EMBEDDINGS")
        print("="*60)
        
        # Prepare texts (title + text if available)
        texts = []
        for _, row in self.df.iterrows():
            text = row['clean_text']
            if 'title' in self.df.columns and pd.notna(row['title']):
                text = f"{row['title']} {text}"
            texts.append(text)
        
        print(f"Generating embeddings for {len(texts):,} reviews...")
        self.embeddings = self.model.encode(
            texts, 
            show_progress_bar=True,
            batch_size=500,
            convert_to_numpy=True
        )
        
        print(f"Created embeddings: {self.embeddings.shape}")
        memory_mb = self.embeddings.nbytes / (1024 * 1024)
        print(f"Memory usage: {memory_mb:.1f} MB")
        
        return self.embeddings
    
    def apply_umap_structure(self, n_neighbors=50, min_dist=0.1, n_components=10):
        """Step 3: UMAP for better cluster structure"""
        print("\n" + "="*60)
        print("STEP 3: APPLYING UMAP FOR STRUCTURE")
        print("="*60)
        
        print(f"UMAP parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}")
        
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric='cosine',
            random_state=42,
            verbose=True
        )
        
        self.umap_embeddings = reducer.fit_transform(self.embeddings)
        print(f"UMAP reduction: {self.embeddings.shape[1]} -> {self.umap_embeddings.shape[1]} dimensions")
        
        return self.umap_embeddings
    
    def cluster_with_hdbscan(self, min_cluster_size=100, min_samples=25):
        """Step 4: HDBSCAN clustering with proper parameters"""
        print("\n" + "="*60)
        print("STEP 4: CLUSTERING WITH HDBSCAN")
        print("="*60)
        
        # Adjust parameters based on dataset size
        dataset_size = len(self.df)
        adjusted_min_cluster_size = max(min_cluster_size, int(dataset_size * 0.01))
        adjusted_min_samples = max(min_samples, int(adjusted_min_cluster_size * 0.25))
        
        print(f"Dataset size: {dataset_size:,}")
        print(f"Using min_cluster_size={adjusted_min_cluster_size}, min_samples={adjusted_min_samples}")
        
        self.clusterer = HDBSCAN(
            min_cluster_size=adjusted_min_cluster_size,
            min_samples=adjusted_min_samples,
            cluster_selection_method='eom',
            metric='euclidean' if self.umap_embeddings is not None else 'cosine'
        )
        
        # Use UMAP embeddings if available, otherwise raw embeddings
        embeddings_to_use = self.umap_embeddings if self.umap_embeddings is not None else self.embeddings
        
        self.clusters = self.clusterer.fit_predict(embeddings_to_use)
        
        # Calculate outlier scores manually for noise points
        self.outlier_scores = np.zeros(len(self.clusters))
        if hasattr(self.clusterer, 'outlier_scores_'):
            self.outlier_scores = self.clusterer.outlier_scores_
        else:
            # Manual outlier scoring based on distance to nearest cluster
            noise_mask = self.clusters == -1
            if np.any(noise_mask):
                self.outlier_scores[noise_mask] = self._calculate_manual_outlier_scores(embeddings_to_use[noise_mask])
        
        # Analysis
        n_clusters = len(set(self.clusters)) - (1 if -1 in self.clusters else 0)
        n_noise = list(self.clusters).count(-1)
        clustered_pct = (len(self.clusters) - n_noise) / len(self.clusters) * 100
        
        print(f"Found {n_clusters} clusters")
        print(f"Noise points: {n_noise:,} ({n_noise/len(self.clusters)*100:.1f}%)")
        print(f"Coverage: {clustered_pct:.1f}% of reviews clustered")
        
        # Add to dataframe
        self.df['cluster'] = self.clusters
        self.df['outlier_score'] = self.outlier_scores
        
        return self.clusters
    
    def _calculate_manual_outlier_scores(self, noise_embeddings):
        """Calculate outlier scores manually for noise points"""
        if len(noise_embeddings) == 0:
            return np.array([])
        
        # Get cluster centers for non-noise points
        cluster_centers = {}
        for cluster_id in set(self.clusters):
            if cluster_id != -1:
                cluster_mask = self.clusters == cluster_id
                embeddings_to_use = self.umap_embeddings if self.umap_embeddings is not None else self.embeddings
                cluster_embeddings = embeddings_to_use[cluster_mask]
                cluster_centers[cluster_id] = np.mean(cluster_embeddings, axis=0)
        
        if not cluster_centers:
            return np.ones(len(noise_embeddings))  # All equally outliers if no clusters
        
        # Calculate distance to nearest cluster center for each noise point
        outlier_scores = []
        for noise_point in noise_embeddings:
            min_distance = float('inf')
            for center in cluster_centers.values():
                distance = np.linalg.norm(noise_point - center)
                min_distance = min(min_distance, distance)
            outlier_scores.append(min_distance)
        
        # Normalize scores
        outlier_scores = np.array(outlier_scores)
        if len(outlier_scores) > 1:
            outlier_scores = (outlier_scores - np.min(outlier_scores)) / (np.max(outlier_scores) - np.min(outlier_scores))
        
        return outlier_scores
    
    def identify_divergent_patterns(self, top_outlier_pct=0.05, micro_cluster_threshold=0.005):
        """Step 5: Identify divergent material (noise outliers + minority micro-clusters)"""
        print("\n" + "="*60)
        print("STEP 5: IDENTIFYING DIVERGENT PATTERNS")
        print("="*60)
        
        divergent_patterns = {
            'noise_outliers': [],
            'minority_microclusters': [],
            'summary': {}
        }
        
        # A. Noise Outliers (HDBSCAN label -1 with high outlier scores)
        noise_df = self.df[self.df['cluster'] == -1].copy()
        if len(noise_df) > 0:
            # Get top outliers by score
            n_top_outliers = max(1, int(len(self.df) * top_outlier_pct))
            top_outliers = noise_df.nlargest(n_top_outliers, 'outlier_score')
            
            print(f"Noise outliers: {len(noise_df):,} total, analyzing top {len(top_outliers):,}")
            
            # Analyze outlier patterns
            outlier_analysis = self._analyze_outlier_content(top_outliers)
            divergent_patterns['noise_outliers'] = {
                'count': len(top_outliers),
                'percentage': len(top_outliers) / len(self.df) * 100,
                'analysis': outlier_analysis,
                'sample_reviews': top_outliers['clean_text'].head(5).tolist(),
                'outlier_scores': top_outliers['outlier_score'].tolist()
            }
        
        # B. Minority Micro-clusters
        cluster_sizes = Counter(self.clusters)
        micro_cluster_threshold_size = max(10, int(len(self.df) * micro_cluster_threshold))
        
        micro_clusters = []
        for cluster_id, size in cluster_sizes.items():
            if cluster_id != -1 and size <= micro_cluster_threshold_size:
                micro_clusters.append((cluster_id, size))
        
        print(f"Minority micro-clusters: {len(micro_clusters)} clusters with ≤{micro_cluster_threshold_size} reviews")
        
        # Analyze each micro-cluster
        micro_cluster_analysis = []
        for cluster_id, size in sorted(micro_clusters, key=lambda x: x[1], reverse=True):
            cluster_df = self.df[self.df['cluster'] == cluster_id]
            
            # Calculate distinctiveness using c-TF-IDF
            distinctiveness_score = self._calculate_cluster_distinctiveness(cluster_df)
            coherence_score = self._calculate_cluster_coherence(cluster_df)
            
            analysis = {
                'cluster_id': cluster_id,
                'size': size,
                'percentage': size / len(self.df) * 100,
                'distinctiveness_score': distinctiveness_score,
                'coherence_score': coherence_score,
                'divergence_score': self._calculate_divergence_score(size, distinctiveness_score, coherence_score),
                'distinctive_terms': self._get_distinctive_terms(cluster_df),
                'sample_reviews': cluster_df['clean_text'].head(3).tolist(),
                'product_distribution': cluster_df['product_name'].value_counts().to_dict(),
                'interpretation': self._interpret_cluster(cluster_df)
            }
            
            micro_cluster_analysis.append(analysis)
        
        # Sort by divergence score
        micro_cluster_analysis.sort(key=lambda x: x['divergence_score'], reverse=True)
        divergent_patterns['minority_microclusters'] = micro_cluster_analysis
        
        # Summary
        divergent_patterns['summary'] = {
            'total_reviews': len(self.df),
            'clustered_reviews': len(self.df) - list(self.clusters).count(-1),
            'noise_outliers': len(divergent_patterns['noise_outliers']) if divergent_patterns['noise_outliers'] else 0,
            'minority_microclusters': len(micro_cluster_analysis),
            'coverage_pct': (len(self.df) - list(self.clusters).count(-1)) / len(self.df) * 100
        }
        
        return divergent_patterns
    
    def _analyze_outlier_content(self, outlier_df):
        """Analyze outlier content for meaningful patterns"""
        analysis = {
            'languages': self._detect_languages(outlier_df),
            'themes': self._detect_themes(outlier_df),
            'product_distribution': outlier_df['product_name'].value_counts().to_dict(),
            'distinctive_vocabulary': self._get_distinctive_terms(outlier_df),
            'cohorts': self._create_outlier_cohorts(outlier_df)
        }
        return analysis
    
    def _detect_languages(self, df):
        """Simple language detection"""
        spanish_indicators = ['excelente', 'producto', 'muy', 'bueno', 'encanta', 'piel', 'como', 'queda']
        english_indicators = ['love', 'great', 'amazing', 'perfect', 'excellent', 'wonderful']
        
        spanish_count = 0
        english_count = 0
        
        for text in df['clean_text']:
            if any(word in text.lower() for word in spanish_indicators):
                spanish_count += 1
            elif any(word in text.lower() for word in english_indicators):
                english_count += 1
        
        return {
            'spanish_indicators': spanish_count,
            'english_indicators': english_count,
            'other': len(df) - spanish_count - english_count
        }
    
    def _detect_themes(self, df):
        """Detect thematic patterns in text"""
        themes = {
            'sensitive_skin': ['sensitive', 'delicate', 'gentle', 'mild', 'irritation', 'reactive'],
            'unusual_usage': ['scalp', 'eyebrows', 'razor', 'shave', 'mix', 'dilute', 'body'],
            'skin_conditions': ['eczema', 'rosacea', 'keratosis', 'fungal', 'maskne', 'perioral'],
            'travel_portable': ['travel', 'portable', 'small', 'compact', 'trip', 'vacation'],
            'professional_use': ['esthetician', 'salon', 'professional', 'dermatologist']
        }
        
        theme_counts = {}
        for theme, keywords in themes.items():
            count = 0
            for text in df['clean_text']:
                if any(keyword in text.lower() for keyword in keywords):
                    count += 1
            theme_counts[theme] = count
        
        return theme_counts
    
    def _create_outlier_cohorts(self, outlier_df):
        """Create cohorts from outlier reviews using sub-clustering"""
        if len(outlier_df) < 10:  # Need minimum reviews for meaningful cohorts
            return []
        
        print(f"Creating cohorts from {len(outlier_df)} outlier reviews...")
        
        # Get embeddings for outlier reviews
        outlier_indices = outlier_df.index.tolist()
        outlier_embeddings = self.embeddings[outlier_indices]
        
        # Apply UMAP for outlier sub-clustering (more aggressive parameters)
        outlier_umap = umap.UMAP(
            n_neighbors=min(15, len(outlier_df)//3),
            min_dist=0.0,
            n_components=5,
            metric='cosine',
            random_state=42
        )
        
        outlier_reduced = outlier_umap.fit_transform(outlier_embeddings)
        
        # Sub-cluster outliers with more permissive HDBSCAN
        outlier_clusterer = HDBSCAN(
            min_cluster_size=max(5, len(outlier_df)//15),  # Smaller clusters
            min_samples=3,
            cluster_selection_method='leaf',  # More granular
            metric='euclidean'
        )
        
        outlier_clusters = outlier_clusterer.fit_predict(outlier_reduced)
        
        # Analyze each cohort
        cohorts = []
        unique_clusters = [c for c in set(outlier_clusters) if c != -1]
        
        print(f"Found {len(unique_clusters)} outlier cohorts")
        
        for cohort_id in unique_clusters:
            cohort_mask = outlier_clusters == cohort_id
            cohort_df = outlier_df.iloc[cohort_mask].copy()
            
            if len(cohort_df) < 3:  # Skip very small cohorts
                continue
            
            # Analyze cohort
            cohort_analysis = {
                'cohort_id': f"outlier_cohort_{cohort_id}",
                'size': len(cohort_df),
                'percentage': len(cohort_df) / len(outlier_df) * 100,
                'interpretation': self._interpret_outlier_cohort(cohort_df),
                'distinctive_terms': self._get_distinctive_terms(cohort_df, top_n=8),
                'themes': self._detect_themes(cohort_df),
                'languages': self._detect_languages(cohort_df),
                'product_focus': cohort_df['product_name'].value_counts().head(3).to_dict(),
                'sample_reviews': cohort_df['clean_text'].head(3).tolist(),
                'all_reviews': cohort_df['clean_text'].tolist(),  # Store all reviews for modal
                'all_reviews_with_products': cohort_df[['clean_text', 'product_name', 'source']].to_dict('records'),  # Include product info (using cleaned names)
                'business_insight': self._generate_cohort_business_insight(cohort_df)
            }
            
            cohorts.append(cohort_analysis)
        
        # Sort by size descending
        cohorts.sort(key=lambda x: x['size'], reverse=True)
        
        return cohorts
    
    def _interpret_outlier_cohort(self, cohort_df):
        """Generate interpretation for an outlier cohort"""
        distinctive_terms = self._get_distinctive_terms(cohort_df, top_n=5)
        themes = self._detect_themes(cohort_df)
        languages = self._detect_languages(cohort_df)
        
        term_string = ' '.join(distinctive_terms).lower()
        
        # Language-based interpretation
        if languages['spanish_indicators'] > languages['english_indicators']:
            return f"Spanish-Language Users ({', '.join(distinctive_terms[:3])})"
        
        # Theme-based interpretation
        max_theme = max(themes.items(), key=lambda x: x[1]) if themes else ('general', 0)
        theme_name, theme_count = max_theme
        
        if theme_count >= len(cohort_df) * 0.3:  # At least 30% of cohort
            theme_interpretations = {
                'sensitive_skin': 'Sensitive Skin Focus',
                'unusual_usage': 'Creative Off-Label Usage',
                'skin_conditions': 'Medical/Therapeutic Applications',
                'travel_portable': 'Travel & Portability Focused',
                'professional_use': 'Professional/Salon Usage'
            }
            
            base_interpretation = theme_interpretations.get(theme_name, 'Specialized Usage')
            
            # Add distinctive terms context
            if distinctive_terms:
                return f"{base_interpretation} ({', '.join(distinctive_terms[:2])})"
            else:
                return base_interpretation
        
        # Product-focused interpretation
        top_products = cohort_df['product_name'].value_counts()
        if len(top_products) <= 2 and top_products.iloc[0] >= len(cohort_df) * 0.6:
            product_name = top_products.index[0].replace('_', ' ').title()
            return f"{product_name} Specialists"
        
        # Fallback to distinctive terms
        if distinctive_terms:
            return f"Distinctive Pattern: {', '.join(distinctive_terms[:3])}"
        else:
            return f"Outlier Cohort (Size: {len(cohort_df)})"
    
    def _generate_cohort_business_insight(self, cohort_df):
        """Generate business insights for a cohort"""
        themes = self._detect_themes(cohort_df)
        languages = self._detect_languages(cohort_df)
        products = cohort_df['product_name'].value_counts()
        
        insights = []
        
        # Language insights
        if languages['spanish_indicators'] > 5:
            insights.append("International market opportunity in Spanish-speaking regions")
        
        # Theme insights
        if themes.get('sensitive_skin', 0) >= len(cohort_df) * 0.3:
            insights.append("Potential for sensitive skin product line or marketing")
        
        if themes.get('unusual_usage', 0) >= 3:
            insights.append("Consider expanding usage instructions or marketing creative applications")
        
        if themes.get('skin_conditions', 0) >= 3:
            insights.append("Medical/dermatological partnership opportunities")
        
        if themes.get('travel_portable', 0) >= 5:
            insights.append("Travel-size product line opportunity")
        
        if themes.get('professional_use', 0) >= 3:
            insights.append("Professional/salon channel expansion potential")
        
        # Product concentration insights
        if len(products) <= 2 and products.iloc[0] >= len(cohort_df) * 0.7:
            top_product = products.index[0].replace('_', ' ').title()
            insights.append(f"Consider specialized marketing or formulation for {top_product}")
        
        return insights[:3] if insights else ["Niche usage pattern requiring further investigation"]
    
    def _calculate_cluster_distinctiveness(self, cluster_df):
        """Calculate how distinctive a cluster's vocabulary is using c-TF-IDF approach"""
        if len(cluster_df) < 2:
            return 0.0
        
        # Get cluster text and rest of corpus text
        cluster_text = ' '.join(cluster_df['clean_text'].tolist())
        rest_df = self.df[~self.df.index.isin(cluster_df.index)]
        rest_text = ' '.join(rest_df['clean_text'].sample(min(1000, len(rest_df)), random_state=42).tolist())
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=500,
                ngram_range=(1, 3),
                stop_words='english',
                min_df=2
            )
            
            tfidf_matrix = vectorizer.fit_transform([cluster_text, rest_text])
            feature_names = vectorizer.get_feature_names_out()
            
            cluster_scores = tfidf_matrix[0].toarray()[0]
            rest_scores = tfidf_matrix[1].toarray()[0]
            
            # Calculate lift scores
            lift_scores = []
            for i, (cluster_score, rest_score) in enumerate(zip(cluster_scores, rest_scores)):
                if rest_score > 0:
                    lift = cluster_score / rest_score
                    if lift > 1.5:  # Only consider terms that are 50% more frequent
                        lift_scores.append(lift)
            
            return np.mean(lift_scores) if lift_scores else 0.0
            
        except:
            return 0.0
    
    def _calculate_cluster_coherence(self, cluster_df):
        """Calculate internal coherence of a cluster"""
        if len(cluster_df) < 2:
            return 0.0
        
        # Get embeddings for cluster reviews
        cluster_indices = cluster_df.index.tolist()
        cluster_embeddings = self.embeddings[cluster_indices]
        
        # Calculate average pairwise similarity
        similarities = []
        for i in range(len(cluster_embeddings)):
            for j in range(i + 1, len(cluster_embeddings)):
                sim = cosine_similarity([cluster_embeddings[i]], [cluster_embeddings[j]])[0][0]
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_divergence_score(self, size, distinctiveness, coherence):
        """Combine size, distinctiveness, and coherence into a single divergence score"""
        # Normalize size (smaller clusters are more interesting for divergence)
        size_score = 1.0 / (1.0 + np.log(size))
        
        # Combine with weights
        divergence_score = (0.4 * distinctiveness + 0.4 * coherence + 0.2 * size_score)
        
        return divergence_score
    
    def _get_distinctive_terms(self, cluster_df, top_n=10):
        """Get most distinctive terms for a cluster"""
        if len(cluster_df) < 2:
            return []
        
        cluster_text = ' '.join(cluster_df['clean_text'].tolist())
        rest_df = self.df[~self.df.index.isin(cluster_df.index)]
        rest_text = ' '.join(rest_df['clean_text'].sample(min(1000, len(rest_df)), random_state=42).tolist())
        
        try:
            # Custom stop words to filter out technical artifacts
            custom_stop_words = [
                'amazon', 'com', 'gp', 'www', 'https', 'http', 'link', 'customer', 'reviews',
                'review', 'stars', 'out', 'dermalogica', 'product', 'use', 'good', 'like'
            ]
            
            vectorizer = TfidfVectorizer(
                max_features=200,
                ngram_range=(1, 3),
                stop_words='english',
                min_df=1
            )
            
            tfidf_matrix = vectorizer.fit_transform([cluster_text, rest_text])
            feature_names = vectorizer.get_feature_names_out()
            
            cluster_scores = tfidf_matrix[0].toarray()[0]
            rest_scores = tfidf_matrix[1].toarray()[0]
            
            # Calculate lift and filter
            distinctive_terms = []
            for i, (cluster_score, rest_score) in enumerate(zip(cluster_scores, rest_scores)):
                term = feature_names[i]
                if (term not in custom_stop_words and 
                    len(term) > 2 and 
                    not term.isdigit() and
                    not any(tech in term for tech in ['amazon', 'com', 'gp', 'www', 'http'])):
                    
                    if rest_score > 0:
                        lift = cluster_score / rest_score
                        if lift > 1.2:
                            distinctive_terms.append((term, lift))
                    elif cluster_score > 0:
                        distinctive_terms.append((term, float('inf')))
            
            distinctive_terms.sort(key=lambda x: x[1], reverse=True)
            return [term for term, _ in distinctive_terms[:top_n]]
            
        except:
            return []
    
    def _interpret_cluster(self, cluster_df):
        """Generate human interpretation of cluster"""
        distinctive_terms = self._get_distinctive_terms(cluster_df, top_n=5)
        term_string = ' '.join(distinctive_terms).lower()
        
        # Pattern matching for interpretations
        if any(word in term_string for word in ['scalp', 'hair', 'head']):
            return "Off-label scalp/hair usage"
        elif any(word in term_string for word in ['eyebrow', 'brow', 'eye']):
            return "Eyebrow/eye area application"
        elif any(word in term_string for word in ['razor', 'shave', 'after']):
            return "Post-shaving treatment"
        elif any(word in term_string for word in ['mix', 'dilute', 'combine']):
            return "Product mixing/dilution"
        elif any(word in term_string for word in ['eczema', 'rosacea', 'dermatitis']):
            return "Specific skin condition treatment"
        elif any(word in term_string for word in ['travel', 'portable', 'small']):
            return "Travel/portability focus"
        elif any(word in term_string for word in ['professional', 'salon', 'esthetician']):
            return "Professional/salon usage"
        elif any(word in term_string for word in ['body', 'arm', 'leg']):
            return "Body application (off-label)"
        else:
            return f"Distinctive usage pattern: {', '.join(distinctive_terms[:3])}"
    
    def run_full_discovery(self, csv_file, max_sample=15000):
        """Run the complete divergent discovery pipeline"""
        print("="*80)
        print("PROPER DIVERGENT DISCOVERY ANALYSIS")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Step 1: Ingest & normalize
        self.ingest_and_normalize(csv_file, max_sample=max_sample)
        
        # Step 2: Create embeddings
        self.create_semantic_embeddings()
        
        # Step 3: Apply UMAP
        self.apply_umap_structure()
        
        # Step 4: Cluster with HDBSCAN
        self.cluster_with_hdbscan()
        
        # Step 5: Identify divergent patterns
        results = self.identify_divergent_patterns()
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results):
        """Save results to JSON file"""
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)
        
        try:
            # Convert numpy types for JSON serialization
            json_results = self._convert_for_json(results)
            
            with open('proper_divergent_discovery_results.json', 'w') as f:
                json.dump(json_results, f, indent=2)
            
            print("Results saved to proper_divergent_discovery_results.json")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def _convert_for_json(self, obj):
        """Convert numpy types to JSON-serializable types"""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

if __name__ == "__main__":
    # Run proper divergent discovery
    discovery = ProperDivergentDiscovery()
    
    results = discovery.run_full_discovery(
        "dermalogica_aggregated_reviews.csv",
        max_sample=15000  # Reasonable sample size for analysis
    )
    
    print("\n" + "="*80)
    print("DISCOVERY COMPLETE - KEY FINDINGS")
    print("="*80)
    
    summary = results['summary']
    print(f"Dataset: {summary['total_reviews']:,} reviews analyzed")
    print(f"Coverage: {summary['coverage_pct']:.1f}% successfully clustered")
    print(f"Noise outliers found: {summary['noise_outliers']}")
    print(f"Minority micro-clusters: {summary['minority_microclusters']}")
    
    # Show top divergent patterns
    if results['minority_microclusters']:
        print(f"\nTOP DIVERGENT MICRO-CLUSTERS:")
        for i, cluster in enumerate(results['minority_microclusters'][:5], 1):
            print(f"{i}. {cluster['interpretation']} ({cluster['size']} reviews, {cluster['percentage']:.2f}%)")
            print(f"   Distinctiveness: {cluster['distinctiveness_score']:.3f}")
            print(f"   Key terms: {', '.join(cluster['distinctive_terms'][:5])}")
    
    if results['noise_outliers']:
        print(f"\nNOISE OUTLIERS ANALYSIS:")
        outliers = results['noise_outliers']
        print(f"Count: {outliers['count']} ({outliers['percentage']:.2f}%)")
        if 'analysis' in outliers:
            themes = outliers['analysis'].get('themes', {})
            print(f"Key themes: {dict(list(themes.items())[:3])}")
