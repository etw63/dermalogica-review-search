#!/usr/bin/env python3
"""
Improved Divergent Discovery Analysis v3

Implements the high-impact changes for better divergent pattern detection:
1. Two-stage clustering (coarse + granular sub-clustering)
2. Better sampling strategy (stratified, not random)
3. Improved labeling focused on use cases
4. Proper outlier treatment as first-class discovery surface
5. More local UMAP settings for divergent patterns
"""

import pandas as pd
import numpy as np
import re
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

# ML imports
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA, TruncatedSVD
import umap
import hdbscan
from tqdm import tqdm

class ImprovedDivergentDiscovery:
    def __init__(self):
        self.df = None
        self.embeddings = None
        self.sentence_model = None
        self.results = {}
        
    def run_full_discovery(self, csv_file: str, max_sample: Optional[int] = None):
        """Run the complete improved divergent discovery pipeline"""
        
        print("=" * 80)
        print("IMPROVED DIVERGENT DISCOVERY ANALYSIS")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Step 1: Load and prepare data with better sampling
        print("=" * 60)
        print("STEP 1: INTELLIGENT DATA PREPARATION")
        print("=" * 60)
        self._load_and_prepare_data(csv_file, max_sample)
        
        # Step 2: Create better embeddings
        print("\n" + "=" * 60)
        print("STEP 2: ENHANCED SEMANTIC EMBEDDINGS")
        print("=" * 60)
        self._create_embeddings()
        
        # Step 3: Two-stage clustering
        print("\n" + "=" * 60)
        print("STEP 3: TWO-STAGE CLUSTERING")
        print("=" * 60)
        coarse_clusters = self._stage_a_coarse_clustering()
        fine_results = self._stage_b_granular_clustering(coarse_clusters)
        
        # Step 4: Advanced outlier analysis
        print("\n" + "=" * 60)
        print("STEP 4: FIRST-CLASS OUTLIER DISCOVERY")
        print("=" * 60)
        outlier_results = self._advanced_outlier_analysis()
        
        # Step 5: Combine and rank results
        print("\n" + "=" * 60)
        print("STEP 5: RANKING AND INTERPRETATION")
        print("=" * 60)
        final_results = self._combine_and_rank_results(fine_results, outlier_results)
        
        # Step 6: Save results
        print("\n" + "=" * 60)
        print("STEP 6: SAVING RESULTS")
        print("=" * 60)
        self._save_results(final_results)
        
        return final_results
    
    def _load_and_prepare_data(self, csv_file: str, max_sample: Optional[int]):
        """Load data with stratified sampling to preserve tails"""
        print(f"Loading data from {csv_file}...")
        df = pd.read_csv(csv_file)
        print(f"Initial dataset: {len(df):,} reviews")
        
        # Basic cleaning
        required_cols = ['review_text', 'product_name']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        df = df.dropna(subset=required_cols)
        df['review_text'] = df['review_text'].astype(str)
        df['clean_text'] = df['review_text'].apply(self._clean_text)
        
        # Clean product names
        df['product_name'] = df['product_name'].apply(self._clean_product_name)
        
        # Filter short reviews
        df['token_count'] = df['clean_text'].apply(lambda x: len(x.split()))
        initial_count = len(df)
        df = df[df['token_count'] >= 8]
        print(f"Filtered out {initial_count - len(df):,} reviews with < 8 tokens")
        
        # Stratified sampling instead of random
        if max_sample and len(df) > max_sample:
            print(f"Using stratified sampling to preserve tails...")
            df = self._stratified_sample(df, max_sample)
        
        # Use full dataset if manageable (memory check suggests we can handle 40k+)
        print(f"Final dataset: {len(df):,} reviews")
        print(f"Products: {df['product_name'].nunique()}")
        if 'source' in df.columns:
            print(f"Sources: {df['source'].value_counts().to_dict()}")
        
        self.df = df.reset_index(drop=True)
    
    def _stratified_sample(self, df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Stratified sampling to preserve tail distributions"""
        strata = []
        
        # Add robustness guards for missing columns
        df = df.copy()
        if 'rating' not in df.columns: 
            df['rating'] = np.nan
        if 'source' not in df.columns: 
            df['source'] = 'unknown'
        
        # Create strata based on source, rating, and product type
        df['rating_bucket'] = df['rating'].apply(self._bucket_rating)
        df['product_type'] = df['product_name'].apply(self._get_product_type)
        
        # Stratify by source × rating × product_type
        for source in df['source'].unique():
            for rating in df['rating_bucket'].unique():
                for prod_type in df['product_type'].unique():
                    stratum = df[
                        (df['source'] == source) & 
                        (df['rating_bucket'] == rating) & 
                        (df['product_type'] == prod_type)
                    ]
                    if len(stratum) > 0:
                        strata.append((stratum, len(stratum)))
        
        # Calculate sample sizes with oversampling for rare strata
        total_size = sum(size for _, size in strata)
        samples = []
        
        for stratum_df, stratum_size in strata:
            # Base allocation
            base_sample = max(1, int(sample_size * stratum_size / total_size))
            
            # Oversample small strata (especially low ratings and rare product types)
            is_rare = stratum_size < 50
            is_negative = stratum_df['rating_bucket'].iloc[0] == 'low'
            
            if is_rare or is_negative:
                base_sample = min(stratum_size, base_sample * 2)
            
            if len(stratum_df) >= base_sample:
                sample = stratum_df.sample(n=base_sample, random_state=42)
            else:
                sample = stratum_df
            
            samples.append(sample)
        
        result = pd.concat(samples, ignore_index=True)
        print(f"Stratified sampling: {len(result):,} reviews from {len(strata)} strata")
        
        return result
    
    def _bucket_rating(self, rating) -> str:
        """Bucket ratings for stratification"""
        if pd.isna(rating):
            return 'unknown'
        try:
            r = float(rating)
            if r <= 2:
                return 'low'
            elif r <= 3:
                return 'mid'
            else:
                return 'high'
        except:
            return 'unknown'
    
    def _get_product_type(self, product_name: str) -> str:
        """Extract product type for stratification"""
        if not product_name:
            return 'other'
        
        name = str(product_name).lower()
        if any(word in name for word in ['cleanser', 'cleansing', 'wash']):
            return 'cleanser'
        elif any(word in name for word in ['serum', 'treatment']):
            return 'serum'
        elif any(word in name for word in ['moisturizer', 'cream', 'lotion']):
            return 'moisturizer'
        elif any(word in name for word in ['spf', 'sunscreen', 'sun']):
            return 'spf'
        elif any(word in name for word in ['exfoliant', 'peel']):
            return 'exfoliant'
        else:
            return 'other'
    
    def _clean_text(self, text):
        """Clean text for embeddings (keep use-case terms)"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower().strip()
        
        # Remove URLs and emails but keep use-case language
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\[Link\]\([^)]+\)', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _clean_product_name(self, name):
        """Clean product names (same logic as main app)"""
        if not name or name == 'Dermalogica Product':
            return None
            
        name_str = str(name).strip()
        
        # If already clean, just remove brand prefix
        if not any(char in name_str for char in ['$', '★', 'out of', 'stars', 'reviews']):
            clean = name_str.lower().replace('dermalogica ', '').strip()
            if clean and len(clean) > 2:
                return clean
        
        # Extract clean name from messy format
        clean = name_str
        prefixes_to_remove = [
            'Only at Ulta ', 'ULTA BEAUTY EXCLUSIVE ', 
            'Dermalogica ', '2 sizes ', 'Mini ', 'Clear Start '
        ]
        
        for prefix in prefixes_to_remove:
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
        
        # Find end of product name
        end_markers = [' $', ' 4.', ' 5.', ' 3.', ' 2.', ' 1.', ' out of', ' Kit Price']
        min_end = len(clean)
        
        for marker in end_markers:
            pos = clean.find(marker)
            if pos > 0 and pos < min_end:
                min_end = pos
        
        clean = clean[:min_end].strip().lower()
        
        # Remove suffixes but keep them if result too short
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
        
        if len(clean) < 3:
            clean = original_clean
        
        return clean if clean and len(clean) > 2 else name_str.lower()
    
    def _create_embeddings(self):
        """Create enhanced embeddings with PCA debiasing"""
        print("Loading enhanced sentence transformer...")
        # Use a stronger model for better use-case nuances
        self.sentence_model = SentenceTransformer("all-mpnet-base-v2")
        
        print(f"Generating embeddings for {len(self.df):,} reviews...")
        embeddings = self.sentence_model.encode(
            self.df['clean_text'].tolist(),
            batch_size=32,
            show_progress_bar=True,
            convert_to_tensor=False
        )
        
        print(f"Created embeddings: {embeddings.shape}")
        print(f"Memory usage: {embeddings.nbytes / 1024 / 1024:.1f} MB")
        
        # Optional: Remove top principal component to reduce frequency bias (lighter version)
        print("Applying PCA debiasing with TruncatedSVD...")
        svd = TruncatedSVD(n_components=1, random_state=42)
        svd.fit(embeddings)
        
        # Remove top component (captures generic frequency patterns)
        top_component = svd.components_[0]
        embeddings_debiased = embeddings - np.outer(
            embeddings.dot(top_component), top_component
        )
        
        # L2-normalize embeddings for consistent cosine geometry
        emb = embeddings_debiased
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        self.embeddings = emb.astype(np.float32)
        print("Applied PCA debiasing and L2 normalization")
    
    def _stage_a_coarse_clustering(self) -> Dict:
        """Stage A: Coarse segmentation to group related themes"""
        print("Stage A: Coarse clustering for broad segmentation...")
        
        # UMAP with global structure emphasis
        print("Applying UMAP for coarse structure...")
        umap_coarse = umap.UMAP(
            n_neighbors=50,
            min_dist=0.1,
            n_components=15,
            metric='cosine',
            random_state=42,
            verbose=True
        )
        
        umap_embeddings = umap_coarse.fit_transform(self.embeddings)
        print(f"UMAP coarse reduction: {self.embeddings.shape[1]} -> {umap_embeddings.shape[1]} dimensions")
        
        # HDBSCAN with larger min_cluster_size for broad themes
        print("Clustering with HDBSCAN for broad segments...")
        clusterer_coarse = hdbscan.HDBSCAN(
            min_cluster_size=300,
            min_samples=50,
            cluster_selection_method='eom',
            metric='euclidean'
        )
        
        coarse_labels = clusterer_coarse.fit_predict(umap_embeddings)
        
        n_coarse_clusters = len(set(coarse_labels)) - (1 if -1 in coarse_labels else 0)
        n_coarse_noise = list(coarse_labels).count(-1)
        
        print(f"Found {n_coarse_clusters} coarse clusters")
        print(f"Coarse noise points: {n_coarse_noise} ({n_coarse_noise/len(coarse_labels)*100:.1f}%)")
        
        # Store coarse results
        coarse_results = {
            'labels': coarse_labels,
            'umap_embeddings': umap_embeddings,
            'clusterer': clusterer_coarse,
            'n_clusters': n_coarse_clusters,
            'noise_count': n_coarse_noise
        }
        
        return coarse_results
    
    def _stage_b_granular_clustering(self, coarse_results: Dict) -> Dict:
        """Stage B: Granular sub-clustering within each coarse segment"""
        print("Stage B: Granular sub-clustering for micro-niches...")
        
        coarse_labels = coarse_results['labels']
        fine_clusters = []
        
        # Sub-cluster each coarse cluster
        for cluster_id in set(coarse_labels):
            if cluster_id == -1:  # Skip noise for now
                continue
                
            cluster_mask = coarse_labels == cluster_id
            cluster_size = cluster_mask.sum()
            
            if cluster_size < 250:  # Skip small coarse clusters (lowered threshold)
                continue
            
            print(f"Sub-clustering coarse cluster {cluster_id} ({cluster_size} reviews)...")
            
            cluster_embeddings = self.embeddings[cluster_mask]
            cluster_df = self.df[cluster_mask].copy()
            
            # Local UMAP with tighter settings
            umap_local = umap.UMAP(
                n_neighbors=min(20, cluster_size // 10),
                min_dist=0.0,
                n_components=10,
                metric='cosine',
                random_state=42
            )
            
            local_umap = umap_local.fit_transform(cluster_embeddings)
            
            # Granular HDBSCAN
            clusterer_fine = hdbscan.HDBSCAN(
                min_cluster_size=max(30, cluster_size // 25),
                min_samples=10,
                cluster_selection_method='leaf',
                metric='euclidean'
            )
            
            fine_labels = clusterer_fine.fit_predict(local_umap)
            
            # Process fine clusters
            for fine_id in set(fine_labels):
                if fine_id == -1:
                    continue
                    
                fine_mask = fine_labels == fine_id
                fine_size = fine_mask.sum()
                
                if fine_size >= 25:  # Minimum size for micro-cluster
                    fine_cluster_df = cluster_df[fine_mask].copy()
                    
                    analysis = self._analyze_micro_cluster(fine_cluster_df)
                    analysis.update({
                        'cluster_id': f"micro_{cluster_id}_{fine_id}",
                        'parent_cluster': cluster_id,
                        'size': fine_size,
                        'percentage': fine_size / len(self.df) * 100
                    })
                    
                    fine_clusters.append(analysis)
        
        print(f"Found {len(fine_clusters)} micro-clusters")
        
        return {'micro_clusters': fine_clusters}
    
    def _advanced_outlier_analysis(self) -> Dict:
        """Advanced outlier analysis as first-class discovery surface"""
        print("Analyzing outliers as primary discovery surface...")
        
        # Sample data for faster outlier detection on large datasets
        sample_size = min(15000, len(self.embeddings))
        if len(self.embeddings) > sample_size:
            print(f"Sampling {sample_size} reviews for efficient outlier detection...")
            sample_indices = np.random.choice(len(self.embeddings), sample_size, replace=False)
            sample_embeddings = self.embeddings[sample_indices]
            sample_df = self.df.iloc[sample_indices].copy()
        else:
            sample_embeddings = self.embeddings
            sample_df = self.df.copy()
            sample_indices = np.arange(len(self.embeddings))
        
        # Use HDBSCAN to identify outliers on sample
        outlier_detector = hdbscan.HDBSCAN(
            min_cluster_size=max(50, sample_size // 300),  # Scale with sample size
            min_samples=15,
            cluster_selection_method='eom',
            metric='euclidean',  # Use euclidean on normalized embeddings (equivalent to cosine)
            prediction_data=False  # Disable for speed - we'll calculate scores manually
        )
        
        outlier_labels = outlier_detector.fit_predict(sample_embeddings)
        noise_mask = outlier_labels == -1
        
        if noise_mask.sum() == 0:
            print("No outliers detected")
            return {'outlier_cohorts': []}
        
        print(f"Found {noise_mask.sum()} outlier points in sample")
        
        # Calculate outlier scores manually (faster than prediction_data=True)
        outlier_scores = self._calculate_manual_outlier_scores(
            sample_embeddings, outlier_labels, sample_indices
        )
        
        # Take top 35% of outliers by score
        outlier_df = sample_df[noise_mask].copy()
        outlier_embeddings = sample_embeddings[noise_mask]
        scores = outlier_scores[noise_mask] if outlier_scores is not None else np.ones(noise_mask.sum())
        
        # Sort by outlier score and take top 35%
        top_pct = 0.35
        n_top_outliers = max(50, int(len(outlier_df) * top_pct))
        top_indices = np.argsort(scores)[-n_top_outliers:]
        
        top_outlier_df = outlier_df.iloc[top_indices].copy()
        top_outlier_embeddings = outlier_embeddings[top_indices]
        top_scores = scores[top_indices]
        
        print(f"Analyzing top {len(top_outlier_df)} outliers ({top_pct*100}% of noise)")
        
        # Cluster outliers to create cohorts
        cohort_results = self._create_outlier_cohorts(
            top_outlier_df, top_outlier_embeddings, top_scores
        )
        
        return cohort_results
    
    def _calculate_manual_outlier_scores(self, embeddings, labels, indices):
        """Calculate outlier scores manually for better performance"""
        scores = np.zeros(len(labels))
        noise_mask = labels == -1
        
        if noise_mask.sum() == 0:
            return scores
        
        print(f"Computing outlier scores for {noise_mask.sum()} outlier points...")
        
        # For noise points, calculate distance to nearest cluster centroid
        cluster_ids = set(labels) - {-1}
        if len(cluster_ids) == 0:
            # No clusters found, assign random scores
            scores[noise_mask] = np.random.random(noise_mask.sum())
            return scores
        
        # Calculate cluster centroids
        centroids = {}
        for cluster_id in cluster_ids:
            cluster_mask = labels == cluster_id
            if cluster_mask.sum() > 0:
                centroids[cluster_id] = embeddings[cluster_mask].mean(axis=0)
        
        if len(centroids) == 0:
            scores[noise_mask] = np.random.random(noise_mask.sum())
            return scores
        
        # Calculate outlier scores as distance to nearest centroid
        centroid_matrix = np.vstack(list(centroids.values()))
        
        for i, is_noise in enumerate(noise_mask):
            if is_noise:
                # Distance to all centroids
                distances = np.linalg.norm(embeddings[i] - centroid_matrix, axis=1)
                # Outlier score is minimum distance (higher = more outlying)
                scores[i] = np.min(distances)
        
        return scores
    
    def _calculate_outlier_scores(self, labels, clusterer):
        """Calculate outlier scores manually if not provided by HDBSCAN"""
        try:
            # Try to access outlier scores
            if hasattr(clusterer, 'outlier_scores_') and clusterer.outlier_scores_ is not None:
                return clusterer.outlier_scores_
        except:
            pass
        
        # Manual calculation based on distance to nearest cluster
        noise_mask = labels == -1
        if noise_mask.sum() == 0:
            return np.zeros(len(labels))
        
        scores = np.zeros(len(labels))
        
        # For noise points, calculate distance to nearest non-noise point
        for i in range(len(labels)):
            if labels[i] == -1:  # Noise point
                # Find nearest non-noise points
                non_noise_mask = labels != -1
                if non_noise_mask.sum() > 0:
                    distances = cosine_similarity(
                        [self.embeddings[i]], 
                        self.embeddings[non_noise_mask]
                    )[0]
                    scores[i] = 1.0 - np.max(distances)  # Higher score = more outlying
                else:
                    scores[i] = 1.0
        
        return scores
    
    def _create_outlier_cohorts(self, outlier_df: pd.DataFrame, 
                              outlier_embeddings: np.ndarray, 
                              outlier_scores: np.ndarray) -> Dict:
        """Create coherent cohorts from outliers"""
        print("Creating outlier cohorts...")
        
        # Sub-cluster outliers with very local settings
        umap_outliers = umap.UMAP(
            n_neighbors=min(15, len(outlier_df) // 3),
            min_dist=0.0,
            n_components=8,
            metric='cosine',
            random_state=42
        )
        
        outlier_umap = umap_outliers.fit_transform(outlier_embeddings)
        
        # Tight clustering for cohorts
        cohort_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(15, len(outlier_df) // 15),
            min_samples=5,
            cluster_selection_method='leaf',
            metric='euclidean'
        )
        
        cohort_labels = cohort_clusterer.fit_predict(outlier_umap)
        
        cohorts = []
        for cohort_id in set(cohort_labels):
            if cohort_id == -1:
                continue
                
            cohort_mask = cohort_labels == cohort_id
            cohort_df = outlier_df[cohort_mask].copy()
            cohort_scores = outlier_scores[cohort_mask]
            
            if len(cohort_df) < 10:  # Skip very small cohorts
                continue
            
            # Calculate divergence score
            divergence_score = self._calculate_divergence_score(
                cohort_df, cohort_scores
            )
            
            analysis = self._analyze_outlier_cohort(cohort_df)
            analysis.update({
                'cohort_id': f"outlier_cohort_{cohort_id}",
                'size': len(cohort_df),
                'percentage': len(cohort_df) / len(self.df) * 100,
                'mean_outlier_score': np.mean(cohort_scores),
                'divergence_score': divergence_score
            })
            
            cohorts.append(analysis)
        
        # Sort by divergence score
        cohorts.sort(key=lambda x: x['divergence_score'], reverse=True)
        
        print(f"Created {len(cohorts)} outlier cohorts")
        
        return {'outlier_cohorts': cohorts}
    
    def _calculate_divergence_score(self, cohort_df: pd.DataFrame, 
                                   outlier_scores: np.ndarray) -> float:
        """Calculate divergence score combining outlier score, distinctiveness, and coherence"""
        
        # Component 1: Mean outlier score (0.5 weight)
        outlier_component = np.mean(outlier_scores)
        
        # Component 2: Distinctiveness via log-odds (0.3 weight)
        distinctiveness = self._calculate_distinctiveness(cohort_df)
        
        # Component 3: Coherence via cosine similarity (0.2 weight)
        coherence = self._calculate_coherence(cohort_df)
        
        # Combined score
        divergence_score = (
            0.5 * outlier_component +
            0.3 * distinctiveness +
            0.2 * coherence
        )
        
        return float(divergence_score)
    
    def _calculate_distinctiveness(self, cohort_df: pd.DataFrame) -> float:
        """Calculate distinctiveness using log-odds vs rest of corpus"""
        if len(cohort_df) < 5:
            return 0.0
        
        # Get distinctive terms using TF-IDF
        cohort_texts = cohort_df['clean_text'].tolist()
        rest_texts = self.df[~self.df.index.isin(cohort_df.index)]['clean_text'].tolist()
        
        if len(rest_texts) == 0:
            return 0.0
        
        # Simple distinctiveness: unique n-grams in cohort
        cohort_words = set()
        for text in cohort_texts:
            cohort_words.update(text.split())
        
        rest_words = set()
        for text in rest_texts[:1000]:  # Sample for efficiency
            rest_words.update(text.split())
        
        unique_ratio = len(cohort_words - rest_words) / max(len(cohort_words), 1)
        return min(unique_ratio, 1.0)
    
    def _calculate_coherence(self, cohort_df: pd.DataFrame) -> float:
        """Calculate internal coherence of cohort using consistent embedding space"""
        if len(cohort_df) < 2:
            return 1.0
        
        idx = cohort_df.index.values
        # Optional: subsample for efficiency
        if len(idx) > 50:
            rng = np.random.default_rng(42)
            idx = rng.choice(idx, size=50, replace=False)
        
        X = self.embeddings[idx]  # use debiased embeddings
        sims = cosine_similarity(X)
        mask = ~np.eye(sims.shape[0], dtype=bool)
        return float(sims[mask].mean())
    
    def _analyze_micro_cluster(self, cluster_df: pd.DataFrame) -> Dict:
        """Analyze micro-cluster for use-case patterns"""
        
        # Use-case focused analysis
        distinctive_terms = self._get_use_case_terms(cluster_df)
        interpretation = self._interpret_use_case_cluster(cluster_df, distinctive_terms)
        coh = self._calculate_coherence(cluster_df)
        dist = self._calculate_distinctiveness(cluster_df)  # reuse your function
        
        analysis = {
            'interpretation': interpretation,
            'distinctive_terms': distinctive_terms[:10],
            'coherence': coh,
            'distinctiveness': dist,
            'sample_reviews': cluster_df['clean_text'].head(3).tolist(),
            'all_reviews': cluster_df['clean_text'].tolist(),
            'all_reviews_with_products': cluster_df[['clean_text', 'product_name', 'source']].to_dict('records'),
            'product_focus': cluster_df['product_name'].value_counts().head(3).to_dict(),
            'source_distribution': cluster_df['source'].value_counts().to_dict() if 'source' in cluster_df.columns else {}
        }
        
        return analysis
    
    def _analyze_outlier_cohort(self, cohort_df: pd.DataFrame) -> Dict:
        """Analyze outlier cohort for divergent patterns"""
        
        distinctive_terms = self._get_use_case_terms(cohort_df)
        interpretation = self._interpret_outlier_cohort(cohort_df, distinctive_terms)
        
        analysis = {
            'interpretation': interpretation,
            'distinctive_terms': distinctive_terms[:10],
            'sample_reviews': cohort_df['clean_text'].head(3).tolist(),
            'all_reviews': cohort_df['clean_text'].tolist(),
            'all_reviews_with_products': cohort_df[['clean_text', 'product_name', 'source']].to_dict('records'),
            'product_focus': cohort_df['product_name'].value_counts().head(3).to_dict(),
            'source_distribution': cohort_df['source'].value_counts().to_dict() if 'source' in cohort_df.columns else {}
        }
        
        return analysis
    
    def _get_use_case_terms(self, cluster_df: pd.DataFrame) -> List[str]:
        """Extract use-case focused distinctive terms"""
        
        # Domain stoplist for use-case focus
        domain_stopwords = {
            'skin', 'sensitive', 'product', 'use', 'works', 'great', 'love', 
            'smell', 'fragrance', 'buy', 'bottle', 'price', 'size', 'face',
            'dermalogica', 'good', 'nice', 'really', 'well', 'time', 'day',
            'night', 'morning', 'feel', 'feels', 'look', 'looks', 'make',
            'makes', 'get', 'got', 'go', 'goes', 'come', 'comes', 'take',
            'takes', 'give', 'gives', 'put', 'puts', 'apply', 'applied'
        }
        
        # Collect all text
        cluster_text = ' '.join(cluster_df['clean_text'].tolist())
        rest_text = ' '.join(
            self.df[~self.df.index.isin(cluster_df.index)]['clean_text'].head(1000).tolist()
        )
        
        # TF-IDF with bi/tri-grams (phrases only, no stop words)
        vectorizer = TfidfVectorizer(
            ngram_range=(2, 3),          # phrases only
            max_features=40000,
            stop_words=None,             # keep function words like with/after/before
            min_df=2
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform([cluster_text, rest_text])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get cluster vs rest scores
            cluster_scores = tfidf_matrix[0].toarray()[0]
            rest_scores = tfidf_matrix[1].toarray()[0]
            
            # Calculate lift (cluster_score / rest_score)
            lifts = []
            for i, (cluster_score, rest_score) in enumerate(zip(cluster_scores, rest_scores)):
                if rest_score > 0:
                    lift = cluster_score / rest_score
                else:
                    lift = cluster_score * 10  # High lift for unique terms
                
                term = feature_names[i]
                
                # Filter out domain stopwords and focus on use-case patterns
                if (term not in domain_stopwords and 
                    len(term) > 2 and
                    not term.isdigit() and
                    ('with' in term or 'after' in term or 'before' in term or 
                     'on' in term or 'under' in term or 'over' in term or
                     'for' in term or 'during' in term or 'while' in term or
                     any(prep in term for prep in ['mix', 'combine', 'layer', 'apply']))):
                    lifts.append((term, lift))
            
            # Sort by lift and return top terms
            lifts.sort(key=lambda x: x[1], reverse=True)
            return [term for term, _ in lifts[:15]]
            
        except Exception as e:
            print(f"Error in use-case term extraction: {e}")
            return []
    
    def _interpret_use_case_cluster(self, cluster_df: pd.DataFrame, distinctive_terms: List[str]) -> str:
        """Generate interpretation focused on use cases"""
        
        if not distinctive_terms:
            return "Distinctive usage pattern"
        
        # Look for use-case patterns in top terms
        use_patterns = []
        for term in distinctive_terms[:5]:
            if any(prep in term for prep in ['with', 'after', 'before', 'on', 'under', 'over']):
                use_patterns.append(term)
        
        if use_patterns:
            return f"Usage pattern: {', '.join(use_patterns[:3])}"
        else:
            return f"Distinctive pattern: {', '.join(distinctive_terms[:3])}"
    
    def _interpret_outlier_cohort(self, cohort_df: pd.DataFrame, distinctive_terms: List[str]) -> str:
        """Generate interpretation for outlier cohort"""
        
        if not distinctive_terms:
            return "Unusual usage pattern"
        
        return f"Divergent pattern: {', '.join(distinctive_terms[:3])}"
    
    def _combine_and_rank_results(self, fine_results: Dict, outlier_results: Dict) -> Dict:
        """Combine and rank all discovered patterns"""
        
        all_patterns = []
        
        # Add micro-clusters with improved ranking
        for cluster in fine_results.get('micro_clusters', []):
            cluster['pattern_type'] = 'micro_cluster'
            # Improved ranking: reward divergence, not just size
            total_reviews = len(self.df)
            s = (0.4 * cluster.get('distinctiveness', 0) + 
                 0.4 * cluster.get('coherence', 0) + 
                 0.2 * (cluster.get('size', 0) / max(1, total_reviews)))
            cluster['rank_score'] = float(s)
            all_patterns.append(cluster)
        
        # Add outlier cohorts
        for cohort in outlier_results.get('outlier_cohorts', []):
            cohort['pattern_type'] = 'outlier_cohort'
            cohort['rank_score'] = cohort.get('divergence_score', 0)
            all_patterns.append(cohort)
        
        # Sort by rank score
        all_patterns.sort(key=lambda x: x['rank_score'], reverse=True)
        
        # Take top patterns
        top_patterns = all_patterns[:20]
        
        results = {
            'summary': {
                'total_reviews': len(self.df),
                'micro_clusters': len([p for p in top_patterns if p['pattern_type'] == 'micro_cluster']),
                'outlier_cohorts': len([p for p in top_patterns if p['pattern_type'] == 'outlier_cohort']),
                'top_patterns': len(top_patterns)
            },
            'top_patterns': top_patterns
        }
        
        return results
    
    def _save_results(self, results: Dict):
        """Save results to JSON file"""
        
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.floating, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_, np.bool8)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        results_clean = convert_types(results)
        
        output_file = "improved_divergent_discovery_results.json"
        with open(output_file, 'w') as f:
            json.dump(results_clean, f, indent=2)
        
        print(f"Results saved to {output_file}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("DISCOVERY COMPLETE - IMPROVED RESULTS")
        print("=" * 80)
        print(f"Dataset: {results['summary']['total_reviews']:,} reviews analyzed")
        print(f"Micro-clusters found: {results['summary']['micro_clusters']}")
        print(f"Outlier cohorts found: {results['summary']['outlier_cohorts']}")
        print(f"Top patterns: {results['summary']['top_patterns']}")


def main():
    discovery = ImprovedDivergentDiscovery()
    
    # Run with full dataset (memory should handle it)
    results = discovery.run_full_discovery(
        "dermalogica_aggregated_reviews.csv",
        max_sample=None  # Use full dataset
    )
    
    return results


if __name__ == "__main__":
    main()
