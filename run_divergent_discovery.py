#!/usr/bin/env python3
"""
Run unsupervised discovery on full Dermalogica dataset to find divergent usage patterns.
This analyzes ALL products together to discover cross-product and novel usage patterns.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
import umap
from sklearn.decomposition import PCA, NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import json
from datetime import datetime

class DivergentUsageDiscovery:
    def __init__(self, embeddings_model='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(embeddings_model)
        self.df = None
        self.embeddings = None
        self.reduced_embeddings = None
        self.clusters = None
        self.topics_by_cluster = None
        
    def load_and_prepare_data(self, csv_file, min_review_length=30, sample_size=None):
        """Load and prepare full dataset for discovery"""
        
        print(f"Loading data from {csv_file}...")
        self.df = pd.read_csv(csv_file)
        print(f"Initial dataset: {len(self.df)} reviews")
        
        # Clean data
        self.df = self.df.dropna(subset=['review_text', 'product_name'])
        self.df['review_text'] = self.df['review_text'].astype(str)
        
        # Filter very short reviews (likely low signal)
        self.df = self.df[self.df['review_text'].str.len() >= min_review_length]
        
        # Remove near-duplicates (keep diversity)
        print("Removing near-duplicate reviews...")
        initial_count = len(self.df)
        
        # Simple deduplication: remove exact duplicates and very similar reviews
        self.df = self.df.drop_duplicates(subset=['review_text'])
        
        # Remove reviews that are >90% similar (simple Jaccard similarity)
        self.df = self._remove_similar_reviews(similarity_threshold=0.9)
        
        print(f"Removed {initial_count - len(self.df)} similar/duplicate reviews")
        
        # Sample for computational efficiency if dataset is very large
        if sample_size and len(self.df) > sample_size:
            print(f"Sampling {sample_size} reviews for discovery (computational efficiency)")
            # Stratified sample by product to maintain diversity
            sampled_dfs = []
            for product in self.df['product_name'].unique():
                product_df = self.df[self.df['product_name'] == product]
                sample_n = max(1, int(sample_size * len(product_df) / len(self.df)))
                sampled_dfs.append(product_df.sample(min(sample_n, len(product_df)), random_state=42))
            
            self.df = pd.concat(sampled_dfs, ignore_index=True)
        
        # Product distribution
        product_counts = self.df['product_name'].value_counts()
        print(f"\nProduct distribution ({len(product_counts)} products):")
        for product, count in product_counts.head(10).items():
            print(f"  {product}: {count} reviews")
        if len(product_counts) > 10:
            print(f"  ... and {len(product_counts)-10} more products")
        
        print(f"\nFinal dataset: {len(self.df)} reviews for discovery")
        return self.df
    
    def _remove_similar_reviews(self, similarity_threshold=0.9):
        """Remove very similar reviews using simple text similarity"""
        
        def jaccard_similarity(text1, text2):
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return intersection / union if union > 0 else 0
        
        # For large datasets, only check within same product
        unique_reviews = []
        for product in self.df['product_name'].unique():
            product_reviews = self.df[self.df['product_name'] == product].copy()
            
            keep_indices = []
            for i, row1 in product_reviews.iterrows():
                is_unique = True
                for j in keep_indices:
                    row2 = product_reviews.loc[j]
                    similarity = jaccard_similarity(row1['review_text'], row2['review_text'])
                    if similarity > similarity_threshold:
                        is_unique = False
                        break
                
                if is_unique:
                    keep_indices.append(i)
            
            unique_reviews.append(product_reviews.loc[keep_indices])
        
        return pd.concat(unique_reviews, ignore_index=True)
    
    def create_semantic_embeddings(self, batch_size=500):
        """Create semantic embeddings for all reviews"""
        print("Creating semantic embeddings...")
        
        review_texts = self.df['review_text'].tolist()
        
        # Check if embeddings already exist
        embeddings_file = f"embeddings_discovery_{len(review_texts)}.pkl"
        try:
            with open(embeddings_file, 'rb') as f:
                self.embeddings = pickle.load(f)
            print(f"Loaded existing embeddings from {embeddings_file}")
            return self.embeddings
        except FileNotFoundError:
            pass
        
        # Create new embeddings
        all_embeddings = []
        for i in range(0, len(review_texts), batch_size):
            batch_texts = review_texts[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(review_texts)-1)//batch_size + 1}")
            
            batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)
        
        self.embeddings = np.vstack(all_embeddings)
        
        # Save embeddings
        with open(embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
        print(f"Saved embeddings to {embeddings_file}")
        
        print(f"Created {self.embeddings.shape[0]} embeddings of dimension {self.embeddings.shape[1]}")
        return self.embeddings
    
    def reduce_dimensionality(self, n_components=50, n_neighbors=15, min_dist=0.1):
        """Use UMAP for dimensionality reduction"""
        print("Reducing dimensionality with UMAP...")
        
        # Pre-reduce with PCA if necessary
        if self.embeddings.shape[1] > 100:
            print("Pre-reducing with PCA...")
            pca = PCA(n_components=100)
            embeddings_pca = pca.fit_transform(self.embeddings)
            explained_var = np.sum(pca.explained_variance_ratio_)
            print(f"PCA explained variance: {explained_var:.3f}")
        else:
            embeddings_pca = self.embeddings
        
        # UMAP reduction
        umap_reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='cosine',
            random_state=42
        )
        
        self.reduced_embeddings = umap_reducer.fit_transform(embeddings_pca)
        print(f"UMAP reduction: {embeddings_pca.shape[1]} -> {self.reduced_embeddings.shape[1]}")
        
        return self.reduced_embeddings
    
    def discover_natural_clusters(self, min_cluster_size=15, min_samples=5):
        """Use HDBSCAN for density-based clustering"""
        print("Discovering natural clusters with HDBSCAN...")
        
        # Adjust parameters based on dataset size
        dataset_size = len(self.reduced_embeddings)
        adjusted_min_cluster_size = max(min_cluster_size, int(dataset_size * 0.01))  # At least 1% of data
        
        print(f"Using min_cluster_size={adjusted_min_cluster_size} for dataset of size {dataset_size}")
        
        clusterer = HDBSCAN(
            min_cluster_size=adjusted_min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_epsilon=0.5
        )
        
        cluster_labels = clusterer.fit_predict(self.reduced_embeddings)
        
        # Analysis
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"Found {n_clusters} natural clusters")
        print(f"Outliers (noise points): {n_noise} ({n_noise/len(cluster_labels)*100:.1f}%)")
        
        self.clusters = cluster_labels
        self.df['cluster'] = cluster_labels
        
        # Cluster analysis
        cluster_sizes = Counter(cluster_labels)
        print("\nCluster size distribution:")
        for cluster_id, size in sorted(cluster_sizes.items()):
            percentage = size / len(cluster_labels) * 100
            if cluster_id == -1:
                print(f"  Noise: {size} reviews ({percentage:.1f}%)")
            else:
                print(f"  Cluster {cluster_id}: {size} reviews ({percentage:.1f}%)")
        
        return cluster_labels
    
    def analyze_clusters_by_product(self):
        """Analyze how clusters distribute across products - key for finding cross-product patterns"""
        print("\nAnalyzing cluster-product relationships...")
        
        cluster_product_matrix = pd.crosstab(self.df['cluster'], self.df['product_name'])
        
        cross_product_clusters = []
        single_product_clusters = []
        
        for cluster_id in cluster_product_matrix.index:
            if cluster_id == -1:  # Skip noise
                continue
                
            # Count products with >10% of cluster
            products_in_cluster = (cluster_product_matrix.loc[cluster_id] > 
                                 cluster_product_matrix.loc[cluster_id].sum() * 0.1).sum()
            
            if products_in_cluster > 1:
                cross_product_clusters.append(cluster_id)
            else:
                single_product_clusters.append(cluster_id)
        
        print(f"Cross-product clusters: {len(cross_product_clusters)} (span multiple products)")
        print(f"Single-product clusters: {len(single_product_clusters)} (product-specific)")
        
        return {
            'cross_product_clusters': cross_product_clusters,
            'single_product_clusters': single_product_clusters,
            'cluster_product_matrix': cluster_product_matrix
        }
    
    def extract_cluster_topics(self, n_topics_per_cluster=3, max_features=150):
        """Extract topics from each cluster using NMF"""
        print("Extracting topics from each cluster...")
        
        self.topics_by_cluster = {}
        unique_clusters = sorted([c for c in set(self.clusters) if c != -1])
        
        for cluster_id in unique_clusters:
            cluster_df = self.df[self.df['cluster'] == cluster_id]
            cluster_reviews = cluster_df['review_text'].tolist()
            
            if len(cluster_reviews) < 5:
                continue
            
            print(f"Processing cluster {cluster_id} ({len(cluster_reviews)} reviews)...")
            
            # Get product distribution for this cluster
            product_dist = cluster_df['product_name'].value_counts()
            is_cross_product = len(product_dist) > 1 and product_dist.iloc[1] > len(cluster_df) * 0.1
            
            try:
                # TF-IDF vectorization
                vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 3),
                    stop_words='english',
                    lowercase=True,
                    max_df=0.8,
                    min_df=2
                )
                
                tfidf_matrix = vectorizer.fit_transform(cluster_reviews)
                feature_names = vectorizer.get_feature_names_out()
                
                # NMF topic modeling
                n_topics = min(n_topics_per_cluster, len(cluster_reviews)//5, tfidf_matrix.shape[1]//10)
                if n_topics < 1:
                    n_topics = 1
                
                nmf = NMF(n_components=n_topics, random_state=42, max_iter=200)
                nmf_topics = nmf.fit_transform(tfidf_matrix)
                
                topics = []
                for topic_idx, topic in enumerate(nmf.components_):
                    top_indices = topic.argsort()[-15:][::-1]
                    top_terms = [(feature_names[i], topic[i]) for i in top_indices]
                    topics.append({
                        'topic_id': topic_idx,
                        'terms': top_terms,
                        'weight': np.sum(nmf_topics[:, topic_idx])
                    })
                
                self.topics_by_cluster[cluster_id] = {
                    'topics': topics,
                    'cluster_size': len(cluster_reviews),
                    'is_cross_product': is_cross_product,
                    'product_distribution': product_dist.to_dict(),
                    'sample_reviews': cluster_reviews[:5]
                }
                
            except Exception as e:
                print(f"Could not extract topics for cluster {cluster_id}: {e}")
                continue
        
        return self.topics_by_cluster
    
    def identify_divergent_outliers(self):
        """Analyze outliers for truly novel usage patterns"""
        print("Analyzing divergent outliers...")
        
        outlier_df = self.df[self.df['cluster'] == -1]
        
        if len(outlier_df) == 0:
            return {'n_outliers': 0, 'analysis': 'No outliers found'}
        
        print(f"Analyzing {len(outlier_df)} outlier reviews...")
        
        # Product distribution of outliers
        outlier_products = outlier_df['product_name'].value_counts()
        clustered_products = self.df[self.df['cluster'] != -1]['product_name'].value_counts()
        
        # Products over-represented in outliers
        outlier_product_rates = {}
        for product in outlier_products.index:
            outlier_rate = outlier_products[product] / len(outlier_df)
            normal_rate = clustered_products.get(product, 0) / len(self.df[self.df['cluster'] != -1])
            if normal_rate > 0:
                outlier_product_rates[product] = outlier_rate / normal_rate
        
        # Sort by over-representation
        over_represented_products = sorted(outlier_product_rates.items(), 
                                         key=lambda x: x[1], reverse=True)
        
        # Distinctive terms in outliers
        outlier_text = ' '.join(outlier_df['review_text'].tolist()[:1000])  # Limit for memory
        clustered_text = ' '.join(self.df[self.df['cluster'] != -1]['review_text'].head(2000).tolist())
        
        try:
            # Enhanced stop words to filter out technical artifacts
            custom_stop_words = {
                'amazon', 'com', 'gp', 'www', 'https', 'http', 'link', 'customer', 'reviews', 
                'review', 'stars', 'out', 'of', 'reviews', 'dollar', 'price', 'buy', 'purchase',
                'click', 'here', 'read', 'more', 'see', 'full', 'review', 'rating', 'rated',
                'helpful', 'vote', 'voted', 'verified', 'purchase', 'verified', 'purchase'
            }
            
            vectorizer = TfidfVectorizer(
                max_features=150,
                ngram_range=(1, 3),
                stop_words='english',
                lowercase=True,
                min_df=2,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform([outlier_text, clustered_text])
            feature_names = vectorizer.get_feature_names_out()
            
            outlier_scores = tfidf_matrix[0].toarray()[0]
            clustered_scores = tfidf_matrix[1].toarray()[0]
            
            distinctive_terms = []
            for i, (outlier_score, clustered_score) in enumerate(zip(outlier_scores, clustered_scores)):
                term = feature_names[i]
                
                # Filter out technical artifacts and meaningless terms
                if (outlier_score > clustered_score * 1.5 and 
                    term not in custom_stop_words and
                    len(term) > 2 and
                    not term.isdigit() and
                    not any(tech in term for tech in ['amazon', 'com', 'gp', 'www', 'http', 'link']) and
                    not term.startswith(('http', 'www', 'amazon', 'com'))):
                    
                    distinctive_terms.append((term, outlier_score - clustered_score))
            
            distinctive_terms.sort(key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            print(f"Error in distinctive term analysis: {e}")
            distinctive_terms = []
        
        # Analyze outlier content for meaningful patterns
        outlier_analysis = self._analyze_outlier_content(outlier_df)
        
        return {
            'n_outliers': len(outlier_df),
            'percentage': len(outlier_df) / len(self.df) * 100,
            'over_represented_products': over_represented_products[:10],
            'distinctive_terms': distinctive_terms[:15],
            'sample_reviews': outlier_df['review_text'].head(8).tolist(),
            'content_analysis': outlier_analysis
        }
    
    def _analyze_outlier_content(self, outlier_df):
        """Analyze outlier reviews for meaningful content patterns"""
        analysis = {
            'languages': {},
            'themes': {},
            'product_categories': {},
            'sentiment_patterns': {}
        }
        
        # Language detection (simple heuristic)
        spanish_words = ['excelente', 'producto', 'muy', 'bueno', 'me', 'encanta', 'como', 'queda', 'piel']
        english_words = ['love', 'great', 'amazing', 'perfect', 'excellent', 'wonderful', 'fantastic']
        
        spanish_count = 0
        english_count = 0
        
        for review in outlier_df['review_text']:
            review_lower = review.lower()
            if any(word in review_lower for word in spanish_words):
                spanish_count += 1
            elif any(word in review_lower for word in english_words):
                english_count += 1
        
        analysis['languages'] = {
            'spanish_indicators': spanish_count,
            'english_indicators': english_count,
            'total_reviews': len(outlier_df)
        }
        
        # Theme analysis
        themes = {
            'sensitive_skin': ['sensitive', 'delicate', 'gentle', 'mild', 'irritation'],
            'anti_aging': ['aging', 'wrinkle', 'firm', 'tighten', 'mature'],
            'acne_treatment': ['acne', 'breakout', 'blemish', 'clear', 'pimple'],
            'hydration': ['hydrate', 'moisture', 'dry', 'dehydrated', 'plump'],
            'travel_portable': ['travel', 'portable', 'small', 'compact', 'trip']
        }
        
        for theme, keywords in themes.items():
            count = 0
            for review in outlier_df['review_text']:
                if any(keyword in review.lower() for keyword in keywords):
                    count += 1
            analysis['themes'][theme] = count
        
        return analysis
    
    def generate_interpretations(self):
        """Generate human interpretations of discovered patterns"""
        print("Generating pattern interpretations...")
        
        interpretations = {}
        
        for cluster_id, cluster_info in self.topics_by_cluster.items():
            # Extract key terms across all topics
            all_terms = []
            for topic in cluster_info['topics']:
                all_terms.extend([term[0] for term in topic['terms'][:5]])
            
            # Generate interpretation
            interpretation = self._interpret_cluster(
                all_terms, 
                cluster_info['sample_reviews'],
                cluster_info['product_distribution'],
                cluster_info['is_cross_product']
            )
            
            interpretations[cluster_id] = {
                'interpretation': interpretation,
                'cluster_size': cluster_info['cluster_size'],
                'percentage': cluster_info['cluster_size'] / len(self.df) * 100,
                'is_cross_product': cluster_info['is_cross_product'],
                'products': list(cluster_info['product_distribution'].keys()),
                'key_terms': all_terms[:10],
                'sample_reviews': cluster_info['sample_reviews'][:3]
            }
        
        return interpretations
    
    def _interpret_cluster(self, terms, samples, product_dist, is_cross_product):
        """Generate human interpretation of cluster"""
        
        term_string = ' '.join(terms).lower()
        top_product = max(product_dist, key=product_dist.get)
        
        # Cross-product patterns (most interesting for divergent discovery)
        if is_cross_product:
            if any(word in term_string for word in ['travel', 'trip', 'vacation', 'hotel']):
                return "Cross-Product Travel/Portable Routine"
            elif any(word in term_string for word in ['workout', 'gym', 'sweat', 'active']):
                return "Multi-Product Fitness/Active Routine"
            elif any(word in term_string for word in ['mix', 'combine', 'together', 'layering']):
                return "Product Layering/Combination Strategy"
            elif any(word in term_string for word in ['gift', 'mom', 'mother', 'daughter']):
                return "Cross-Generational/Gift Usage"
            elif any(word in term_string for word in ['professional', 'esthetician', 'salon']):
                return "Professional/Salon Multi-Product Usage"
            else:
                return "Cross-Product Usage Pattern"
        
        # Product-specific patterns
        else:
            if any(word in term_string for word in ['sensitive', 'gentle', 'mild']):
                return f"Sensitive Skin Focus ({top_product})"
            elif any(word in term_string for word in ['acne', 'breakout', 'teen']):
                return f"Acne Treatment Focus ({top_product})"
            elif any(word in term_string for word in ['aging', 'mature', 'wrinkle']):
                return f"Anti-Aging Focus ({top_product})"
            elif any(word in term_string for word in ['daily', 'routine', 'morning', 'evening']):
                return f"Daily Routine Integration ({top_product})"
            else:
                return f"Product-Specific Pattern ({top_product})"
    
    def run_full_discovery(self, csv_file, sample_size=10000):
        """Run complete divergent discovery pipeline"""
        
        print("="*60)
        print("DIVERGENT USAGE PATTERN DISCOVERY")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Load and prepare data
        self.load_and_prepare_data(csv_file, sample_size=sample_size)
        
        # Create embeddings
        self.create_semantic_embeddings()
        
        # Reduce dimensionality
        self.reduce_dimensionality()
        
        # Discover clusters
        self.discover_natural_clusters()
        
        # Analyze cross-product patterns
        cluster_analysis = self.analyze_clusters_by_product()
        
        # Extract topics
        self.extract_cluster_topics()
        
        # Analyze outliers
        outlier_analysis = self.identify_divergent_outliers()
        
        # Generate interpretations
        interpretations = self.generate_interpretations()
        
        # Create final results
        results = {
            'dataset_summary': {
                'total_reviews': len(self.df),
                'products': self.df['product_name'].nunique(),
                'clusters_found': len(interpretations),
                'outliers': outlier_analysis['n_outliers']
            },
            'cross_product_analysis': cluster_analysis,
            'cluster_interpretations': interpretations,
            'divergent_outliers': outlier_analysis,
            'summary': self._generate_summary(interpretations, outlier_analysis)
        }
        
        return results
    
    def _generate_summary(self, interpretations, outliers):
        """Generate executive summary"""
        
        total_reviews = len(self.df)
        n_clusters = len(interpretations)
        n_products = self.df['product_name'].nunique()
        
        # Identify most interesting patterns
        cross_product_patterns = [(k, v) for k, v in interpretations.items() if v['is_cross_product']]
        single_product_patterns = [(k, v) for k, v in interpretations.items() if not v['is_cross_product']]
        
        summary = f"""
DIVERGENT DISCOVERY RESULTS:
• Dataset: {total_reviews:,} reviews across {n_products} products
• Natural patterns found: {n_clusters} clusters
• Cross-product patterns: {len(cross_product_patterns)} (most divergent)
• Product-specific patterns: {len(single_product_patterns)}
• Outlier reviews: {outliers['n_outliers']} ({outliers['percentage']:.1f}%)

CROSS-PRODUCT DISCOVERIES:"""
        
        for cluster_id, info in sorted(cross_product_patterns, key=lambda x: x[1]['cluster_size'], reverse=True):
            summary += f"\n• {info['interpretation']}: {info['cluster_size']} reviews ({info['percentage']:.1f}%)"
        
        if not cross_product_patterns:
            summary += "\n• No significant cross-product patterns found"
        
        summary += f"\n\nTOP SINGLE-PRODUCT PATTERNS:"
        
        for cluster_id, info in sorted(single_product_patterns, key=lambda x: x[1]['cluster_size'], reverse=True)[:3]:
            summary += f"\n• {info['interpretation']}: {info['cluster_size']} reviews"
        
        if outliers['distinctive_terms']:
            summary += f"\n\nDIVERGENT OUTLIER SIGNALS:"
            for term, score in outliers['distinctive_terms'][:5]:
                summary += f"\n• '{term}'"
        
        return summary.strip()

if __name__ == "__main__":
    # Run discovery on full dataset
    discovery = DivergentUsageDiscovery()
    
    # Use sample_size=None to process all data (may be slow)
    # Use sample_size=5000-10000 for faster processing
    results = discovery.run_full_discovery(
        "dermalogica_aggregated_reviews.csv",
        sample_size=8000  # Adjust based on your computational resources
    )
    
    # Print results
    print(results['summary'])
    
    print("\n" + "="*60)
    print("DETAILED CLUSTER ANALYSIS")
    print("="*60)
    
    # Sort clusters by size for reporting
    sorted_clusters = sorted(results['cluster_interpretations'].items(), 
                           key=lambda x: x[1]['cluster_size'], reverse=True)
    
    for cluster_id, info in sorted_clusters:
        print(f"\nCluster {cluster_id}: {info['interpretation']}")
        print(f"  Size: {info['cluster_size']} reviews ({info['percentage']:.1f}%)")
        print(f"  Type: {'Cross-product' if info['is_cross_product'] else 'Single-product'}")
        print(f"  Products: {', '.join(info['products'][:3])}")
        print(f"  Key terms: {', '.join(info['key_terms'][:5])}")
        print(f"  Sample: {info['sample_reviews'][0][:200]}...")
    
    if results['divergent_outliers']['sample_reviews']:
        print(f"\n" + "="*60)
        print("DIVERGENT OUTLIER EXAMPLES")
        print("="*60)
        
        for i, review in enumerate(results['divergent_outliers']['sample_reviews'][:5]):
            print(f"\n{i+1}. {review[:300]}...")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    # Save results to JSON for dashboard
    print("\nSaving results to JSON file...")
    try:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            'dataset_summary': results['dataset_summary'],
            'cross_product_analysis': {
                'cross_product_clusters': results['cross_product_analysis']['cross_product_clusters'],
                'single_product_clusters': results['cross_product_analysis']['single_product_clusters']
            },
            'cluster_interpretations': {},
                        'divergent_outliers': {
                'n_outliers': int(results['divergent_outliers']['n_outliers']),
                'percentage': float(results['divergent_outliers']['percentage']),
                'over_represented_products': results['divergent_outliers']['over_represented_products'],
                'distinctive_terms': results['divergent_outliers']['distinctive_terms'],
                'sample_reviews': results['divergent_outliers']['sample_reviews']
            },
            'summary': results['summary']
        }
        
        # Convert cluster interpretations
        for cluster_id, info in results['cluster_interpretations'].items():
            json_results['cluster_interpretations'][str(cluster_id)] = {
                'interpretation': info['interpretation'],
                'cluster_size': int(info['cluster_size']),
                'percentage': float(info['percentage']),
                'is_cross_product': bool(info['is_cross_product']),
                'products': info['products'],
                'key_terms': info['key_terms'],
                'sample_reviews': info['sample_reviews']
            }
        
        with open('divergent_discovery_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        print("Results saved to divergent_discovery_results.json")
        
    except Exception as e:
        print(f"Error saving results: {e}")
