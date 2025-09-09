#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meta-Occasion Clustering Framework
Consolidates sparse usage occasion cohorts into business-meaningful segments
"""

import json
import re
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.cluster import AgglomerativeClustering

class MetaOccasionConsolidator:
    def __init__(self, json_path: str, min_cohort_size: int = 5):
        """
        Initialize the consolidator with sparse cohort data
        
        Args:
            json_path: Path to the JSON file with occasion cohorts
            min_cohort_size: Minimum reviews to consider a cohort valid
        """
        self.min_cohort_size = min_cohort_size
        self.data = self._load_data(json_path)
        self.model = SentenceTransformer('all-mpnet-base-v2')
        
    def _load_data(self, json_path: str) -> Dict:
        """Load and validate JSON data"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data['outlier_cohorts'])} cohorts")
        return data
    
    def filter_low_signal_cohorts(self) -> List[Dict]:
        """
        Step 1: Filter out low-signal patterns
        - Remove cohorts with size < min_cohort_size
        - Remove fragment phrases without business meaning
        """
        filtered_cohorts = []
        
        # Define noise patterns to filter
        noise_patterns = [
            r'^[A-Z][a-z]+\s[a-z]+$',  # Random two-word fragments
            r'^\w+\s\w+$',  # Two-word fragments
            r'^(Is|Was|Has|The|This|That)\s\w+$',  # Fragment starters
        ]
        
        for cohort in self.data['outlier_cohorts']:
            # Size filter
            if cohort['size'] < self.min_cohort_size:
                continue
                
            # Interpretation quality filter
            interpretation = cohort['interpretation'].lower()
            
            # Skip fragments and non-meaningful phrases
            if len(interpretation.split()) < 3 and not any([
                'spf' in interpretation,
                'retinol' in interpretation,
                'acne' in interpretation,
                'dry' in interpretation,
                'oily' in interpretation,
                'sensitive' in interpretation
            ]):
                continue
                
            # Skip pure product mentions without context
            if interpretation.startswith('dermalogica') and len(interpretation.split()) < 4:
                continue
                
            filtered_cohorts.append(cohort)
        
        print(f"Filtered to {len(filtered_cohorts)} cohorts from {len(self.data['outlier_cohorts'])}")
        return filtered_cohorts
    
    def extract_semantic_features(self, cohorts: List[Dict]) -> np.ndarray:
        """
        Step 2a: Extract semantic features for clustering
        Combines interpretation, distinctive terms, and review samples
        """
        feature_texts = []
        
        for cohort in cohorts:
            # Combine multiple signals
            text_parts = [
                cohort['interpretation'],
                ' '.join(cohort.get('distinctive_terms', [])[:5]),
                ' '.join(cohort.get('sample_reviews', [])[:2])
            ]
            combined_text = ' '.join(text_parts)
            
            # Clean and normalize
            combined_text = re.sub(r'\[link\]\(.*?\)', '', combined_text)
            combined_text = re.sub(r'\s+', ' ', combined_text).strip()
            
            feature_texts.append(combined_text)
        
        # Generate embeddings
        embeddings = self.model.encode(feature_texts, show_progress_bar=True)
        return embeddings
    
    def cluster_into_meta_occasions(self, cohorts: List[Dict], embeddings: np.ndarray, 
                                   n_clusters: int = 15) -> Dict[int, List[int]]:
        """
        Step 2b: Cluster cohorts into meta-occasions using hierarchical clustering
        """
        # Use Agglomerative Clustering for more control over cluster count
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
        
        labels = clustering.fit_predict(embeddings)
        
        # Group cohort indices by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append(idx)
        
        return clusters
    
    def define_meta_occasion_themes(self, cohorts: List[Dict], clusters: Dict[int, List[int]]) -> List[Dict]:
        """
        Step 3: Define business-meaningful themes for each meta-cluster
        """
        meta_occasions = []
        
        # Business theme patterns
        theme_patterns = {
            'Daily Gentle Routine': ['daily', 'gentle', 'routine', 'morning', 'everyday'],
            'Acne & Breakout Control': ['acne', 'breakout', 'pimple', 'blemish', 'clear'],
            'Anti-Aging & Wrinkles': ['aging', 'wrinkle', 'fine line', 'mature', 'firming'],
            'Sensitive Skin Care': ['sensitive', 'redness', 'calm', 'soothing', 'irritation'],
            'Hydration & Moisture': ['hydrate', 'moisture', 'dry', 'dehydrated', 'plump'],
            'Dark Spots & Brightening': ['dark spot', 'brighten', 'pigment', 'even tone', 'glow'],
            'Sun Protection': ['spf', 'sun', 'protect', 'uv', 'sunscreen'],
            'Exfoliation & Texture': ['exfoliate', 'texture', 'smooth', 'polish', 'dead skin'],
            'Makeup Prep': ['makeup', 'primer', 'base', 'prep', 'foundation'],
            'Professional/Spa Experience': ['spa', 'professional', 'treatment', 'facial', 'esthetician'],
            'Travel & Convenience': ['travel', 'size', 'portable', 'convenient', 'kit'],
            'Recovery & Repair': ['repair', 'recovery', 'barrier', 'restore', 'heal'],
            'Oil Control': ['oil', 'greasy', 'shine', 'matte', 'sebum'],
            'Eye Care': ['eye', 'dark circle', 'puffy', 'crow feet', 'under eye'],
            'Teen/Young Adult': ['teen', 'young', 'first', 'starter', 'beginner']
        }
        
        for cluster_id, cohort_indices in clusters.items():
            cluster_cohorts = [cohorts[i] for i in cohort_indices]
            
            # Aggregate all text from cluster
            all_text = []
            all_products = Counter()
            total_size = 0
            
            for c in cluster_cohorts:
                all_text.extend([
                    c['interpretation'],
                    ' '.join(c.get('distinctive_terms', [])),
                    ' '.join(c.get('sample_reviews', [])[:3])
                ])
                
                # Track products
                for product, count in c.get('product_focus', {}).items():
                    all_products[product] += count
                
                total_size += c.get('size', 0)
            
            combined_text = ' '.join(all_text).lower()
            
            # Score against theme patterns
            theme_scores = {}
            for theme_name, keywords in theme_patterns.items():
                score = sum(1 for kw in keywords if kw in combined_text)
                theme_scores[theme_name] = score
            
            # Select best theme
            best_theme = max(theme_scores, key=theme_scores.get)
            
            # Extract key phrases using TF-IDF
            if len(cluster_cohorts) > 1:
                vectorizer = TfidfVectorizer(
                    ngram_range=(2, 3),
                    max_features=10,
                    stop_words='english'
                )
                try:
                    tfidf_matrix = vectorizer.fit_transform([combined_text])
                    key_phrases = vectorizer.get_feature_names_out()[:5]
                except:
                    key_phrases = []
            else:
                key_phrases = cluster_cohorts[0].get('distinctive_terms', [])[:5]
            
            meta_occasion = {
                'theme': best_theme,
                'cluster_id': f'meta_occasion_{cluster_id}',
                'cohort_count': len(cluster_cohorts),
                'total_reviews': total_size,
                'percentage': (total_size / self.data['summary']['total_reviews']) * 100,
                'top_products': dict(all_products.most_common(5)),
                'key_phrases': list(key_phrases),
                'original_cohorts': [c['cohort_id'] for c in cluster_cohorts],
                'sample_interpretations': [c['interpretation'] for c in cluster_cohorts[:3]]
            }
            
            meta_occasions.append(meta_occasion)
        
        # Sort by size
        meta_occasions.sort(key=lambda x: x['total_reviews'], reverse=True)
        
        return meta_occasions
    
    def generate_business_report(self, meta_occasions: List[Dict]) -> pd.DataFrame:
        """
        Step 4-5: Generate business-ready report with strategic insights
        """
        report_data = []
        
        for occasion in meta_occasions:
            # Calculate business metrics
            market_size = occasion['percentage']
            
            # Determine strategic priority
            if market_size > 5:
                priority = 'HIGH'
            elif market_size > 2:
                priority = 'MEDIUM'
            else:
                priority = 'LOW'
            
            # Format top products
            top_products_str = ', '.join([
                f"{p} ({c})" for p, c in list(occasion['top_products'].items())[:3]
            ])
            
            report_data.append({
                'Theme': occasion['theme'],
                'Market Size (%)': f"{market_size:.2f}%",
                'Review Count': occasion['total_reviews'],
                'Priority': priority,
                'Top Products': top_products_str,
                'Key Behaviors': ', '.join(occasion['key_phrases'][:3]),
                'Cohorts Consolidated': occasion['cohort_count']
            })
        
        df = pd.DataFrame(report_data)
        return df
    
    def run_consolidation(self, n_clusters: int = 15):
        """
        Execute the full consolidation pipeline
        """
        print("=" * 80)
        print("META-OCCASION CONSOLIDATION PIPELINE")
        print("=" * 80)
        
        # Step 1: Filter
        print("\nStep 1: Filtering low-signal cohorts...")
        filtered_cohorts = self.filter_low_signal_cohorts()
        
        if len(filtered_cohorts) < 10:
            print("Warning: Very few cohorts remain after filtering. Consider lowering min_cohort_size.")
            return None
        
        # Step 2: Extract features and cluster
        print("\nStep 2: Extracting semantic features...")
        embeddings = self.extract_semantic_features(filtered_cohorts)
        
        print(f"\nStep 3: Clustering into {n_clusters} meta-occasions...")
        clusters = self.cluster_into_meta_occasions(filtered_cohorts, embeddings, n_clusters)
        
        # Step 3: Define themes
        print("\nStep 4: Defining business themes...")
        meta_occasions = self.define_meta_occasion_themes(filtered_cohorts, clusters)
        
        # Step 4-5: Generate report
        print("\nStep 5: Generating business report...")
        report_df = self.generate_business_report(meta_occasions)
        
        # Display results
        print("\n" + "=" * 80)
        print("BUSINESS-READY OCCASION SEGMENTS")
        print("=" * 80)
        print(report_df.to_string(index=False))
        
        # Save outputs
        output_path = 'meta_occasions_consolidated.json'
        with open(output_path, 'w') as f:
            json.dump({
                'summary': {
                    'total_meta_occasions': len(meta_occasions),
                    'original_cohorts': len(self.data['outlier_cohorts']),
                    'filtered_cohorts': len(filtered_cohorts),
                    'total_reviews_covered': sum(m['total_reviews'] for m in meta_occasions)
                },
                'meta_occasions': meta_occasions
            }, f, indent=2)
        
        print(f"\n✅ Saved detailed results to {output_path}")
        
        # Save CSV report
        csv_path = 'meta_occasions_report.csv'
        report_df.to_csv(csv_path, index=False)
        print(f"📊 Saved business report to {csv_path}")
        
        return meta_occasions, report_df

# Example usage
if __name__ == "__main__":
    # Initialize with your JSON file
    consolidator = MetaOccasionConsolidator(
        json_path='v5_occasion_discovery_results short.json',
        min_cohort_size=3  # Lower threshold due to sparse data
    )
    
    # Run consolidation
    meta_occasions, report = consolidator.run_consolidation(n_clusters=12)
    
    # Additional analysis: Find gaps
    if meta_occasions:
        print("\n" + "=" * 80)
        print("STRATEGIC OPPORTUNITIES")
        print("=" * 80)
        
        # Identify underserved segments
        low_coverage = [m for m in meta_occasions if m['percentage'] < 2]
        if low_coverage:
            print("\n🎯 Underserved segments (potential growth opportunities):")
            for seg in low_coverage[:3]:
                print(f"  - {seg['theme']}: Only {seg['percentage']:.2f}% market share")
        
        # Identify product gaps
        print("\n📦 Product-occasion mismatches:")
        for occasion in meta_occasions[:5]:
            if len(occasion['top_products']) < 3:
                print(f"  - {occasion['theme']}: Only {len(occasion['top_products'])} products serving this need")
