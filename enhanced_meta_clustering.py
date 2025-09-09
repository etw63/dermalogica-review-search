#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Meta-Occasion Clustering Framework
Generates business-ready cohort descriptions with clean product names and review access
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

class EnhancedMetaOccasionConsolidator:
    def __init__(self, json_path: str, main_dataset_path: str = "dermalogica_aggregated_reviews.csv", min_cohort_size: int = 3):
        """
        Initialize the enhanced consolidator with business-ready output
        
        Args:
            json_path: Path to the JSON file with occasion cohorts
            main_dataset_path: Path to the main dataset with cleaned product names
            min_cohort_size: Minimum reviews to consider a cohort valid
        """
        self.min_cohort_size = min_cohort_size
        self.data = self._load_data(json_path)
        self.model = SentenceTransformer('all-mpnet-base-v2')
        
        # Load the main dataset to get the cleaned product names mapping
        self.product_name_mapping = self._load_product_name_mapping(main_dataset_path)
        
    def _load_data(self, json_path: str) -> Dict:
        """Load and validate JSON data"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data['outlier_cohorts'])} cohorts")
        return data
    
    def _load_product_name_mapping(self, csv_path: str) -> Dict[str, str]:
        """Load the main dataset and create a mapping from original to cleaned product names"""
        try:
            df = pd.read_csv(csv_path)
            
            # Create mapping from original product_name to clean_product_name
            mapping = {}
            if 'product_name' in df.columns and 'clean_product_name' in df.columns:
                unique_mappings = df[['product_name', 'clean_product_name']].drop_duplicates()
                mapping = dict(zip(unique_mappings['product_name'], unique_mappings['clean_product_name']))
                print(f"Loaded {len(mapping)} product name mappings")
            else:
                print("Warning: Could not find product_name and clean_product_name columns in main dataset")
            
            return mapping
        except Exception as e:
            print(f"Warning: Could not load product name mapping: {e}")
            return {}
    
    def clean_product_name(self, name):
        """Extract clean product name from messy names - EXACT COPY from main app"""
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
        
        if len(clean) < 3:
            return None
            
        return clean
    
    def filter_meaningful_cohorts(self) -> List[Dict]:
        """
        Filter cohorts to keep only meaningful ones with business value
        """
        filtered_cohorts = []
        
        for cohort in self.data['outlier_cohorts']:
            # Size filter
            if cohort['size'] < self.min_cohort_size:
                continue
                
            # Interpretation quality filter
            interpretation = cohort['interpretation'].lower().strip()
            
            # Skip very short or meaningless interpretations
            if len(interpretation) < 5:
                continue
                
            # Skip pure fragments without context
            if len(interpretation.split()) < 2:
                continue
                
            # Skip interpretations that are just product names
            if interpretation.startswith('dermalogica') and len(interpretation.split()) < 3:
                continue
                
            filtered_cohorts.append(cohort)
        
        print(f"Filtered to {len(filtered_cohorts)} meaningful cohorts from {len(self.data['outlier_cohorts'])}")
        return filtered_cohorts
    
    def extract_semantic_features(self, cohorts: List[Dict]) -> np.ndarray:
        """
        Extract semantic features for clustering using multiple signals
        """
        feature_texts = []
        
        for cohort in cohorts:
            # Combine interpretation with sample reviews for better context
            text_parts = [cohort['interpretation']]
            
            # Add sample reviews if available
            if 'sample_reviews' in cohort and cohort['sample_reviews']:
                text_parts.extend(cohort['sample_reviews'][:2])
            
            # Add distinctive terms if available
            if 'distinctive_terms' in cohort and cohort['distinctive_terms']:
                text_parts.append(' '.join(cohort['distinctive_terms'][:3]))
            
            combined_text = ' '.join(text_parts)
            
            # Clean the text
            combined_text = re.sub(r'\[link\]\(.*?\)', '', combined_text)
            combined_text = re.sub(r'\s+', ' ', combined_text).strip()
            
            feature_texts.append(combined_text)
        
        # Generate embeddings
        embeddings = self.model.encode(feature_texts, show_progress_bar=True)
        return embeddings
    
    def cluster_into_meta_occasions(self, cohorts: List[Dict], embeddings: np.ndarray, 
                                   n_clusters: int = 12) -> Dict[int, List[int]]:
        """
        Cluster cohorts into meta-occasions using hierarchical clustering
        """
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
    
    def generate_business_description(self, cluster_cohorts: List[Dict]) -> str:
        """
        Generate a meaningful business description for a meta-cohort
        """
        # Collect all interpretations and reviews
        all_text = []
        for cohort in cluster_cohorts:
            all_text.append(cohort['interpretation'])
            if 'sample_reviews' in cohort:
                all_text.extend(cohort.get('sample_reviews', [])[:2])
        
        combined_text = ' '.join(all_text).lower()
        
        # Define comprehensive usage occasions based on business document
        business_patterns = {
            # 1. Daily Gentle Exfoliation
            'exfoliat|microfoliant|milkfoliant|polish|shower.*staple|morning.*refresh|forever.*product': {
                'title': "Everyday Skin Polish Ritual",
                'description': "Built into daily showers or AM cleanses — skin feels lightly polished and reset without irritation. Trusted 'forever product' for routine balance."
            },
            
            # 2. Acne & Texture Rescue  
            'acne|breakout|pimple|blemish|texture|clear.*start|oily.*zone|bump|pore': {
                'title': "Clear & Smooth Control",
                'description': "Used daily or every other day by acne-prone and textured-skin users. Customers frame it as a must-have rescue step that prevents future breakouts and unclogs pores."
            },
            
            # 3. Recovery / Flareup Reset
            'flareup|ultracalm|barrier.*repair|travel|over.*exfoliat|eczema|reset|compromise': {
                'title': "Calm & Restore After Stress", 
                'description': "Functions as a 'reset button' — chosen when skin is compromised. Customers value the calming, non-stinging relief that restores comfort and confidence."
            },
            
            # 4. Makeup Prep / Glass Skin
            'makeup.*prep|glass.*skin|canvas|glow.*prep|foundation|smooth.*canvas|lit.*from.*within': {
                'title': "Smooth Canvas Glow Prep",
                'description': "Exfoliation and brightening serums are used hours before applying makeup — customers describe it as 'the difference between dull vs. lit-from-within skin.'"
            },
            
            # 5. Teen / First Routine
            'teen|first.*routine|clear.*start|parent.*approved|confidence.*boost|young.*user': {
                'title': "Clear Start Confidence",
                'description': "Designed for young users just learning skincare basics. Parents trust it as safe, while teens connect with the confidence boost from clearer skin."
            },
            
            # 6. Anti-Aging Ritual
            'retinol|anti.*aging|wrinkle|firm|age.*smart|forever.*young|spf.*50|vitamin.*c': {
                'title': "Forever-Young Maintenance",
                'description': "Midlife and older customers use these as anchor products in 'age-smart' routines — balancing potency (retinol, vitamin C) with barrier safety."
            },
            
            # 7. Weekend Skin Detox / Mask Moment
            'mask|weekend|detox|recovery.*masque|fizz.*mask|me.*time|sunday.*reset|indulgence': {
                'title': "Weekend Recovery or Skin SOS",
                'description': "Masks framed as a ritual of indulgence and repair — either 'me-time' or a fix after late nights/travel."
            },
            
            # 8. Sun & Environment Defense
            'spf|sun.*defense|uv|pollution|physical.*defense|environment|outdoor': {
                'title': "Sun & Environment Defense",
                'description': "Customers use these to shield against UV, pollution, and visible aging triggers. Often layered with antioxidants."
            },
            
            # 9. Post-Workout Cleanse & Refresh
            'workout|gym|exercise|sweat|post.*workout|cleanse.*sweat|gym.*bag|hot.*yoga': {
                'title': "Post-Workout Cleanse & Refresh",
                'description': "Chosen to prevent clogged pores and breakouts after workouts — convenience and freshness are key."
            },
            
            # 10. Seasonal Skin Shift (Winter Rescue)
            'winter|seasonal|dry.*month|heating|cold.*weather|rich.*repair|moisture.*balance': {
                'title': "Seasonal Skin Shift (Winter Rescue)",
                'description': "Used during seasonal skin stress. Customers describe it as 'skin survival kit' for colder months."
            },
            
            # 11. Body Confidence Boost
            'body|sculptor|firming|post.*pregnancy|summer.*shorts|confidence.*boost|tighten': {
                'title': "Body Confidence Boost",
                'description': "Firming and smoothing body treatments used when skin confidence matters — tightening arms, stomach, thighs."
            },
            
            # 12. Pre-Event Glow Prep
            'special.*occasion|date.*night|wedding|event|glow.*prep|facial.*in.*bottle': {
                'title': "Pre-Event Glow Prep",
                'description': "Customers describe it as 'like a facial in a bottle' — reserved for high-stakes days where glow and brightness matter most."
            },
            
            # 13. Active Lifestyle / Sweat-Proofing
            'precleanse|pre.*workout|running|sport|sweat.*proof|training|active.*lifestyle': {
                'title': "Active Lifestyle / Sweat-Proofing",
                'description': "Used on dry skin pre-exercise to prevent sweat and grime from clogging pores. Fits gym-goers and runners who build skincare into training."
            }
        }
        
        # Score patterns and select best match
        pattern_scores = {}
        for pattern, pattern_info in business_patterns.items():
            matches = len(re.findall(pattern, combined_text, re.IGNORECASE))
            if matches > 0:
                pattern_scores[pattern] = {
                    'score': matches,
                    'title': pattern_info['title'],
                    'description': pattern_info['description']
                }
        
        # If we have matches, return the best one
        if pattern_scores:
            best_pattern = max(pattern_scores.items(), key=lambda x: x[1]['score'])
            best_title = best_pattern[1]['title']
            best_description = best_pattern[1]['description']
            
            # Return both title and description
            return {'title': best_title, 'description': best_description}
        
        # Fallback: analyze interpretations for business insight
        if cluster_cohorts:
            interpretations = [c.get('interpretation', '') for c in cluster_cohorts if c.get('interpretation')]
            if interpretations:
                # Look for action words and contexts
                combined_interp = ' '.join(interpretations).lower()
                
                if any(word in combined_interp for word in ['routine', 'daily', 'regular']):
                    return {
                        'title': "Daily Skin Maintenance Ritual",
                        'description': "Customers who rely on consistent, daily skincare habits to maintain healthy, balanced skin over time."
                    }
                elif any(word in combined_interp for word in ['problem', 'issue', 'concern', 'fix']):
                    return {
                        'title': "Problem-Solving Skincare Mission", 
                        'description': "Used when customers need targeted solutions for specific skin concerns or unexpected issues."
                    }
                elif any(word in combined_interp for word in ['gentle', 'sensitive', 'mild']):
                    return {
                        'title': "Gentle Care for Delicate Skin",
                        'description': "Customers with sensitive skin who need mild, non-irritating products that still deliver results."
                    }
                elif any(word in combined_interp for word in ['special', 'occasion', 'event']):
                    return {
                        'title': "Special Occasion Skin Prep",
                        'description': "Used before important events when customers want their skin to look its absolute best."
                    }
                else:
                    # Use the longest, most descriptive interpretation
                    best_interp = max(interpretations, key=len)
                    if len(best_interp) > 15:
                        return {
                            'title': f"Specialized {best_interp.title()} Approach",
                            'description': f"Customers using products for specific {best_interp.lower()} needs and applications."
                        }
        
        return {
            'title': "Specialized Skincare Applications",
            'description': "Unique usage patterns and applications that don't fit standard skincare routines but serve specific customer needs."
        }
    
    def get_top_products(self, cluster_cohorts: List[Dict]) -> List[Tuple[str, int]]:
        """
        Get top products for a cluster - product_name is already clean
        """
        product_counter = Counter()
        
        for cohort in cluster_cohorts:
            # Get product from cohort data - product_name is already clean
            if 'product_name' in cohort and cohort['product_name']:
                product_name = cohort['product_name'].strip()
                if product_name and len(product_name) > 2:
                    product_counter[product_name] += cohort.get('size', 1)
            
            # Also check product_focus if available - these might need cleaning
            if 'product_focus' in cohort:
                for product, count in cohort['product_focus'].items():
                    # product_focus might have messy names, so clean them
                    clean_name = self.clean_product_name(product)
                    if clean_name:
                        product_counter[clean_name] += count
        
        return product_counter.most_common(5)
    
    def collect_all_reviews(self, cluster_cohorts: List[Dict]) -> List[Dict]:
        """
        Collect all reviews from cohorts in the cluster and standardize field names
        Filter out 1 and 2-star reviews to focus on constructive feedback
        """
        all_reviews = []
        
        for cohort in cluster_cohorts:
            if 'all_reviews_with_products' in cohort:
                # Standardize field names from cohort reviews
                for review in cohort['all_reviews_with_products']:
                    # Filter out low-rated reviews (1-2 stars)
                    rating = review.get('rating')
                    if rating is not None:
                        try:
                            rating_num = float(rating)
                            if rating_num <= 2.0:
                                continue  # Skip 1 and 2-star reviews
                        except (ValueError, TypeError):
                            pass  # Keep reviews with invalid ratings
                    
                    standardized_review = {}
                    
                    # Map different text field names to 'review_text' 
                    if 'clean_text' in review:
                        standardized_review['review_text'] = review['clean_text']
                    elif 'text' in review:
                        standardized_review['review_text'] = review['text']
                    elif 'review_text' in review:
                        standardized_review['review_text'] = review['review_text']
                    else:
                        standardized_review['review_text'] = 'No review text available'
                    
                    # Keep other important fields
                    standardized_review['product_name'] = review.get('product_name', cohort.get('product_name', ''))
                    standardized_review['source'] = review.get('source', 'unknown')
                    
                    if 'rating' in review:
                        standardized_review['rating'] = review['rating']
                        
                    all_reviews.append(standardized_review)
                    
            elif 'sample_reviews' in cohort:
                # Create review objects from sample reviews (assume these are quality reviews)
                for review_text in cohort['sample_reviews']:
                    all_reviews.append({
                        'review_text': review_text,
                        'product_name': cohort.get('product_name', ''),
                        'cohort_id': cohort.get('cohort_id', ''),
                        'source': 'sample'
                    })
        
        # Limit to manageable size while keeping diversity
        if len(all_reviews) > 100:
            import random
            random.seed(42)  # For reproducibility
            all_reviews = random.sample(all_reviews, 100)
        
        return all_reviews
    
    def create_enhanced_meta_occasions(self, cohorts: List[Dict], clusters: Dict[int, List[int]]) -> List[Dict]:
        """
        Create business-ready meta-occasions with meaningful descriptions
        """
        meta_occasions = []
        
        for cluster_id, cohort_indices in clusters.items():
            cluster_cohorts = [cohorts[i] for i in cohort_indices]
            
            # Generate business description
            business_info = self.generate_business_description(cluster_cohorts)
            
            # Get top products with clean names
            top_products = self.get_top_products(cluster_cohorts)
            
            # Calculate metrics
            total_reviews = sum(c.get('size', 0) for c in cluster_cohorts)
            
            # Collect all reviews
            all_reviews = self.collect_all_reviews(cluster_cohorts)
            
            meta_occasion = {
                'meta_cohort_id': f'meta_cohort_{cluster_id}',
                'theme': business_info['title'],
                'description': business_info['description'],
                'products': [{'name': name, 'review_count': count} for name, count in top_products],
                'percentage_of_reviews': round((total_reviews / self.data['summary']['total_reviews']) * 100, 1),
                'total_reviews': total_reviews,
                'cohort_count': len(cluster_cohorts),
                'original_cohorts': [c['cohort_id'] for c in cluster_cohorts],
                'all_reviews': all_reviews[:100],  # Limit to first 100 reviews for performance
                'sample_interpretations': [c['interpretation'] for c in cluster_cohorts[:5]]
            }
            
            meta_occasions.append(meta_occasion)
        
        # Sort by review count descending
        meta_occasions.sort(key=lambda x: x['total_reviews'], reverse=True)
        
        return meta_occasions
    
    def run_enhanced_consolidation(self, n_clusters: int = 10):
        """
        Execute the enhanced consolidation pipeline
        """
        print("=" * 80)
        print("ENHANCED META-OCCASION CONSOLIDATION PIPELINE")
        print("=" * 80)
        
        # Step 1: Filter meaningful cohorts
        print("\nStep 1: Filtering meaningful cohorts...")
        filtered_cohorts = self.filter_meaningful_cohorts()
        
        if len(filtered_cohorts) < 5:
            print("Warning: Very few cohorts remain after filtering.")
            return None
        
        # Step 2: Extract features and cluster
        print("\nStep 2: Extracting semantic features...")
        embeddings = self.extract_semantic_features(filtered_cohorts)
        
        print(f"\nStep 3: Clustering into {n_clusters} meta-occasions...")
        clusters = self.cluster_into_meta_occasions(filtered_cohorts, embeddings, n_clusters)
        
        # Step 4: Create enhanced meta-occasions
        print("\nStep 4: Creating business-ready meta-occasions...")
        meta_occasions = self.create_enhanced_meta_occasions(filtered_cohorts, clusters)
        
        # Display results
        print("\n" + "=" * 80)
        print("ENHANCED BUSINESS-READY META-OCCASIONS")
        print("=" * 80)
        
        for i, occasion in enumerate(meta_occasions[:10], 1):
            print(f"\n{i}. {occasion['theme']}")
            print(f"   Description: {occasion['description']}")
            print(f"   Products: {', '.join([p['name'] for p in occasion['products'][:3]])}")
            print(f"   Reviews: {occasion['total_reviews']} ({occasion['percentage_of_reviews']}%)")
        
        # Save enhanced results
        output_path = 'enhanced_meta_occasions.json'
        with open(output_path, 'w') as f:
            json.dump({
                'summary': {
                    'total_meta_occasions': len(meta_occasions),
                    'original_cohorts': len(self.data['outlier_cohorts']),
                    'filtered_cohorts': len(filtered_cohorts),
                    'total_reviews_covered': sum(m['total_reviews'] for m in meta_occasions),
                    'total_reviews_dataset': self.data['summary']['total_reviews']
                },
                'meta_occasions': meta_occasions
            }, f, indent=2)
        
        print(f"\n✅ Saved enhanced results to {output_path}")
        
        return meta_occasions

# Example usage
if __name__ == "__main__":
    # Initialize with your JSON file
    consolidator = EnhancedMetaOccasionConsolidator(
        json_path='v5_occasion_discovery_results.json',
        min_cohort_size=3
    )
    
    # Run enhanced consolidation (more clusters for better specificity)
    meta_occasions = consolidator.run_enhanced_consolidation(n_clusters=15)
