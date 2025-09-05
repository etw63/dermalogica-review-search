#!/usr/bin/env python3
"""
Deduplicate Reviews Script

This script performs the same deduplication logic as the divergent discovery script
but saves the cleaned dataset to avoid repeating this expensive operation.

Usage: python deduplicate_reviews.py
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHashLSH, MinHash
import re
import hashlib
from tqdm import tqdm
import json
from datetime import datetime

class ReviewDeduplicator:
    def __init__(self, input_file="dermalogica_aggregated_reviews.csv", 
                 output_file="dermalogica_deduplicated_reviews.csv"):
        self.input_file = input_file
        self.output_file = output_file
        self.min_tokens = 8
        self.similarity_threshold = 0.85
        self.stats = {}
        
    def clean_text(self, text):
        """Clean and normalize text for analysis"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower().strip()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def count_tokens(self, text):
        """Count meaningful tokens in text"""
        if not text:
            return 0
        return len([word for word in text.split() if len(word) > 1])
    
    def create_text_hash(self, text):
        """Create hash for exact duplicate detection"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def create_minhash(self, text, num_perm=128):
        """Create MinHash for near-duplicate detection"""
        minhash = MinHash(num_perm=num_perm)
        words = text.split()
        for word in words:
            minhash.update(word.encode('utf8'))
        return minhash
    
    def deduplicate_reviews(self):
        """Main deduplication process"""
        print("=" * 80)
        print("REVIEW DEDUPLICATION PROCESS")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Load data
        print(f"Loading data from {self.input_file}...")
        df = pd.read_csv(self.input_file)
        print(f"Initial dataset: {len(df):,} reviews")
        self.stats['initial_count'] = len(df)
        
        # Clean text
        print("Cleaning and preprocessing text...")
        df['clean_text'] = df['review_text'].apply(self.clean_text)
        df['token_count'] = df['clean_text'].apply(self.count_tokens)
        
        # Filter short reviews
        print(f"Filtering reviews with < {self.min_tokens} tokens...")
        before_filter = len(df)
        df = df[df['token_count'] >= self.min_tokens].copy()
        filtered_count = before_filter - len(df)
        print(f"Filtered out {filtered_count:,} reviews with < {self.min_tokens} tokens")
        self.stats['short_reviews_removed'] = filtered_count
        self.stats['after_length_filter'] = len(df)
        
        # Remove exact duplicates
        print("Removing exact duplicates...")
        df['text_hash'] = df['clean_text'].apply(self.create_text_hash)
        before_exact = len(df)
        df = df.drop_duplicates(subset=['text_hash'], keep='first')
        exact_dupes = before_exact - len(df)
        print(f"Removed {exact_dupes:,} exact duplicates")
        self.stats['exact_duplicates_removed'] = exact_dupes
        self.stats['after_exact_dedup'] = len(df)
        
        # Remove near-duplicates using MinHash LSH
        print("Detecting near-duplicates using MinHash LSH...")
        lsh = MinHashLSH(threshold=self.similarity_threshold, num_perm=128)
        
        # Create MinHashes and add to LSH
        minhashes = {}
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating MinHashes"):
            minhash = self.create_minhash(row['clean_text'])
            minhashes[idx] = minhash
            lsh.insert(idx, minhash)
        
        # Find near-duplicates
        near_duplicates = set()
        for idx in tqdm(df.index, desc="Finding near-duplicates"):
            if idx not in near_duplicates:
                similar_items = lsh.query(minhashes[idx])
                if len(similar_items) > 1:
                    # Keep the first one, mark others as duplicates
                    similar_items = list(similar_items)
                    similar_items.remove(idx)  # Remove self
                    near_duplicates.update(similar_items)
        
        # Remove near-duplicates
        before_near = len(df)
        df = df[~df.index.isin(near_duplicates)]
        near_dupes = before_near - len(df)
        print(f"Removed {near_dupes:,} near-duplicates")
        self.stats['near_duplicates_removed'] = near_dupes
        self.stats['final_count'] = len(df)
        
        # Calculate final statistics
        print("\n" + "=" * 60)
        print("DEDUPLICATION SUMMARY")
        print("=" * 60)
        print(f"Initial reviews: {self.stats['initial_count']:,}")
        print(f"Short reviews removed: {self.stats['short_reviews_removed']:,}")
        print(f"Exact duplicates removed: {self.stats['exact_duplicates_removed']:,}")
        print(f"Near duplicates removed: {self.stats['near_duplicates_removed']:,}")
        print(f"Final clean dataset: {self.stats['final_count']:,}")
        print(f"Retention rate: {self.stats['final_count']/self.stats['initial_count']*100:.1f}%")
        
        # Analyze sources and products
        print(f"\nFinal dataset composition:")
        if 'source' in df.columns:
            source_dist = df['source'].value_counts()
            print("Sources:", dict(source_dist))
        
        if 'product_name' in df.columns:
            print(f"Products: {df['product_name'].nunique()}")
        
        # Drop helper columns before saving
        columns_to_drop = ['clean_text', 'token_count', 'text_hash']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        # Save deduplicated dataset
        print(f"\nSaving deduplicated dataset to {self.output_file}...")
        df.to_csv(self.output_file, index=False)
        print(f"Saved {len(df):,} deduplicated reviews")
        
        # Save statistics
        stats_file = "deduplication_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"Deduplication statistics saved to {stats_file}")
        
        return df

def main():
    deduplicator = ReviewDeduplicator()
    deduplicated_df = deduplicator.deduplicate_reviews()
    print("\nDeduplication complete!")

if __name__ == "__main__":
    main()
