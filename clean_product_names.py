#!/usr/bin/env python3
"""
Clean Product Names Script

This script applies the same product name cleaning logic used in the main application
to ensure consistency between the search engine and the discovery analysis.
"""

import pandas as pd
import re

def clean_product_name(name):
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

def main():
    print("Loading deduplicated reviews...")
    df = pd.read_csv('dermalogica_deduplicated_reviews.csv')
    
    print(f"Original dataset: {len(df):,} reviews")
    print(f"Unique product names: {df['product_name'].nunique()}")
    
    # Show some examples of current product names
    print("\nCurrent product name examples:")
    current_names = df['product_name'].unique()[:5]
    for name in current_names:
        print(f"  - {name}")
    
    # Apply cleaning
    print("\nApplying product name cleaning...")
    df['cleaned_product_name'] = df['product_name'].apply(clean_product_name)
    
    # Show cleaned examples
    print("\nCleaned product name examples:")
    for i, name in enumerate(current_names):
        cleaned = clean_product_name(name)
        print(f"  - {name} → {cleaned}")
    
    # Replace the product_name column with cleaned names
    df['product_name'] = df['cleaned_product_name']
    df = df.drop('cleaned_product_name', axis=1)
    
    print(f"\nAfter cleaning:")
    print(f"Unique product names: {df['product_name'].nunique()}")
    
    # Save the cleaned dataset
    output_file = 'dermalogica_deduplicated_reviews_clean.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSaved cleaned dataset to {output_file}")
    
    print("\nFinal product name examples:")
    final_names = df['product_name'].unique()[:10]
    for name in final_names:
        print(f"  - {name}")

if __name__ == "__main__":
    main()
