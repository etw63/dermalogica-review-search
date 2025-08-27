#!/usr/bin/env python3
import pandas as pd

# Load the CSV file
df = pd.read_csv('dermalogica_aggregated_reviews.csv')

print("Column names:", list(df.columns))
print("\nFirst few rows:")
print(df[['product_name', 'original_product_name']].head())

print("\nUnique product_name values (first 10):")
unique_products = df['product_name'].unique()
clean_products = sorted([p for p in unique_products 
                        if p != 'Dermalogica Product' and p.strip() and len(p.strip()) > 0])

for i, p in enumerate(clean_products[:10]):
    print(f"  {i+1}. {p}")

print(f"\nTotal clean products: {len(clean_products)}")

print("\nUnique original_product_name values (first 10):")
original_products = df['original_product_name'].unique()
clean_original = sorted([p for p in original_products 
                        if p != 'Dermalogica Product' and str(p).strip() and len(str(p).strip()) > 0])

for i, p in enumerate(clean_original[:10]):
    print(f"  {i+1}. {p}")

print(f"\nTotal original products: {len(clean_original)}")
