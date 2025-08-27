#!/usr/bin/env python3
"""
Test script for the Dermalogica Review Search API
"""

import requests
import time
import json

def wait_for_server(url="http://localhost:8000", max_wait=300):
    """Wait for the server to be ready"""
    print("Waiting for server to be ready...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("✅ Server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print("⏳ Server still starting up...")
        time.sleep(10)
    
    print("❌ Server failed to start within timeout")
    return False

def test_products_api():
    """Test the products API endpoint"""
    print("\n🔍 Testing products API...")
    try:
        response = requests.get("http://localhost:8000/products")
        if response.status_code == 200:
            products = response.json()["products"]
            print(f"✅ Found {len(products)} products")
            print(f"📝 First 5 products: {products[:5]}")
            return products
        else:
            print(f"❌ Products API failed: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Error testing products API: {e}")
        return []

def test_search_api(product="precleanse", query="active lifestyle"):
    """Test the search API endpoint"""
    print(f"\n🔍 Testing search API with product='{product}' and query='{query}'...")
    try:
        params = {
            "product": product,
            "query": query,
            "limit": 5
        }
        response = requests.get("http://localhost:8000/api/search", params=params)
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Found {results['count']} matching reviews")
            
            for i, review in enumerate(results['results'][:3], 1):
                print(f"\n📄 Review {i}:")
                print(f"   Product: {review['product_name']}")
                print(f"   Source: {review['source']}")
                print(f"   Rating: {review['rating']}")
                print(f"   Similarity: {review['similarity_score']:.3f}")
                print(f"   Text: {review['review_text'][:150]}...")
            
            return results
        else:
            print(f"❌ Search API failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error testing search API: {e}")
        return None

def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 DERMALOGICA REVIEW SEARCH - API TEST")
    print("=" * 60)
    
    # Wait for server to be ready
    if not wait_for_server():
        return
    
    # Test products API
    products = test_products_api()
    
    # Test search API with different queries
    test_cases = [
        ("precleanse", "active lifestyle"),
        ("daily microfoliant", "sensitive skin"),
        ("special cleansing gel", "oily skin"),
        ("all", "anti-aging")
    ]
    
    for product, query in test_cases:
        test_search_api(product, query)
    
    print(f"\n✅ All tests completed!")
    print(f"🌐 Open your browser to http://localhost:8000 to use the web interface")

if __name__ == "__main__":
    main()
