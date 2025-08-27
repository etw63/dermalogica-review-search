#!/bin/bash

# Deployment script for Dermalogica Review Search

echo "🚀 Preparing for deployment..."

# Check if data files exist
if [ ! -f "dermalogica_aggregated_reviews.csv" ]; then
    echo "❌ Error: dermalogica_aggregated_reviews.csv not found!"
    echo "Please run the aggregator first: node aggregator.js"
    exit 1
fi

# Remove old embeddings to force regeneration
rm -f review_embeddings.pkl

echo "✅ Data files ready for deployment"
echo "📦 You can now deploy to your chosen platform:"
echo ""
echo "For Railway:"
echo "  railway up"
echo ""
echo "For Render:"
echo "  # Connect your GitHub repo to Render"
echo ""
echo "For Heroku:"
echo "  heroku create"
echo "  git push heroku main"
echo ""
echo "For Vercel:"
echo "  vercel"
