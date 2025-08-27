# 🔍 Dermalogica Review Search Application

A powerful AI-powered semantic search application for finding Dermalogica product reviews based on specific topics and use cases.

## Features

- **Semantic Search**: Uses AI embeddings to find reviews by meaning, not just keywords
- **Product Filtering**: Search within specific products or across all products
- **Beautiful UI**: Modern, responsive web interface
- **Similarity Scoring**: Shows how well each review matches your search query
- **Rich Metadata**: Displays ratings, sources, reviewers, and dates

## Quick Start

1. **Install and Run**:
   ```bash
   python run_search_app.py
   ```

2. **Open your browser** to: http://localhost:8000

3. **Search for reviews**:
   - Select a product from the dropdown (e.g., "precleanse")
   - Enter your search query (e.g., "active lifestyle")
   - Click "Search Reviews"

## Example Searches

### For PreCleanse + "active lifestyle":
- "use before runs to keep sweat from clogging pores"
- "used along on dry skin before workouts"
- "great for gym sessions"

### For Daily Microfoliant + "sensitive skin":
- "gentle enough for daily use"
- "doesn't irritate my sensitive skin"
- "mild exfoliation"

### For Special Cleansing Gel + "oily skin":
- "controls oil production"
- "great for combination skin"
- "reduces shine"

## How It Works

1. **Vector Database**: All 60,440+ reviews are converted into AI embeddings using sentence transformers
2. **Semantic Matching**: Your search query is also converted to embeddings and matched against reviews
3. **Cosine Similarity**: Results are ranked by semantic similarity, not just keyword matching
4. **Product Filtering**: Results can be filtered to specific products or searched across all products

## Technical Details

- **Backend**: FastAPI with ChromaDB vector database
- **AI Model**: Sentence Transformers (all-MiniLM-L6-v2)
- **Frontend**: Modern HTML/CSS with responsive design
- **Data**: 60,440 reviews from Amazon, Ulta, Reddit, and Sephora

## API Endpoints

- `GET /`: Main search interface
- `GET /products`: Get list of all products
- `POST /search`: Perform search (form submission)
- `GET /api/search?product=X&query=Y`: Programmatic search API

## Requirements

- Python 3.7+
- FastAPI
- ChromaDB
- Sentence Transformers
- Pandas
- NumPy

All dependencies are automatically installed when you run `run_search_app.py`.

## File Structure

```
├── vector_search_app.py      # Main application
├── run_search_app.py         # Startup script
├── requirements.txt          # Python dependencies
├── templates/
│   └── index.html           # Web interface
├── chroma_db/               # Vector database (created on first run)
└── dermalogica_aggregated_reviews.csv  # Source data
```

## Performance

- **Initial Setup**: 2-3 minutes to create vector database (first run only)
- **Search Speed**: < 1 second for most queries
- **Memory Usage**: ~2GB RAM for full dataset
- **Storage**: ~500MB for vector database

## Troubleshooting

- **Port 8000 in use**: Change the port in `vector_search_app.py`
- **Memory issues**: Reduce batch_size in the vector creation code
- **Slow performance**: Ensure you have sufficient RAM and CPU

Enjoy exploring the Dermalogica reviews! 🧴✨
