# Deployment Guide - DigitalOcean App Platform

## Prerequisites
- DigitalOcean account
- GitHub repository connected to your project

## Method 1: Web Interface (Recommended)

1. **Go to DigitalOcean App Platform**
   - Visit https://cloud.digitalocean.com/apps
   - Click "Create App"

2. **Connect Your Repository**
   - Choose "GitHub" as source
   - Select your repository: `etw63/dermalogica-review-search`
   - Select branch: `main`

3. **Configure the App**
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Run Command**: `uvicorn vector_search_app:app --host 0.0.0.0 --port $PORT`
   - **HTTP Port**: `8000`

4. **Set Environment Variables**
   - `PYTHON_VERSION`: `3.11`
   - `SIMILARITY_THRESHOLD`: `0.1`
   - `MAX_RESULTS`: `1000`

5. **Choose Plan**
   - **Basic Plan**: $5/month (recommended for testing)
   - **Professional Plan**: $12/month (for production)

6. **Deploy**
   - Click "Create Resources"
   - Wait for build and deployment

## Method 2: CLI Deployment

1. **Install doctl CLI**
   ```bash
   brew install doctl
   ```

2. **Authenticate**
   ```bash
   doctl auth init
   ```

3. **Deploy**
   ```bash
   doctl apps create --spec .do/app.yaml
   ```

## Important Notes

### Data Files
- Your `dermalogica_aggregated_reviews.csv` file will be included in the deployment
- The `review_embeddings.pkl` file will be regenerated on first startup
- This may take 5-10 minutes on the first deployment

### Scaling
- Start with Basic plan ($5/month)
- Upgrade to Professional if you need more resources
- You can scale horizontally by increasing instance count

### Monitoring
- View logs in the DigitalOcean dashboard
- Set up alerts for performance monitoring
- Monitor resource usage

## Cost Estimation
- **Basic Plan**: $5/month (1 vCPU, 1GB RAM, 25GB storage)
- **Professional Plan**: $12/month (1 vCPU, 2GB RAM, 50GB storage)

## Troubleshooting

### Build Failures
- Check that all dependencies are in `requirements.txt`
- Ensure Python version is compatible (3.11)

### Memory Issues
- Upgrade to Professional plan for more RAM
- The embeddings generation requires significant memory

### Slow Startup
- First deployment will be slow due to embedding generation
- Subsequent deployments will be faster
