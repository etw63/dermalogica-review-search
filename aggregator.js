import Papa from 'papaparse';

// Complete script to aggregate and standardize Dermalogica reviews
async function aggregateAndStandardizeReviews() {
    console.log("=== Dermalogica Review Aggregation Tool ===\n");
    
    // 1. LOAD PRODUCT MASTER LIST
    console.log("Step 1: Loading product master list...");
    const productsContent = await window.fs.readFile('products_us.md', { encoding: 'utf8' });
    const responseBodyStart = productsContent.indexOf('## Response Body');
    const jsonStart = productsContent.indexOf('{', responseBodyStart);
    const jsonEnd = productsContent.lastIndexOf('}') + 1;
    const jsonContent = productsContent.substring(jsonStart, jsonEnd);
    const productsData = JSON.parse(jsonContent);
    const standardProductNames = productsData.all_products.map(product => product.name.toLowerCase().trim());
    console.log(`✓ Loaded ${standardProductNames.length} standard product names`);
    
    // 2. LOAD REVIEW FILES
    console.log("\nStep 2: Loading review files...");
    
    const amazonCsv = await window.fs.readFile('dermalogica_amazon_reviews.csv', { encoding: 'utf8' });
    const amazonData = Papa.parse(amazonCsv, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        delimitersToGuess: [',', '\t', '|', ';']
    });
    console.log(`✓ Amazon: ${amazonData.data.length} reviews`);
    
    const sephoraCsv = await window.fs.readFile('dermalogica_sephora_reviews.csv', { encoding: 'utf8' });
    const sephoraData = Papa.parse(sephoraCsv, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        delimitersToGuess: [',', '\t', '|', ';']
    });
    console.log(`✓ Sephora: ${sephoraData.data.length} reviews`);
    
    const ultaCsv = await window.fs.readFile('dermalogica_ulta_reviews.csv', { encoding: 'utf8' });
    const ultaData = Papa.parse(ultaCsv, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        delimitersToGuess: [',', '\t', '|', ';']
    });
    console.log(`✓ Ulta: ${ultaData.data.length} reviews`);
    
    // 3. DEFINE MAPPING FUNCTIONS
    function normalizeProductName(name) {
        if (!name) return '';
        
        let normalized = name.toLowerCase()
            .replace(/dermalogica\s*/gi, '')  // Remove brand
            .replace(/\s*-\s*/g, ' ')          // Replace hyphens
            .replace(/[^\w\s]/g, ' ')          // Remove special chars
            .replace(/\s+/g, ' ')              // Normalize spaces
            .trim();
        
        // Remove size/variant indicators
        const patternsToRemove = [
            /\d+\.\d+\s*(oz|ml|g|fl|lb)/gi,
            /\d+\s*(oz|ml|g|fl|lb)/gi,
            /\bmini\b/gi,
            /\btravel\s*size\b/gi,
            /\brefillable\b/gi,
            /\brefill\b/gi,
            /\bkit\b/gi,
            /\bbundle\b/gi,
            /\$[\d.]+/g,
            /\bexclusive\b/gi,
            /\bulta\s*beauty\b/gi,
            /\bnew\s*arrival\b/gi,
            /\bshop\s*now\b/gi,
            /\b\d+\s*x\s*\d+\s*ml\b/gi,
            /\btube[s]?\b/gi,
            /\bwrinkles\b/gi,
            /\bhyaluronic\b/gi
        ];
        
        patternsToRemove.forEach(pattern => {
            normalized = normalized.replace(pattern, ' ').replace(/\s+/g, ' ').trim();
        });
        
        return normalized;
    }
    
    function findBestMatch(productName, standardNames) {
        if (!productName) return null;
        
        const normalized = normalizeProductName(productName);
        
        // Check for exact match
        if (standardNames.includes(normalized)) {
            return normalized;
        }
        
        let bestMatch = null;
        let maxScore = 0;
        
        // Split into meaningful words (3+ characters)
        const normalizedWords = normalized.split(' ').filter(w => w.length > 2);
        
        for (const standardName of standardNames) {
            const standardWords = standardName.split(' ').filter(w => w.length > 2);
            
            let score = 0;
            
            // Count matching words
            for (const word of normalizedWords) {
                if (standardWords.includes(word)) {
                    score += 2;
                }
            }
            
            // Bonus if standard name is substring of normalized
            if (normalized.includes(standardName)) {
                score += standardWords.length * 3;
            }
            
            // Check if all standard words are present
            const allWordsPresent = standardWords.every(word => 
                normalizedWords.includes(word) || normalized.includes(word)
            );
            if (allWordsPresent) {
                score += standardWords.length * 2;
            }
            
            // Penalty for length difference
            const lengthDiff = Math.abs(normalizedWords.length - standardWords.length);
            score -= lengthDiff * 0.5;
            
            if (score > maxScore) {
                maxScore = score;
                bestMatch = standardName;
            }
        }
        
        // Return match only if confident (score threshold)
        return maxScore >= 3 ? bestMatch : null;
    }
    
    // 4. PROCESS AND STANDARDIZE REVIEWS
    console.log("\nStep 3: Processing and standardizing product names...");
    
    const processedAmazon = amazonData.data.map(row => ({
        source: 'Amazon',
        product_name_original: row.product_name,
        product_name: findBestMatch(row.product_name, standardProductNames) || 
                      normalizeProductName(row.product_name),
        rating: row.review_rating,
        reviewer: row.reviewer || '',
        date: row.date || '',
        title: row.title || '',
        review_text: row.text || ''
    }));
    
    const processedSephora = sephoraData.data.map(row => ({
        source: 'Sephora',
        product_name_original: row.product_name,
        product_name: findBestMatch(row.product_name, standardProductNames) || 
                      normalizeProductName(row.product_name),
        rating: row.review_rating,
        reviewer: row.reviewer || '',
        date: row.date || '',
        title: row.title || '',
        review_text: row.text || ''
    }));
    
    const processedUlta = ultaData.data.map(row => ({
        source: 'Ulta',
        product_name_original: row.Product,
        product_name: findBestMatch(row.Product, standardProductNames) || 
                      normalizeProductName(row.Product),
        rating: row.Rating,
        reviewer: row.Reviewer || '',
        date: row.Date || '',
        title: row.Title || '',
        review_text: row.Text || ''
    }));
    
    // 5. COMBINE ALL REVIEWS
    const allReviews = [...processedAmazon, ...processedSephora, ...processedUlta];
    
    // 6. CALCULATE STATISTICS
    const matchedAmazon = processedAmazon.filter(r => 
        standardProductNames.includes(r.product_name)).length;
    const matchedSephora = processedSephora.filter(r => 
        standardProductNames.includes(r.product_name)).length;
    const matchedUlta = processedUlta.filter(r => 
        standardProductNames.includes(r.product_name)).length;
    
    console.log("\n=== PROCESSING COMPLETE ===");
    console.log(`Total reviews aggregated: ${allReviews.length.toLocaleString()}`);
    console.log("\nBreakdown by source:");
    console.log(`  Amazon:  ${processedAmazon.length.toLocaleString()} reviews`);
    console.log(`  Sephora: ${processedSephora.length.toLocaleString()} reviews`);
    console.log(`  Ulta:    ${processedUlta.length.toLocaleString()} reviews`);
    
    console.log("\nStandardization success rates:");
    console.log(`  Amazon:  ${matchedAmazon.toLocaleString()}/${processedAmazon.length.toLocaleString()} (${(matchedAmazon/processedAmazon.length*100).toFixed(1)}%)`);
    console.log(`  Sephora: ${matchedSephora.toLocaleString()}/${processedSephora.length.toLocaleString()} (${(matchedSephora/processedSephora.length*100).toFixed(1)}%)`);
    console.log(`  Ulta:    ${matchedUlta.toLocaleString()}/${processedUlta.length.toLocaleString()} (${(matchedUlta/processedUlta.length*100).toFixed(1)}%)`);
    
    // 7. ANALYZE PRODUCT DISTRIBUTION
    const productCounts = {};
    allReviews.forEach(review => {
        const name = review.product_name;
        productCounts[name] = (productCounts[name] || 0) + 1;
    });
    
    const uniqueProducts = Object.keys(productCounts).length;
    const sortedProducts = Object.entries(productCounts)
        .sort((a, b) => b[1] - a[1]);
    
    console.log(`\nUnique products after standardization: ${uniqueProducts}`);
    console.log("\nTop 10 products by review count:");
    sortedProducts.slice(0, 10).forEach(([product, count], idx) => {
        console.log(`  ${idx + 1}. ${product}: ${count.toLocaleString()} reviews`);
    });
    
    // 8. CREATE CSV FOR EXPORT
    console.log("\nStep 4: Creating CSV for export...");
    const csv = Papa.unparse(allReviews, {
        header: true,
        skipEmptyLines: true
    });
    
    console.log(`✓ CSV created: ${(csv.length / 1024 / 1024).toFixed(2)} MB`);
    
    return {
        data: allReviews,
        csv: csv,
        statistics: {
            totalReviews: allReviews.length,
            uniqueProducts: uniqueProducts,
            sources: {
                amazon: processedAmazon.length,
                sephora: processedSephora.length,
                ulta: processedUlta.length
            },
            matchRates: {
                amazon: (matchedAmazon/processedAmazon.length*100).toFixed(1),
                sephora: (matchedSephora/processedSephora.length*100).toFixed(1),
                ulta: (matchedUlta/processedUlta.length*100).toFixed(1)
            }
        }
    };
}

// Function to save CSV to file
function downloadCSV(csvContent, filename = 'dermalogica_aggregated_reviews.csv') {
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Run the aggregation
console.log("Starting aggregation process...\n");
const result = await aggregateAndStandardizeReviews();

// Store results globally for access
window.aggregatedReviews = result;

console.log("\n✅ AGGREGATION COMPLETE!");
console.log("\nTo download the CSV file, run:");
console.log("downloadCSV(window.aggregatedReviews.csv)");
console.log("\nTo access the data programmatically:");
console.log("window.aggregatedReviews.data");
console.log("\nTo see statistics:");
console.log("window.aggregatedReviews.statistics");