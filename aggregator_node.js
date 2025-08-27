const fs = require('fs');

// Complete script to aggregate and standardize Dermalogica reviews
async function aggregateAndStandardizeReviews() {
    console.log("=== Dermalogica Review Aggregation Tool ===\n");
    
    // 1. LOAD PRODUCT MASTER LIST
    console.log("Step 1: Loading product master list...");
    let standardProductNames = [];
    try {
        const productsContent = fs.readFileSync('products_us.md', 'utf8');
        const responseBodyStart = productsContent.indexOf('## Response Body');
        const jsonStart = productsContent.indexOf('{', responseBodyStart);
        const jsonEnd = productsContent.lastIndexOf('}') + 1;
        const jsonContent = productsContent.substring(jsonStart, jsonEnd);
        const productsData = JSON.parse(jsonContent);
        standardProductNames = productsData.all_products.map(product => product.name.toLowerCase().trim());
        console.log(`✓ Loaded ${standardProductNames.length} standard product names`);
    } catch (error) {
        console.log("⚠️  Could not load products_us.md, using empty product list");
    }
    
    // 2. LOAD UNIFIED REVIEWS FILE
    console.log("\nStep 2: Loading unified reviews file...");
    
    // Proper CSV parser that handles quoted fields with commas
    function parseCSV(csvText) {
        const lines = csvText.split('\n').filter(line => line.trim());
        if (lines.length === 0) return [];
        
        function parseCSVLine(line) {
            const result = [];
            let current = '';
            let inQuotes = false;
            let i = 0;
            
            while (i < line.length) {
                const char = line[i];
                
                if (char === '"') {
                    if (inQuotes && line[i + 1] === '"') {
                        // Escaped quote
                        current += '"';
                        i += 2;
                    } else {
                        // Toggle quote state
                        inQuotes = !inQuotes;
                        i++;
                    }
                } else if (char === ',' && !inQuotes) {
                    // Field separator
                    result.push(current.trim());
                    current = '';
                    i++;
                } else {
                    current += char;
                    i++;
                }
            }
            
            // Add the last field
            result.push(current.trim());
            return result;
        }
        
        const headers = parseCSVLine(lines[0]).map(h => h.replace(/"/g, ''));
        const data = [];
        
        for (let i = 1; i < lines.length; i++) {
            const values = parseCSVLine(lines[i]).map(v => v.replace(/^"|"$/g, ''));
            const row = {};
            headers.forEach((header, index) => {
                row[header] = values[index] || null;
            });
            data.push(row);
        }
        
        return data;
    }
    
    let unifiedData = { data: [] };
    
    // Load all source files directly
    const sourceFiles = [
        'dermalogica_amazon_reviews.csv',
        'dermalogica_sephora_reviews.csv',
        'dermalogica_ulta_reviews.csv'
    ];
    
    let allReviews = [];
    
    for (const fileName of sourceFiles) {
        try {
            const csvContent = fs.readFileSync(fileName, 'utf8');
            const reviews = parseCSV(csvContent);
            console.log(`✓ ${fileName}: ${reviews.length} records loaded`);
            
            // Add source information
            reviews.forEach((review, index) => {
                review._source_file = fileName;
                review._source = fileName.includes('amazon') ? 'Amazon' :
                               fileName.includes('sephora') ? 'Sephora' : 
                               fileName.includes('ulta') ? 'Ulta' : 'Unknown';
                review._id = `${review._source.toLowerCase()}_${index + 1}`;
            });
            
            allReviews.push(...reviews);
        } catch (error) {
            console.log(`⚠️  Could not load ${fileName}: ${error.message}`);
        }
    }
    
    console.log(`✓ Total reviews loaded: ${allReviews.length}`);
    unifiedData.data = allReviews;
    
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
        const words = normalized.split(/\s+/).filter(word => word.length >= 3);
        
        standardNames.forEach(standardName => {
            const standardWords = standardName.split(/\s+/).filter(word => word.length >= 3);
            
            // Calculate word overlap
            const commonWords = words.filter(word => 
                standardWords.some(sw => sw.includes(word) || word.includes(sw))
            );
            
            const score = commonWords.length / Math.max(words.length, standardWords.length);
            
            if (score > maxScore && score > 0.3) { // Minimum 30% match
                maxScore = score;
                bestMatch = standardName;
            }
        });
        
        return bestMatch;
    }
    
    // 4. PROCESS UNIFIED DATA
    console.log("\nStep 3: Processing unified reviews...");
    
    // Process reviews with proper field mappings for all sources
    const processedReviews = unifiedData.data
        .filter(row => {
            const reviewText = row.text || row.Text || row.content || '';
            return reviewText && String(reviewText).trim();
        })
        .map((row, index) => {
            // Handle different field names across sources
            const originalName = row.product_name || row.Product || '';
            const standardizedName = findBestMatch(originalName, standardProductNames) || originalName;
            
            return {
                id: row._id || `review_${index + 1}`,
                source: row._source || 'unknown',
                product_name: standardizedName,
                original_product_name: originalName,
                rating: row.review_rating || row.Rating || row.rating || null,
                title: row.title || row.Title || '',
                review_text: row.text || row.Text || row.content || '',
                content_type: 'review',
                reviewer: row.reviewer || row.Reviewer || '',
                date: row.date || row.Date || '',
                location: row.location || row.Location || '',
                price: row.price || '',
                review_count: row.review_count || '',
                overall_rating: row.overall_rating || '',
                traits: row.traits || ''
            };
        });
    
    // 5. USE PROCESSED REVIEWS
    const finalReviews = processedReviews;
    
    // 6. CALCULATE STATISTICS
    const matchedReviews = finalReviews.filter(r => 
        standardProductNames.includes(r.product_name)).length;
    
    console.log("\n=== PROCESSING COMPLETE ===");
    console.log(`Total reviews aggregated: ${finalReviews.length.toLocaleString()}`);
    
    // Breakdown by source
    const sourceStats = {};
    finalReviews.forEach(review => {
        const source = review.source;
        sourceStats[source] = (sourceStats[source] || 0) + 1;
    });
    
    console.log("\nBreakdown by source:");
    Object.entries(sourceStats).forEach(([source, count]) => {
        console.log(`  ${source}: ${count.toLocaleString()} reviews`);
    });
    
    if (standardProductNames.length > 0) {
        console.log(`\nStandardization success rate: ${matchedReviews.toLocaleString()}/${finalReviews.length.toLocaleString()} (${(matchedReviews/finalReviews.length*100).toFixed(1)}%)`);
    }
    
    // 7. ANALYZE PRODUCT DISTRIBUTION
    const productCounts = {};
    finalReviews.forEach(review => {
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
    
    // Simple CSV converter
    function convertToCSV(data) {
        if (data.length === 0) return '';
        
        const headers = Object.keys(data[0]);
        const csvRows = [headers.join(',')];
        
        for (const row of data) {
            const values = headers.map(header => {
                const value = row[header] || '';
                // Escape quotes and wrap in quotes if contains comma or quote
                const escaped = String(value).replace(/"/g, '""');
                return escaped.includes(',') || escaped.includes('"') ? `"${escaped}"` : escaped;
            });
            csvRows.push(values.join(','));
        }
        
        return csvRows.join('\n');
    }
    
    const csv = convertToCSV(finalReviews);
    
    console.log(`✓ CSV created: ${(csv.length / 1024 / 1024).toFixed(2)} MB`);
    
    return {
        data: finalReviews,
        csv: csv,
        statistics: {
            totalReviews: finalReviews.length,
            uniqueProducts: uniqueProducts,
            sources: sourceStats,
            matchRate: standardProductNames.length > 0 ? (matchedReviews/finalReviews.length*100).toFixed(1) : 0
        }
    };
}

// Run the aggregation
async function main() {
    try {
        console.log("Starting aggregation process...\n");
        const result = await aggregateAndStandardizeReviews();

        // Save CSV to file
        fs.writeFileSync('dermalogica_aggregated_reviews.csv', result.csv, 'utf8');

        console.log("\n✅ AGGREGATION COMPLETE!");
        console.log(`✅ CSV file 'dermalogica_aggregated_reviews.csv' saved to disk`);
        console.log(`✅ Total reviews: ${result.statistics.totalReviews.toLocaleString()}`);
        console.log(`✅ Unique products: ${result.statistics.uniqueProducts}`);
        
        // Save statistics to JSON file
        fs.writeFileSync('aggregation_statistics.json', JSON.stringify(result.statistics, null, 2), 'utf8');
        console.log(`✅ Statistics saved to 'aggregation_statistics.json'`);
        
    } catch (error) {
        console.error("Error running aggregation:", error);
    }
}

main();
