# API Test: GET_ALL_PRODUCTS

Timestamp: 2025-08-27 20:41:57 UTC

**Method:** `POST`  
**URL:** `https://dioxide.herokuapp.com/recommendations/get_all_products/`

## Request Body
```json
{
  "country_code": "US",
  "host_site": "dermalogica.com",
  "lang_code": "en-US"
}
```

## cURL executed
```bash
curl -i -X POST "https://dioxide.herokuapp.com/recommendations/get_all_products/" \
  -H "Content-Type: application/json" \
  -d '{"country_code":"US","host_site":"dermalogica.com","lang_code":"en-US"}'
```

## Response Status

200

## Response Headers
```
Content-Length: 85188
Content-Type: text/html; charset=utf-8
Cross-Origin-Opener-Policy: same-origin
Date: Wed, 27 Aug 2025 20:41:57 GMT
Nel: {"report_to":"heroku-nel","response_headers":["Via"],"max_age":3600,"success_fraction":0.01,"failure_fraction":0.1}
Referrer-Policy: same-origin
Report-To: {"group":"heroku-nel","endpoints":[{"url":"https://nel.heroku.com/reports?s=MOX2WHG7PMf54KjwpdD0d168%2B8%2FiblP3K3glRd1mQAw%3D\u0026sid=af571f24-03ee-46d1-9f90-ab9030c2c74c\u0026ts=1756327317"}],"max_age":3600}
Reporting-Endpoints: heroku-nel="https://nel.heroku.com/reports?s=MOX2WHG7PMf54KjwpdD0d168%2B8%2FiblP3K3glRd1mQAw%3D&sid=af571f24-03ee-46d1-9f90-ab9030c2c74c&ts=1756327317"
Server: Heroku
Vary: Origin
Via: 1.1 heroku-router
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
```

## Response Body
```json
{
  "status": 200,
  "matrix_directory": "./resources/matrices/US",
  "US_default": false,
  "all_products": [
    {
      "pimcore_id": 9172,
      "name": "multivitamin thermafoliant",
      "priceCurrency": "$",
      "price": "69.00",
      "shopify_id": "5710542012568",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070253887640",
          "default": true,
          "default_fallback": false,
          "size": "2.5 oz",
          "price": "69.00",
          "sku": "111030"
        }
      ],
      "productUrl": "https://dermalogica.com/products/multivitamin-thermafoliant",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/75-ml-2.5-oz-multivitamin-thermafoliant_Size.png",
      "name_original": "multivitamin thermafoliant",
      "mediumDescription": "Thermal skin exfoliant infuses skin with age-fighting ingredients. This powerful skin polisher combines physical and chemical exfoliants to refine skin texture, enhance penetration of age-fighting vitamins into skin, and reveal smoother, fresher skin immediately.",
      "tagline": "heat-activated scrub",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9144,
      "name": "skin smoothing cream",
      "priceCurrency": "$",
      "price": "19.00",
      "shopify_id": "5710551187608",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070281674904",
          "default": false,
          "default_fallback": false,
          "size": "1.7 oz",
          "price": "49.00",
          "sku": "111324"
        },
        {
          "shopify_variant_id": "36070281707672",
          "default": false,
          "default_fallback": false,
          "size": "3.4 oz",
          "price": "82.00",
          "sku": "111323"
        },
        {
          "shopify_variant_id": "42642308464792",
          "default": false,
          "default_fallback": false,
          "size": "5.1 oz",
          "price": "99.00",
          "sku": "811323"
        },
        {
          "shopify_variant_id": "36070281740440",
          "default": true,
          "default_fallback": false,
          "size": "0.5 oz",
          "price": "19.00",
          "sku": "111325"
        }
      ],
      "productUrl": "https://dermalogica.com/products/skin-smoothing-cream",
      "imageUrl": "https://dioxide.s3-us-west-1.amazonaws.com/Product_Images_Updates/skin-smoothing-cream_15-01_428x448.png",
      "name_original": "skin smoothing cream",
      "mediumDescription": "This next-generation formulation of our best-selling moisturizer features state-of-the-art Active HydraMesh Technology™ to infuse skin with 48 hours of continuous hydration and help protect against environmental stress.",
      "tagline": "medium-weight moisturizer",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9184,
      "name": "multivitamin power firm",
      "priceCurrency": "$",
      "price": "69.00",
      "shopify_id": "5710541652120",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070252871832",
          "default": true,
          "default_fallback": false,
          "size": "0.5 oz",
          "price": "69.00",
          "sku": "111033"
        },
        {
          "shopify_variant_id": "42715738505368",
          "default": false,
          "default_fallback": false,
          "size": "1.0 oz",
          "price": "99.00",
          "sku": "811033"
        }
      ],
      "productUrl": "https://dermalogica.com/products/multivitamin-power-firm",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/15-ml-0.5-oz-multivitamin-power-firm_Size.png",
      "name_original": "multivitamin power firm",
      "mediumDescription": "Combat visible lines around the eye area with this powerful firming complex of skin-rebuilding antioxidant vitamins, protective Silicones and Red Seaweed Extract.",
      "tagline": "smoothing vitamin complex",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9209,
      "name": "ultracalming mist",
      "priceCurrency": "$",
      "price": "47.00",
      "shopify_id": "5710549418136",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070276792472",
          "default": true,
          "default_fallback": false,
          "size": "6.0 oz",
          "price": "47.00",
          "sku": "110545"
        }
      ],
      "productUrl": "https://dermalogica.com/products/ultracalming-mist",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/ultracalming-mist.png",
      "name_original": "ultracalming mist",
      "mediumDescription": "A soothing, hydrating mist to calm redness and sensitivity. Used post-cleanse, this lightweight mist helps create a shield against environmental assaults, and synergistically relieves and restores skin while fighting future flare-ups.",
      "tagline": "soothing, cooling spritz",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 17470,
      "name": "biolumin-c gel moisturizer",
      "priceCurrency": "$",
      "price": "69.00",
      "shopify_id": "6919116390552",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "40574560567448",
          "default": true,
          "default_fallback": false,
          "size": "1.7 oz",
          "price": "69.00",
          "sku": "111441"
        }
      ],
      "productUrl": "https://dermalogica.com/products/biolumin-c-gel-moisturizer",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/Biolumin-C_Gel_Moisturizer.png",
      "name_original": "biolumin-c gel moisturizer",
      "mediumDescription": "Daily brightening gel moisturizer provides weightless hydration and gives skin a radiance boost for healthy-looking skin.",
      "tagline": "brightening vitamin c moisturizer",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 56133,
      "name": "pro-collagen banking serum",
      "priceCurrency": "$",
      "price": "89.00",
      "shopify_id": "7823861678232",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "43028927217816",
          "default": true,
          "default_fallback": false,
          "size": "1.0 oz",
          "price": "89.00",
          "sku": "111483"
        },
        {
          "shopify_variant_id": "46890575036568",
          "default": false,
          "default_fallback": false,
          "size": "2.0 oz",
          "price": "129.00",
          "sku": "811483"
        }
      ],
      "productUrl": "https://dermalogica.com/products/pro-collagen-banking-serum",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/4x5_procollagen_bankingserum.png",
      "name_original": "pro-collagen banking serum",
      "mediumDescription": "Powerful serum helps promote, protect, and preserve skin’s collagen today, so you have more for tomorrow – supporting visibly plumper, more luminous skin and hydrating to help reduce the look of fine lines into the future.",
      "tagline": "plumping + preserving serum",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9178,
      "name": "dynamic skin recovery spf50",
      "priceCurrency": "$",
      "price": "28.50",
      "shopify_id": "5710550958232",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "43378556829848",
          "default": true,
          "default_fallback": false,
          "size": "0.5 oz",
          "price": "28.50",
          "sku": "411485"
        },
        {
          "shopify_variant_id": "36070280855704",
          "default": false,
          "default_fallback": false,
          "size": "1.7 oz",
          "price": "85.00",
          "sku": "111456"
        },
        {
          "shopify_variant_id": "42432981893272",
          "default": false,
          "default_fallback": false,
          "size": "3.4 oz",
          "price": "129.00",
          "sku": "811048"
        }
      ],
      "productUrl": "https://dermalogica.com/products/dynamic-skin-recovery-spf50",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/50-ml-1.7-oz-dynamic-skin-recovery-spf50_Size.png",
      "name_original": "dynamic skin recovery spf50",
      "mediumDescription": "Broad Spectrum moisturizer helps combat the appearance of skin aging. Help minimize the appearance of skin aging with this medium-weight, emollient daily moisturizer with Broad Spectrum SPF 50.",
      "tagline": "firming, emollient moisturizer",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 11329,
      "name": "retinol clearing oil",
      "priceCurrency": "$",
      "price": "89.00",
      "shopify_id": "5710540734616",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070249726104",
          "default": true,
          "default_fallback": false,
          "size": "1.0 oz",
          "price": "89.00",
          "sku": "111395"
        }
      ],
      "productUrl": "https://dermalogica.com/products/retinol-clearing-oil",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/Retinol_Clearing_Oil.png",
      "name_original": "retinol clearing oil",
      "mediumDescription": "A high-performance night oil combining Retinol and Salicylic Acid into one skin-soothing formula that helps reduce visible signs of premature skin aging and clear breakouts.",
      "tagline": "clears skin overnight",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 10703,
      "name": "phyto-nature firming serum",
      "priceCurrency": "$",
      "price": "164.00",
      "shopify_id": "5710543814808",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070261096600",
          "default": true,
          "default_fallback": false,
          "size": "1.3 oz",
          "price": "164.00",
          "sku": "111369"
        }
      ],
      "productUrl": "https://dermalogica.com/products/phyto-nature-firming-serum",
      "imageUrl": "https://dioxide.s3-us-west-1.amazonaws.com/Product_Images_Full_Size/Phyto-Nature_Firming_Serum.png",
      "name_original": "phyto-nature firming serum",
      "mediumDescription": "Our most advanced, dual-phase serum combines highly-active botanicals with biomimetic technology to reduce visible signs of skin aging and reawakens the nature of younger-looking skin.",
      "tagline": "lifting firming serum",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 16812,
      "name": "dark spot solutions kit",
      "priceCurrency": "$",
      "price": "68.00",
      "shopify_id": "6285635027096",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "42268082471064",
          "default": true,
          "default_fallback": false,
          "size": "KIT",
          "price": "68.00",
          "sku": "111403"
        }
      ],
      "productUrl": "https://dermalogica.com/products/dark-spot-solutions-kit",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Kits/dark-spot-solutions-kit.png",
      "name_original": "dark spot solutions kit",
      "mediumDescription": "Starts working fast to visibly fade + help prevent hyperpigmentation.",
      "tagline": "visibly fade + help prevent hyperpigmentation",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9146,
      "name": "intensive moisture balance",
      "priceCurrency": "$",
      "price": "19.00",
      "shopify_id": "5710550663320",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070280364184",
          "default": true,
          "default_fallback": false,
          "size": "0.5 oz",
          "price": "19.00",
          "sku": "111329"
        },
        {
          "shopify_variant_id": "36070280298648",
          "default": false,
          "default_fallback": false,
          "size": "1.7 oz",
          "price": "49.00",
          "sku": "111328"
        },
        {
          "shopify_variant_id": "36070280331416",
          "default": false,
          "default_fallback": false,
          "size": "3.4 oz",
          "price": "82.00",
          "sku": "111327"
        },
        {
          "shopify_variant_id": "42642311119000",
          "default": false,
          "default_fallback": false,
          "size": "5.1 oz",
          "price": "99.00",
          "sku": "811327"
        }
      ],
      "productUrl": "https://dermalogica.com/products/intensive-moisture-balance",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/50-ml-1.7-oz-intensive-moisture-balance_Size.png",
      "name_original": "intensive moisture balance",
      "mediumDescription": "Ultra-nourishing moisturizer restores lipid balance to dry, depleted skin for optimal barrier performance. BioReplenish Complex™ delivers a proven combination of key barrier lipids to help enhance the skin’s natural resilience and support barrier recovery.",
      "tagline": "ultra-nourishing moisturizer",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 37865,
      "name": "breakout clearing liquid peel",
      "priceCurrency": "$",
      "price": "29.50",
      "shopify_id": "7295554158744",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "41798222086296",
          "default": true,
          "default_fallback": false,
          "size": "1 oz",
          "price": "29.50",
          "sku": "111464"
        }
      ],
      "productUrl": "https://dermalogica.com/products/breakout-clearing-liquid-peel",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/breakout+clearing+liquid+peel_white+(1).png",
      "name_original": "breakout clearing liquid peel",
      "mediumDescription": "An AHA | BHA exfoliating peel that combats breakouts and resurfaces skin for a smoother, brighter, more even skin tone.",
      "tagline": "resurfaces for clearer, smoother skin",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9137,
      "name": "daily microfoliant",
      "priceCurrency": "$",
      "price": "19.50",
      "shopify_id": "5710551548056",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "41330184749208",
          "default": true,
          "default_fallback": false,
          "size": "0.45 oz",
          "price": "19.50",
          "sku": "111248"
        },
        {
          "shopify_variant_id": "44059552415896",
          "default": false,
          "default_fallback": false,
          "size": "1.4 oz",
          "price": "48.00",
          "sku": "811249"
        },
        {
          "shopify_variant_id": "36070282690712",
          "default": false,
          "default_fallback": false,
          "size": "2.6 oz",
          "price": "69.00",
          "sku": "111249"
        },
        {
          "shopify_variant_id": "36530739085464",
          "default": false,
          "default_fallback": false,
          "size": "2.6 oz - refill",
          "price": "62.00",
          "sku": "111429"
        },
        {
          "shopify_variant_id": "42699676254360",
          "default": false,
          "default_fallback": false,
          "size": "2.6 oz + refill bundle",
          "price": "109.00",
          "sku": "111480"
        }
      ],
      "productUrl": "https://dermalogica.com/products/daily-microfoliant",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/75-g-2.6-oz-daily-microfoliant_Size.png",
      "name_original": "daily microfoliant",
      "mediumDescription": "Achieve brighter, smoother skin every day with this iconic exfoliating powder. Rice-based powder activates upon contact with water, releasing Papain, Salicylic Acid and Rice Enzymes to polish skin to perfection.",
      "tagline": "gentle, brightening polisher",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 10700,
      "name": "age bright clearing serum",
      "priceCurrency": "$",
      "price": "78.00",
      "shopify_id": "5710544371864",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070262538392",
          "default": true,
          "default_fallback": false,
          "size": "1.0 oz",
          "price": "78.00",
          "sku": "111332"
        }
      ],
      "productUrl": "https://dermalogica.com/products/age-bright-clearing-serum",
      "imageUrl": "https://dioxide.s3-us-west-1.amazonaws.com/Product_Images_Full_Size/AGE_bright_clearing_serum.png",
      "name_original": "age bright clearing serum",
      "mediumDescription": "This active two-in-one serum clears and helps prevent breakouts while reducing visible skin aging. Salicylic Acid reduces breakouts to clear skin. This highly-concentrated serum exfoliates to help prevent breakouts and accelerates cell turnover to reduce signs of skin aging.",
      "tagline": "brightening clearing serum",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 25016,
      "name": "awaken peptide eye gel",
      "priceCurrency": "$",
      "price": "59.00",
      "shopify_id": "7122206228632",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "41305551995032",
          "default": true,
          "default_fallback": false,
          "size": "0.5 oz",
          "price": "59.00",
          "sku": "111449"
        }
      ],
      "productUrl": "https://dermalogica.com/products/awaken-peptide-eye-gel",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/awaken_peptide_eye_gel.png",
      "name_original": "awaken peptide eye gel",
      "mediumDescription": "Firming, hydrating eye gel helps minimize the appearance of puffiness and fine lines.",
      "tagline": "depuffing eye gel",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 86094,
      "name": "neurotouch symmetry serum",
      "priceCurrency": "$",
      "price": "138.00",
      "shopify_id": "8770535391384",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "46800155803800",
          "default": true,
          "default_fallback": false,
          "size": "1.0 oz",
          "price": "138.00",
          "sku": "111515"
        }
      ],
      "productUrl": "https://dermalogica.com/products/neurotouch-symmetry-serum",
      "imageUrl": "https://dermalogica.widen.net/content/lhg7szmcco/png/NeuroTouch%20Symmetry%20Oil%20-%20Retail%201.png?color=ffffff00&u=dzfqbh&w=2048&h=2048&position=c&crop=false",
      "name_original": "neurotouch symmetry serum",
      "mediumDescription": "Nourishing face serum combines neurotechnology and a biomimetic botanical blend to help visibly restore facial symmetry for a more sculpted appearance.",
      "tagline": "neuroscience-powered sculpting oil",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 15442,
      "name": "sensitive skin rescue",
      "priceCurrency": "$",
      "price": "39.50",
      "shopify_id": "5710540374168",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "42268082602136",
          "default": true,
          "default_fallback": false,
          "size": "KIT",
          "price": "39.50",
          "sku": "111388"
        }
      ],
      "productUrl": "https://dermalogica.com/products/sensitive-skin-rescue-kit",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Kits/sensitive-skin-rescue-kit.png",
      "name_original": "sensitive skin rescue",
      "mediumDescription": "This powerful trio calms, soothes, hydrates and helps defend sensitive skin against future flare-ups.",
      "tagline": "soothes sensitive skin",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 69964,
      "name": "golden hour hydrating spf30 stick",
      "priceCurrency": "$",
      "price": "29.50",
      "shopify_id": "8683922391192",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "46418921554072",
          "default": true,
          "default_fallback": false,
          "size": "0.68 oz",
          "price": "29.50",
          "sku": "111505"
        }
      ],
      "productUrl": "https://dermalogica.com/products/golden-hour-hydrating-spf30-stick",
      "imageUrl": "https://dermalogica.widen.net/content/rfogg202o1/jpeg/US%20golden%20hour%20hydrating%20SPF%20stick%20transparent.jpeg",
      "name_original": "golden hour hydrating spf30 stick",
      "mediumDescription": "Every Hour is Golden Hour with Clear Start by Dermalogica’s 3-in-1 Hydrating SPF stick: Protect with SPF30, Clarify with Willow Bark Extract and Glow with a golden shimmer. Squalane + Jojoba seed oil provide lasting hydration. Throw it in your bag for a glow on the go! Use all over your face + body.",
      "tagline": "SPF30 broad spectrum UV protection",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9285,
      "name": "sound sleep cocoon",
      "priceCurrency": "$",
      "price": "22.50",
      "shopify_id": "5710544928920",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070264045720",
          "default": false,
          "default_fallback": false,
          "size": "1.7 oz",
          "price": "89.00",
          "sku": "111279"
        },
        {
          "shopify_variant_id": "36070264078488",
          "default": true,
          "default_fallback": false,
          "size": "0.34 oz",
          "price": "22.50",
          "sku": "411279"
        }
      ],
      "productUrl": "https://dermalogica.com/products/sound-sleep-cocoon",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/50-ml-1.7-oz-sound-sleep-cocoon_Size.jpg",
      "name_original": "sound sleep cocoon",
      "mediumDescription": "Revitalizing treatment gel-cream transforms skin overnight by optimizing nighttime skin recovery. Motion-activated essential oils work all night to promote deep, restful sleep for healthier-looking skin by morning.",
      "tagline": "transformative night gel-cream",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9291,
      "name": "protection 50 sport spf50",
      "priceCurrency": "$",
      "price": "49.00",
      "shopify_id": "6838275768472",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "40288577618072",
          "default": true,
          "default_fallback": false,
          "size": "5.3 oz",
          "price": "49.00",
          "sku": "111364"
        }
      ],
      "productUrl": "https://dermalogica.com/products/protection-50-sport-spf50",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/156-ml-5.3-oz-protection-50-sport-spf50_Size.png",
      "name_original": "protection 50 sport spf50",
      "mediumDescription": "This sheer solar protection treatment defends against prolonged skin damage from UV light and environmental assault.",
      "tagline": "water-resistant, broad spectrum sun protection",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9301,
      "name": "skin soothing hydrating lotion",
      "priceCurrency": "$",
      "price": "26.00",
      "shopify_id": "5710546993304",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070269321368",
          "default": true,
          "default_fallback": false,
          "size": "2.0 oz",
          "price": "26.00",
          "sku": "111122"
        }
      ],
      "productUrl": "https://dermalogica.com/products/skin-soothing-hydrating-lotion",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/60-ml-2-oz-skin-soothing-hydrating-lotion_Size.jpg",
      "name_original": "skin soothing hydrating lotion",
      "mediumDescription": "Say goodbye to dehydrated, breakout-irritated skin with this lightweight moisturizer! Sheer, easy-to-apply formula helps soothe discomfort and hydrate areas that feel dry.",
      "tagline": "soothing, hydrating relief",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9253,
      "name": "daily superfoliant",
      "priceCurrency": "$",
      "price": "19.50",
      "shopify_id": "5710546272408",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070266994840",
          "default": true,
          "default_fallback": false,
          "size": "0.45 oz",
          "price": "19.50",
          "sku": "111251"
        },
        {
          "shopify_variant_id": "36070266962072",
          "default": false,
          "default_fallback": false,
          "size": "2.0 oz",
          "price": "69.00",
          "sku": "111252"
        }
      ],
      "productUrl": "https://dermalogica.com/products/daily-superfoliant",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/57-g-net-wt-us-2-oz-daily-superfoliant_Size.png",
      "name_original": "daily superfoliant",
      "mediumDescription": "This highly-active resurfacer delivers your smoothest skin ever, and helps fight the biochemical and environmental triggers known to accelerate skin aging.",
      "tagline": "resurfacing, anti-pollution powder exfoliant",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9207,
      "name": "barrier repair",
      "priceCurrency": "$",
      "price": "56.00",
      "shopify_id": "5710549581976",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070277218456",
          "default": true,
          "default_fallback": false,
          "size": "1.0 oz",
          "price": "56.00",
          "sku": "110548"
        }
      ],
      "productUrl": "https://dermalogica.com/products/barrier-repair",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/30-ml-1-oz-barrier-repair_Size.png",
      "name_original": "barrier repair",
      "mediumDescription": "Velvety moisturizer helps fortify sensitized skin with a damaged barrier. Use this unique anhydrous (waterless) moisturizer after toning to help shield against environmental and internal triggers that cause skin stress, and minimize discomfort, burning and itching.",
      "tagline": "shielding, protective moisturizer",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9263,
      "name": "breakout clearing foaming wash",
      "priceCurrency": "$",
      "price": "29.50",
      "shopify_id": "5710547812504",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "41423816392856",
          "default": true,
          "default_fallback": false,
          "size": "10.0 oz",
          "price": "29.50",
          "sku": "111419"
        },
        {
          "shopify_variant_id": "41430037332120",
          "default": false,
          "default_fallback": false,
          "size": "6.0 oz",
          "price": "19.50",
          "sku": "110800"
        }
      ],
      "productUrl": "https://dermalogica.com/products/breakout-clearing-foaming-wash",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/177-ml-6-oz-breakout-clearing-foaming-wash_Size.png",
      "name_original": "breakout clearing foaming wash",
      "mediumDescription": "This breakout fighting, foaming wash clears away dead skin cells, dirt and excess oils that clog pores and cause breakouts.",
      "tagline": "deep cleanse and purifies skin",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9297,
      "name": "biolumin-c serum",
      "priceCurrency": "$",
      "price": "39.00",
      "shopify_id": "5710551351448",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "42914679324824",
          "default": true,
          "default_fallback": false,
          "size": "0.34 oz",
          "price": "39.00",
          "sku": "111372"
        },
        {
          "shopify_variant_id": "36070282166424",
          "default": false,
          "default_fallback": false,
          "size": "1.0 oz",
          "price": "99.00",
          "sku": "111341"
        },
        {
          "shopify_variant_id": "36070282199192",
          "default": false,
          "default_fallback": false,
          "size": "2.0 oz",
          "price": "154.00",
          "sku": "111362"
        }
      ],
      "productUrl": "https://dermalogica.com/products/biolumin-c-serum",
      "imageUrl": "https://dioxide.s3-us-west-1.amazonaws.com/Product_Images_Full_Size/30-ml-1-oz-biolumin-c-serum_Size.png",
      "name_original": "biolumin-c serum",
      "mediumDescription": "A high-performance Vitamin C serum that works with skin’s own defenses to brighten and firm. Advanced bio-technology fuses ultra-stable Vitamin C and Palmitoyl Tripeptide-5, helping to dramatically reduce the appearance of fine lines and wrinkles.",
      "tagline": "brightening vitamin c serum",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9227,
      "name": "skinperfect primer spf30",
      "priceCurrency": "$",
      "price": "59.00",
      "shopify_id": "5710548074648",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070273220760",
          "default": true,
          "default_fallback": false,
          "size": "0.75 oz",
          "price": "59.00",
          "sku": "110640"
        }
      ],
      "productUrl": "https://dermalogica.com/products/skinperfect-primer-spf30",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/22-ml-0.75-oz-skinperfect-primer-spf30_Size.png",
      "name_original": "skinperfect primer spf30",
      "mediumDescription": "Smooth fine lines, brighten and prime for flawless skin, and prep for make-up application. Velvety formula with Soy Protein helps even out skin texture, creating a smoother surface.",
      "tagline": "illuminating make-up prep",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9121,
      "name": "precleanse",
      "priceCurrency": "$",
      "price": "16.00",
      "shopify_id": "5710551974040",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070284492952",
          "default": true,
          "default_fallback": false,
          "size": "1.0 oz",
          "price": "16.00",
          "sku": "111052"
        },
        {
          "shopify_variant_id": "36070284460184",
          "default": false,
          "default_fallback": false,
          "size": "5.1 oz",
          "price": "49.00",
          "sku": "111051"
        },
        {
          "shopify_variant_id": "41726040932504",
          "default": false,
          "default_fallback": false,
          "size": "10 oz",
          "price": "79.00",
          "sku": "811051"
        }
      ],
      "productUrl": "https://dermalogica.com/products/precleanse",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/150-ml-5.1-oz-precleanse_Size.png",
      "name_original": "precleanse",
      "mediumDescription": "Achieve ultra clean and healthy-looking skin with the Double Cleanse regimen that begins with PreCleanse. Thoroughly melt away layers of excess sebum (oil), sunscreen, waterproof makeup, environmental pollutants and residual products that build up on skin throughout the day.",
      "tagline": "oil-busting emulsifyer",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9271,
      "name": "stress positive eye lift",
      "priceCurrency": "$",
      "price": "84.00",
      "shopify_id": "5710545911960",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070266241176",
          "default": true,
          "default_fallback": false,
          "size": "0.85 oz",
          "price": "84.00",
          "sku": "111257"
        }
      ],
      "productUrl": "https://dermalogica.com/products/stress-positive-eye-lift",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/24-ml-0.85-oz-stress-positive-eye-lift_Size.png",
      "name_original": "stress positive eye lift",
      "mediumDescription": "Active, cooling cream-gel masque energizes skin to reduce visible signs of stress. High-performance formula minimizes the appearance of puffiness and dark circles, increases skin luminosity, lifts the eye area and helps restore skin barrier integrity.",
      "tagline": "de-puffing eye treatment and masque",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 10697,
      "name": "clearing defense spf30",
      "priceCurrency": "$",
      "price": "29.50",
      "shopify_id": "5710544011416",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070261457048",
          "default": true,
          "default_fallback": false,
          "size": "2.0 oz",
          "price": "29.50",
          "sku": "111355"
        }
      ],
      "productUrl": "https://dermalogica.com/products/clearing-defense-spf30",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/2oz-clearing-defense-spf30.png",
      "name_original": "clearing defense spf30",
      "mediumDescription": "Shine-reducing SPF moisturizer protects breakout-prone skin from environmental stress. Ultra-lightweight formula provides a long-lasting matte finish without clogging pores.",
      "tagline": "mattifying moisturizer",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 10699,
      "name": "intensive moisture cleanser",
      "priceCurrency": "$",
      "price": "48.00",
      "shopify_id": "5710550171800",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070278561944",
          "default": true,
          "default_fallback": false,
          "size": "5.1 oz",
          "price": "48.00",
          "sku": "111326"
        },
        {
          "shopify_variant_id": "36070278594712",
          "default": false,
          "default_fallback": false,
          "size": "10.0 oz",
          "price": "72.00",
          "sku": "111334"
        }
      ],
      "productUrl": "https://dermalogica.com/products/intensive-moisture-cleanser",
      "imageUrl": "https://dioxide.s3-us-west-1.amazonaws.com/Product_Images_Full_Size/Intensive_Moisture_Cleanser_10oz.png",
      "name_original": "intensive moisture cleanser",
      "mediumDescription": "A light, creamy cleanser that removes impurities while actively nourishing dry, depleted skin. This emollient, lipid-enriched formula cleanses skin while helping to minimize damage of vital proteins and lipids that defend against dryness.",
      "tagline": "nourishing cleanser",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9142,
      "name": "active moist",
      "priceCurrency": "$",
      "price": "49.00",
      "shopify_id": "5710548730008",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070274957464",
          "default": true,
          "default_fallback": false,
          "size": "1.7 oz",
          "price": "49.00",
          "sku": "111064"
        },
        {
          "shopify_variant_id": "36070274990232",
          "default": false,
          "default_fallback": false,
          "size": "3.4 oz",
          "price": "82.00",
          "sku": "111059"
        },
        {
          "shopify_variant_id": "42642312691864",
          "default": false,
          "default_fallback": false,
          "size": "5.1 oz",
          "price": "99.00",
          "sku": "811059"
        }
      ],
      "productUrl": "https://dermalogica.com/products/active-moist",
      "imageUrl": "https://dioxide.s3-us-west-1.amazonaws.com/Product_Images_Full_Size/Active+Moist+3.4.png",
      "name_original": "active moist",
      "mediumDescription": "Sheer, easy-to-apply Active Moist formula contains Silk Amino Acids and a unique combination of plant extracts that help improve skin texture and combat surface dehydration.",
      "tagline": "light, oil-free lotion",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 16348,
      "name": "powerbright moisturizer spf50",
      "priceCurrency": "$",
      "price": "85.00",
      "shopify_id": "6285568082072",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "38178879963288",
          "default": true,
          "default_fallback": false,
          "size": "1.7 oz",
          "price": "85.00",
          "sku": "111011"
        }
      ],
      "productUrl": "https://dermalogica.com/products/powerbright-moisturizer-spf-50",
      "imageUrl": "https://dioxide.s3-us-west-1.amazonaws.com/Product_Images_Full_Size/Powerbright_Moisturizer_SPF50.png",
      "name_original": "powerbright moisturizer spf50",
      "mediumDescription": "Daily SPF moisturizer helps shield against dark spots.",
      "tagline": "dark spot defense",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 66780,
      "name": "multivitamin power recovery cream",
      "priceCurrency": "$",
      "price": "95.00",
      "shopify_id": "8553004499096",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "45722010714264",
          "default": true,
          "default_fallback": false,
          "size": "1.7 oz",
          "price": "95.00",
          "sku": "111507"
        },
        {
          "shopify_variant_id": "45791576850584",
          "default": false,
          "default_fallback": false,
          "size": "KIT",
          "price": "129.00",
          "sku": "111513"
        }
      ],
      "productUrl": "https://dermalogica.com/products/multivitamin-power-recovery-cream",
      "imageUrl": "https://dermalogica.widen.net/content/zyg7wr4jwb/png/Multivitamin%20Power%20Recovery%20Cream_1.7oz_FRONT.png",
      "name_original": "multivitamin power recovery cream",
      "mediumDescription": "Vitamin-rich daily moisturizer treats and helps prevent signs of stressed skin – including the appearance of fine lines, dehydration, and dullness.",
      "tagline": "treats + helps prevent stressed skin",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9186,
      "name": "renewal lip complex",
      "priceCurrency": "$",
      "price": "32.00",
      "shopify_id": "5710541586584",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070252806296",
          "default": true,
          "default_fallback": false,
          "size": "0.06 oz",
          "price": "32.00",
          "sku": "111246"
        }
      ],
      "productUrl": "https://dermalogica.com/products/renewal-lip-complex",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/1.75-ml-0.06-oz-renewal-lip-complex_Size.png",
      "name_original": "renewal lip complex",
      "mediumDescription": "A daily lip treatment that restores delicate tissue, minimizes contour lines and helps prevent the signs of aging.",
      "tagline": "conditioning treatment balm",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 72610,
      "name": "biolumin-c heat aging protector spf 50",
      "priceCurrency": "$",
      "price": "28.50",
      "shopify_id": "8621224493208",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "46073127633048",
          "default": false,
          "default_fallback": false,
          "size": "1.7 oz",
          "price": "79.00",
          "sku": "111510"
        },
        {
          "shopify_variant_id": "46332664512664",
          "default": true,
          "default_fallback": false,
          "size": "0.5 oz",
          "price": "28.50",
          "sku": "411510"
        }
      ],
      "productUrl": "https://dermalogica.com/products/biolumin-c-heat-aging-protector-spf50",
      "imageUrl": "https://dermalogica.widen.net/content/d89gbn9nfq/png/BioLumin-C%20Heat%20Aging%20Protector%20SPF%2050%20FRONT.png",
      "name_original": "biolumin-c heat aging protector spf 50",
      "mediumDescription": "Daily moisturizer with SPF 50 helps defend against signs of skin aging from UVA/UVB rays. Formulated with ultra-stable Vitamin C and ThermaRadiance Complex to brighten and help shield skin from free radicals.",
      "tagline": "vitamin c + daily radiance moisturizer",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 27581,
      "name": "daily brightness boosters kit",
      "priceCurrency": "$",
      "price": "49.50",
      "shopify_id": "7166431068312",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "42268082831512",
          "default": true,
          "default_fallback": false,
          "size": "KIT",
          "price": "49.50",
          "sku": "111450"
        }
      ],
      "productUrl": "https://dermalogica.com/products/daily-brightness-boosters-kit",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Kits/daily-brightness-boosters.png",
      "name_original": "daily brightness boosters kit",
      "mediumDescription": "Brighten, condition and hydrate your skin for a radiance boost.",
      "tagline": "glycolic + vitamin c",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 50587,
      "name": "clarifying bacne spray",
      "priceCurrency": "$",
      "price": "28.00",
      "shopify_id": "7593419505816",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "42526970347672",
          "default": true,
          "default_fallback": false,
          "size": "6 fl oz",
          "price": "28.00",
          "sku": "111478"
        }
      ],
      "productUrl": "https://dermalogica.com/products/clarifying-bacne-spray",
      "imageUrl": "https://dermalogica.widen.net/content/fh2olz3mxo/jpeg/clarifying%20bacne%20spray_DOM%20transparent.jpeg",
      "name_original": "clarifying bacne spray",
      "mediumDescription": "This blemish-fighting body spray acts fast on those hard-to-reach areas to help clear clogged pores and minimize future body breakouts",
      "tagline": "blemish-fighting body spray",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 14300,
      "name": "melting moisture masque",
      "priceCurrency": "$",
      "price": "69.00",
      "shopify_id": "6184201486488",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "38129282580632",
          "default": true,
          "default_fallback": false,
          "size": "1.7 oz",
          "price": "69.00",
          "sku": "111422"
        }
      ],
      "productUrl": "https://dermalogica.com/products/melting-moisture-masque",
      "imageUrl": "https://dioxide.s3-us-west-1.amazonaws.com/Product_Images_Full_Size/Melting_Moisture_Masque.png",
      "name_original": "melting moisture masque",
      "mediumDescription": "Extremely moisturizing masque elegantly transforms from balm to oil to help restore dry skin.",
      "tagline": "extreme moisture for dry skin",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9287,
      "name": "blackhead clearing fizz mask",
      "priceCurrency": "$",
      "price": "27.00",
      "shopify_id": "5710544634008",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070263292056",
          "default": true,
          "default_fallback": false,
          "size": "1.7 oz",
          "price": "27.00",
          "sku": "111287"
        }
      ],
      "productUrl": "https://dermalogica.com/products/blackhead-clearing-fizz-mask",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/50-ml-1.7-oz-blackhead-clearing-fizz-mask_Size.jpg",
      "name_original": "blackhead clearing fizz mask",
      "mediumDescription": "A unique mask that transforms into an active fizzing formula to clear pores and target blackheads.",
      "tagline": "targets blackheads before they turn into breakouts",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 19664,
      "name": "post-breakout fix",
      "priceCurrency": "$",
      "price": "25.00",
      "shopify_id": "7143983841432",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "41375610994840",
          "default": true,
          "default_fallback": false,
          "size": "0.5 oz",
          "price": "25.00",
          "sku": "111447"
        }
      ],
      "productUrl": "https://dermalogica.com/products/post-breakout-fix",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/post-breakout_fix.png",
      "name_original": "post-breakout fix",
      "mediumDescription": "A gel-cream spot treatment that helps brighten and fade Post-Inflammatory Hyperpigmentation (PIH) from breakouts while helping to nourish and restore skin's appearance.",
      "tagline": "fades post-breakout marks",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 40647,
      "name": "micellar prebiotic precleanse",
      "priceCurrency": "$",
      "price": "49.00",
      "shopify_id": "7445199028376",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "42209933426840",
          "default": true,
          "default_fallback": false,
          "size": "5.1 oz",
          "price": "49.00",
          "sku": "111458"
        }
      ],
      "productUrl": "https://dermalogica.com/products/micellar-prebiotic-precleanse",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/Micellar_Prebiotic_Precleanse.png",
      "name_original": "micellar prebiotic precleanse",
      "mediumDescription": "Nourishing micellar milky precleanse lifts away dirt, oil, and make-up while helping balance skin’s microbiome with prebiotics.",
      "tagline": "Nourishing micellar milky precleanse lifts away dirt, oil, and make-up while helping balance skin’s microbiome with prebiotics.",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 16060,
      "name": "powerbright dark spot serum",
      "priceCurrency": "$",
      "price": "99.00",
      "shopify_id": "6270515052696",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "38129288347800",
          "default": true,
          "default_fallback": false,
          "size": "1.0 oz",
          "price": "99.00",
          "sku": "111404"
        }
      ],
      "productUrl": "https://dermalogica.com/products/powerbright-dark-spot-serum",
      "imageUrl": "https://dioxide.s3-us-west-1.amazonaws.com/Product_Images_Full_Size/Powerbright_Dark_Spot_Serum.png",
      "name_original": "powerbright dark spot serum",
      "mediumDescription": "Start fading the appearance of dark spots within days: advanced serum begins to diminish the appearance of uneven pigmentation fast, and keeps working to even skin tone over time.",
      "tagline": "fades spots fast",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 56634,
      "name": "powerbright dark spot peel",
      "priceCurrency": "$",
      "price": "84.00",
      "shopify_id": "7884182814872",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "43249795530904",
          "default": true,
          "default_fallback": false,
          "size": "1.7 oz",
          "price": "84.00",
          "sku": "111481"
        },
        {
          "shopify_variant_id": "43834871808152",
          "default": false,
          "default_fallback": false,
          "size": "KIT",
          "price": "145.00",
          "sku": "111484"
        }
      ],
      "productUrl": "https://dermalogica.com/products/powerbright-dark-spot-peel",
      "imageUrl": "https://dermalogica.widen.net/content/jqkdmzvcyq/jpeg/9x16_powerbright_product_18.tif",
      "name_original": "powerbright dark spot peel",
      "mediumDescription": "At-home peel helps visibly lift surface hyperpigmentation – including UV-induced dark spots, post-blemish marks, and melasma.",
      "tagline": "visibly lifts hyperpigmentation",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 27651,
      "name": "daily milkfoliant",
      "priceCurrency": "$",
      "price": "19.50",
      "shopify_id": "7200921878680",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "41559246799000",
          "default": false,
          "default_fallback": false,
          "size": "2.6 oz",
          "price": "69.00",
          "sku": "111453"
        },
        {
          "shopify_variant_id": "41559246831768",
          "default": true,
          "default_fallback": false,
          "size": "0.45 oz",
          "price": "19.50",
          "sku": "411453"
        }
      ],
      "productUrl": "https://dermalogica.com/products/daily-milkfoliant",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/2.6oz-74g-daily-milkfoliant.png",
      "name_original": "daily milkfoliant",
      "mediumDescription": "Calming vegan exfoliating powder polishes skin while supporting the skin’s moisture barrier.",
      "tagline": "calming oat-based powder exfoliant",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 31497,
      "name": "liquid peelfoliant",
      "priceCurrency": "$",
      "price": "69.00",
      "shopify_id": "7605059420312",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "42557783507096",
          "default": true,
          "default_fallback": false,
          "size": "2.0 oz",
          "price": "69.00",
          "sku": "111461"
        }
      ],
      "productUrl": "https://dermalogica.com/products/liquid-peelfoliant",
      "imageUrl": "https://dermalogica.widen.net/content/7ji9ogajli/jpeg/Liquid%20Peelfoliant%20-%20front%20of%20bottle.jpeg",
      "name_original": "liquid peelfoliant",
      "mediumDescription": "Daily peel with a potent blend of acids and enzymes smooths the appearance of fine lines while helping to visibly minimize pores and even skin tone.",
      "tagline": "smooths + unclogs + evens tone",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 45349,
      "name": "porescreen spf40",
      "priceCurrency": "$",
      "price": "55.00",
      "shopify_id": "7605034254488",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "42557791273112",
          "default": true,
          "default_fallback": false,
          "size": "1.0 oz",
          "price": "55.00",
          "sku": "111473"
        }
      ],
      "productUrl": "https://dermalogica.com/products/porescreen-spf40",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/porescreen_spf40.png",
      "name_original": "porescreen spf40",
      "mediumDescription": "Multitasking sunscreen delivers SPF 40 protection, while supporting healthy-looking pores, minimizing their appearance with a blurring, primer-like effect, and enhancing skin tone with a hint of tint for radiant skin.",
      "tagline": "blurring + hint of tint mineral sunscreen",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9197,
      "name": "body hydrating cream",
      "priceCurrency": "$",
      "price": "39.00",
      "shopify_id": "5710541357208",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070251233432",
          "default": true,
          "default_fallback": false,
          "size": "10.0 oz",
          "price": "39.00",
          "sku": "111386"
        }
      ],
      "productUrl": "https://dermalogica.com/products/body-hydrating-cream",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/295-ml-10-oz-body-hydrating-cream_Size.png",
      "name_original": "body hydrating cream",
      "mediumDescription": "An advanced body cream with hydroxy acids and essential plant oils to smooth and condition skin.",
      "tagline": "conditioning skin smoother",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 10686,
      "name": "prisma protect spf30",
      "priceCurrency": "$",
      "price": "69.00",
      "shopify_id": "5710549909656",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070277841048",
          "default": true,
          "default_fallback": false,
          "size": "1.7 oz",
          "price": "69.00",
          "sku": "111330"
        }
      ],
      "productUrl": "https://dermalogica.com/products/prisma-protect-spf30",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/Prisma_Protect_SPF30.png",
      "name_original": "prisma protect spf30",
      "mediumDescription": "Light-activated multitasking moisturizer provides broad spectrum defense while preventing future signs of skin damage. Intelligent drone technology is activated by visible light to help boost skin’s natural luminosity.",
      "tagline": "light-activated skin defense",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 62528,
      "name": "powerbright dark spot system",
      "priceCurrency": "$",
      "price": "145.00",
      "shopify_id": "7884195135640",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "43249843011736",
          "default": true,
          "default_fallback": false,
          "size": "KIT",
          "price": "145.00",
          "sku": "111484"
        }
      ],
      "productUrl": "https://dermalogica.com/products/powerbright-dark-spot-system",
      "imageUrl": "https://dermalogica.widen.net/content/dklljmorcq/png/2023%20PowerBright%20Dark%20Spot%20System_COMBO%201.png",
      "name_original": "powerbright dark spot system",
      "mediumDescription": "2-step system visibly lifts + fades dark spots",
      "tagline": "2-step system visibly lifts + fades dark spots",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9125,
      "name": "special cleansing gel",
      "priceCurrency": "$",
      "price": "15.00",
      "shopify_id": "5710551744664",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070283280536",
          "default": true,
          "default_fallback": false,
          "size": "1.7 oz",
          "price": "15.00",
          "sku": "101102"
        },
        {
          "shopify_variant_id": "36070283215000",
          "default": false,
          "default_fallback": false,
          "size": "8.4 oz",
          "price": "48.00",
          "sku": "101104"
        },
        {
          "shopify_variant_id": "36070283247768",
          "default": false,
          "default_fallback": false,
          "size": "16.9 oz",
          "price": "72.00",
          "sku": "101106"
        },
        {
          "shopify_variant_id": "42500245160088",
          "default": false,
          "default_fallback": false,
          "size": "16.9 oz - refill",
          "price": "64.00",
          "sku": "111463"
        }
      ],
      "productUrl": "https://dermalogica.com/products/special-cleansing-gel",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/250-ml-8.4-oz-special-cleansing-gel_Size.png",
      "name_original": "special cleansing gel",
      "mediumDescription": "Refreshing lather thoroughly removes impurities, without disturbing the skin’s natural moisture balance. This iconic cleanser, which contains naturally-foaming Quillaja Saponaria, gently rinses away toxins and debris to leave skin feeling smooth and clean.",
      "tagline": "gentle foaming cleanser",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9289,
      "name": "breakout clearing booster",
      "priceCurrency": "$",
      "price": "27.00",
      "shopify_id": "5710544830616",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070263980184",
          "default": true,
          "default_fallback": false,
          "size": "1.0 oz",
          "price": "27.00",
          "sku": "111286"
        }
      ],
      "productUrl": "https://dermalogica.com/products/breakout-clearing-booster",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/30-ml-1-oz-breakout-clearing-booster_Size.png",
      "name_original": "breakout clearing booster",
      "mediumDescription": "This fast-acting booster, formulated with Salicylic Acid, helps combat breakout-causing bacteria for rapid skin clearing. Patented TT Technology and Phytoplankton Extract work with skin’s natural microbiome and help prevent over-drying.",
      "tagline": "stop breakouts in their tracks",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 13136,
      "name": "hydro masque exfoliant",
      "priceCurrency": "$",
      "price": "64.00",
      "shopify_id": "5710540046488",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070248480920",
          "default": true,
          "default_fallback": false,
          "size": "1.7 oz",
          "price": "64.00",
          "sku": "111418"
        }
      ],
      "productUrl": "https://dermalogica.com/products/hydro-masque-exfoliant",
      "imageUrl": "https://dioxide.s3-us-west-1.amazonaws.com/Product_Images_Full_Size/Hydro_Masque_Exfoliant.png",
      "name_original": "hydro masque exfoliant",
      "mediumDescription": "Hydrating and exfoliating five-minute masque smoothes and renews for luminous, healthy-looking skin.",
      "tagline": "exfoliating and hydrating masque",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 16373,
      "name": "powerbright overnight cream",
      "priceCurrency": "$",
      "price": "89.00",
      "shopify_id": "6285576863896",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "38178904408216",
          "default": true,
          "default_fallback": false,
          "size": "1.7 oz",
          "price": "89.00",
          "sku": "111012"
        }
      ],
      "productUrl": "https://dermalogica.com/products/powerbright-overnight-cream",
      "imageUrl": "https://dioxide.s3-us-west-1.amazonaws.com/Product_Images_Full_Size/Powerbright_Overnight_Cream.png",
      "name_original": "powerbright overnight cream",
      "mediumDescription": "Fade dark spots while you sleep: nourishing nighttime cream optimizes skin moisture recovery and helps restore luminosity.",
      "tagline": "nightly dark spot fader",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 63870,
      "name": "biolumin-c night restore",
      "priceCurrency": "$",
      "price": "99.00",
      "shopify_id": "8345160908952",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "45046491414680",
          "default": true,
          "default_fallback": false,
          "size": "0.85 oz",
          "price": "99.00",
          "sku": "111504"
        }
      ],
      "productUrl": "https://dermalogica.com/products/biolumin-c-night-restore-serum",
      "imageUrl": "https://dermalogica.widen.net/content/ictpzplnfx/jpeg/BLC%20Night%20Restore%20Serum_FRONT.jpeg",
      "name_original": "biolumin-c night restore",
      "mediumDescription": "An overnight serum with ultra-stable Vitamin C and Pro-Vitamin D Complex helps restore moisture barrier for long-lasting radiance, hydration, and improved tone.",
      "tagline": "vitamin c + pro-vitamin d serum",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9211,
      "name": "ultracalming serum concentrate",
      "priceCurrency": "$",
      "price": "69.00",
      "shopify_id": "5710549221528",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070276399256",
          "default": true,
          "default_fallback": false,
          "size": "1.3 oz",
          "price": "69.00",
          "sku": "110997"
        }
      ],
      "productUrl": "https://dermalogica.com/products/ultracalming-serum-concentrate",
      "imageUrl": "https://dioxide.s3-us-west-1.amazonaws.com/Product_Images_Full_Size/40-ml-1.3-oz-ultracalming-serum-concentrate_Size.png",
      "name_original": "ultracalming serum concentrate",
      "mediumDescription": "The solution for skin sensitivity. This super-concentrated serum helps calm, restore and defend sensitized skin.",
      "tagline": "sensitized skin antidote",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9168,
      "name": "skin resurfacing cleanser",
      "priceCurrency": "$",
      "price": "49.00",
      "shopify_id": "5710542176408",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "41606935642264",
          "default": true,
          "default_fallback": false,
          "size": "5.1 oz",
          "price": "49.00",
          "sku": "101511"
        }
      ],
      "productUrl": "https://dermalogica.com/products/skin-resurfacing-cleanser",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/150-ml-5.1-oz-skin-resurfacing-cleanser_Size.png",
      "name_original": "skin resurfacing cleanser",
      "mediumDescription": "Dual-action exfoliating cleanser helps retexturize aging skin. Achieve smooth, ultra-clean skin with this highly-active, two-in-one cleanser and exfoliant.",
      "tagline": "smoothing exfoliating cleanser",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 17077,
      "name": "smart response serum",
      "priceCurrency": "$",
      "price": "154.00",
      "shopify_id": "6624376094872",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "42715744010392",
          "default": false,
          "default_fallback": false,
          "size": "2.0 oz",
          "price": "225.00",
          "sku": "811432"
        },
        {
          "shopify_variant_id": "39570802016408",
          "default": true,
          "default_fallback": false,
          "size": "1.0 oz",
          "price": "154.00",
          "sku": "111432"
        }
      ],
      "productUrl": "https://dermalogica.com/products/smart-response-serum",
      "imageUrl": "https://dioxide.s3-us-west-1.amazonaws.com/Product_Images_Full_Size/smart_response_serum.png",
      "name_original": "smart response serum",
      "mediumDescription": "Next-gen smart serum delivers what your skin needs, when it needs it.",
      "tagline": "responds to skin’s changing needs",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9174,
      "name": "multivitamin power recovery masque",
      "priceCurrency": "$",
      "price": "18.00",
      "shopify_id": "5710541914264",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070253723800",
          "default": true,
          "default_fallback": false,
          "size": "0.5 oz",
          "price": "18.00",
          "sku": "410716"
        },
        {
          "shopify_variant_id": "36070253691032",
          "default": false,
          "default_fallback": false,
          "size": "2.5 oz",
          "price": "69.00",
          "sku": "110716"
        }
      ],
      "productUrl": "https://dermalogica.com/products/multivitamin-power-recovery-masque",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/75-ml-2.5-oz-multivitamin-power-recovery-masque_Size.png",
      "name_original": "multivitamin power recovery masque",
      "mediumDescription": "Ultra-replenishing creamy masque helps rescue stressed, aging skin.",
      "tagline": "nutrient-rich rescue masque",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 26657,
      "name": "hyaluronic ceramide mist",
      "priceCurrency": "$",
      "price": "49.00",
      "shopify_id": "7181191413912",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "41484255232152",
          "default": true,
          "default_fallback": false,
          "size": "5.1 oz",
          "price": "49.00",
          "sku": "111452"
        }
      ],
      "productUrl": "https://dermalogica.com/products/hyaluronic-ceramide-mist",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/Hyaluronic_Ceramide_Mist.png",
      "name_original": "hyaluronic ceramide mist",
      "mediumDescription": "Saturate skin with hydration and lock in moisture to help it bounce back: long-lasting hydrating Hyaluronic Acid and ceramide mist helps to smooth fine lines and strengthen skin's barrier.",
      "tagline": "hydrates + strengthens skin's barrier",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 11322,
      "name": "biolumin-c eye serum",
      "priceCurrency": "$",
      "price": "78.00",
      "shopify_id": "5710543454360",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070260310168",
          "default": true,
          "default_fallback": false,
          "size": "0.5 oz",
          "price": "78.00",
          "sku": "111393"
        }
      ],
      "productUrl": "https://dermalogica.com/products/biolumin-c-eye-serum",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/BioLumin-C+Eye+Serum+-+Cap+On.png",
      "name_original": "biolumin-c eye serum",
      "mediumDescription": "Lightweight serum delivers a highly-bioavailable Vitamin C complex to the skin around the eyes to dramatically brighten and visibly firm.",
      "tagline": "brightening vitamin c eye serum",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 19444,
      "name": "daily glycolic cleanser",
      "priceCurrency": "$",
      "price": "39.00",
      "shopify_id": "6908040970392",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "40531714506904",
          "default": true,
          "default_fallback": false,
          "size": "5.1 oz",
          "price": "39.00",
          "sku": "111439"
        },
        {
          "shopify_variant_id": "40915138936984",
          "default": false,
          "default_fallback": false,
          "size": "10.0 oz",
          "price": "63.00",
          "sku": "811439"
        }
      ],
      "productUrl": "https://dermalogica.com/products/daily-glycolic-cleanser",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/Daily_Glycolic_Cleanser.png",
      "name_original": "daily glycolic cleanser",
      "mediumDescription": "Brightening and conditioning cleanser renews dull, uneven skin tone and helps remove buildup caused by environmental factors.",
      "tagline": "brightening cleanser",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 11350,
      "name": "invisible physical defense spf30",
      "priceCurrency": "$",
      "price": "49.00",
      "shopify_id": "5710542995608",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070259261592",
          "default": true,
          "default_fallback": false,
          "size": "1.7 oz",
          "price": "49.00",
          "sku": "111412"
        }
      ],
      "productUrl": "https://dermalogica.com/products/invisible-physical-defense-spf30",
      "imageUrl": "https://dioxide.s3-us-west-1.amazonaws.com/Product_Images_Full_Size/Ecomm+Hero+-+Invisible+Physical+Defense.png",
      "name_original": "invisible physical defense spf30",
      "mediumDescription": "Invisible, weightless defense that blends easily on skin, featuring only non-nano Zinc Oxide.",
      "tagline": "weightless physical sunscreen",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9131,
      "name": "multi-active toner",
      "priceCurrency": "$",
      "price": "15.00",
      "shopify_id": "5710541553816",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070252773528",
          "default": true,
          "default_fallback": false,
          "size": "1.7 oz",
          "price": "15.00",
          "sku": "110615"
        },
        {
          "shopify_variant_id": "36070252740760",
          "default": false,
          "default_fallback": false,
          "size": "8.4 oz",
          "price": "47.00",
          "sku": "110616"
        }
      ],
      "productUrl": "https://dermalogica.com/products/multi-active-toner",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/250-ml-8.4-oz-multi-active-toner_Size.png",
      "name_original": "multi-active toner",
      "mediumDescription": "Light facial toner spray hydrates and refreshes. Help condition the skin and prepare for proper moisture absorption when you spritz over skin after cleansing, and before applying your prescribed Dermalogica Moisturizer.",
      "tagline": "hydrating, refreshing spritz",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 22900,
      "name": "discover healthy skin kit",
      "priceCurrency": "$",
      "price": "39.50",
      "shopify_id": "5710540472472",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "42268081881240",
          "default": true,
          "default_fallback": false,
          "size": "KIT",
          "price": "39.50",
          "sku": "111370"
        }
      ],
      "productUrl": "https://dermalogica.com/products/discover-healthy-skin-kit",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Kits/discover-healthy-skin-kit.png",
      "name_original": "discover healthy skin kit",
      "mediumDescription": "The perfect introduction to Dermalogica, this special collection of our favorite and most popular products is a complete regimen for all skin types.",
      "tagline": "our top sellers for healthy, glowing skin",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 46592,
      "name": "stabilizing repair cream",
      "priceCurrency": "$",
      "price": "25.00",
      "shopify_id": "7651058811032",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "42708180009112",
          "default": false,
          "default_fallback": false,
          "size": "1.7 oz",
          "price": "69.00",
          "sku": "111476"
        },
        {
          "shopify_variant_id": "45595832156312",
          "default": false,
          "default_fallback": false,
          "size": "3.4 oz",
          "price": "99.00",
          "sku": "811476"
        },
        {
          "shopify_variant_id": "45595833368728",
          "default": true,
          "default_fallback": false,
          "size": "0.5 oz",
          "price": "25.00",
          "sku": "411476"
        }
      ],
      "productUrl": "https://dermalogica.com/products/stabilizing-repair-cream",
      "imageUrl": "https://dermalogica.widen.net/content/gubtqjzdhk/jpeg/Primary-Stabilizing-Repair-Cream-Front",
      "name_original": "stabilizing repair cream",
      "mediumDescription": "Ultra-soothing actives calm on contact – helping to break the pattern of sensitive skin as they immediately comfort and help skin become more resilient over time.  Balmy-cream formula melts into skin to quickly alleviate redness and help prevent future irritation.",
      "tagline": "barrier-repairing moisturizer",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9299,
      "name": "rapid reveal peel",
      "priceCurrency": "$",
      "price": "79.00",
      "shopify_id": "5710544470168",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070262800536",
          "default": true,
          "default_fallback": false,
          "size": "10 x 3ml tubes",
          "price": "79.00",
          "sku": "111291"
        }
      ],
      "productUrl": "https://dermalogica.com/products/rapid-reveal-peel",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/3-ml-x-10-0.1-oz-rapid-reveal-peel_Size.png",
      "name_original": "rapid reveal peel",
      "mediumDescription": "A professional-grade at-home peel that helps reveal brighter, healthier skin in just minutes a week. This maximum-strength exfoliant delivers powerful results with no downtime using a unique complex of phytoactive AHA extracts, Lactic Acid and fermented plant enzymes to help reveal new, firmer skin.",
      "tagline": "professional-grade at-home peel",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 14096,
      "name": "micro-pore mist",
      "priceCurrency": "$",
      "price": "25.00",
      "shopify_id": "5759778324632",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36267584815256",
          "default": true,
          "default_fallback": false,
          "size": "4.0 oz",
          "price": "25.00",
          "sku": "111430"
        }
      ],
      "productUrl": "https://dermalogica.com/products/micro-pore-mist-toner",
      "imageUrl": "https://dioxide.s3-us-west-1.amazonaws.com/Product_Images_Full_Size/Micro-pore_mist.jpg",
      "name_original": "micro-pore mist",
      "mediumDescription": "Minimize visible pores, reduce excess oil, and help diminish the appearance of post-breakout marks.",
      "tagline": "minimize visible pores",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 70836,
      "name": "dynamic skin strengthening serum",
      "priceCurrency": "$",
      "price": "92.00",
      "shopify_id": "8612754489496",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "45876870250648",
          "default": true,
          "default_fallback": false,
          "size": "1.0 oz",
          "price": "92.00",
          "sku": "111508"
        }
      ],
      "productUrl": "https://dermalogica.com/products/dynamic-skin-strengthening-serum",
      "imageUrl": "https://dermalogica.widen.net/content/rbhmrzp06t/png/DynamicSkin_Strengthening_Serum_Retail_1OZ_Primary_FRONT.png",
      "name_original": "dynamic skin strengthening serum",
      "mediumDescription": "Age stronger: nourishing multi-defense serum builds barrier resilience to enhance endurance against skin-aging triggers.",
      "tagline": "barrier resilience + age resistance",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9180,
      "name": "super rich repair",
      "priceCurrency": "$",
      "price": "36.00",
      "shopify_id": "5710541815960",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070253133976",
          "default": false,
          "default_fallback": false,
          "size": "1.7 oz",
          "price": "98.00",
          "sku": "111063"
        },
        {
          "shopify_variant_id": "45345192444056",
          "default": true,
          "default_fallback": false,
          "size": "0.5 oz",
          "price": "36.00",
          "sku": "411063"
        },
        {
          "shopify_variant_id": "40366074822808",
          "default": false,
          "default_fallback": false,
          "size": "3.4 oz",
          "price": "149.00",
          "sku": "811063"
        }
      ],
      "productUrl": "https://dermalogica.com/products/super-rich-repair",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/50-ml-1.7-oz-super-rich-repair_Size.png",
      "name_original": "super rich repair",
      "mediumDescription": "Deeply nourishing skin treatment cream for chronically dry, dehydrated skin. Heavyweight cream helps replenish skin’s natural moisture levels to defend against environmental assaults. Powerful peptides and an acid-free smoothing complex support skin resilience and tone.",
      "tagline": "super-concentrated moisturizer",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 78252,
      "name": "magnetic[+] afterglow cleanser",
      "priceCurrency": "$",
      "price": "49.00",
      "shopify_id": "8693066334360",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "46455724966040",
          "default": true,
          "default_fallback": false,
          "size": "5.1 oz",
          "price": "49.00",
          "sku": "111512"
        },
        {
          "shopify_variant_id": "46455724998808",
          "default": false,
          "default_fallback": false,
          "size": "10.0 oz",
          "price": "79.00",
          "sku": "811512"
        }
      ],
      "productUrl": "https://dermalogica.com/products/magnetic-afterglow-cleanser",
      "imageUrl": "https://dermalogica.widen.net/content/fou1b4vasy/png/Magnetic%5B%2B%5D%20Afterglow%20Cleanser_BTL_10oz_FRONT.png?color=ffffff00&u=dzfqbh&w=2048&h=2048&position=c&crop=false",
      "name_original": "magnetic[+] afterglow cleanser",
      "mediumDescription": "Creamy cleanser with Cellular HydraBond Technology delivers glowing skin.",
      "tagline": "moisture-bonding cleansing cream",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 74595,
      "name": "dynamic skin sculptor",
      "priceCurrency": "$",
      "price": "98.00",
      "shopify_id": "8699579695256",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "46489951404184",
          "default": true,
          "default_fallback": false,
          "size": "5.1 oz",
          "price": "98.00",
          "sku": "111511"
        }
      ],
      "productUrl": "https://dermalogica.com/products/dynamic-skin-sculptor",
      "imageUrl": "https://dermalogica.widen.net/content/sujzgi8ro1/png/Dynamic%20Skin%20Sculptor_BTL_5.1oz_FRONT.png?color=ffffff00&u=dzfqbh&w=2048&h=2048&position=c&crop=false",
      "name_original": "dynamic skin sculptor",
      "mediumDescription": "Sculpting body serum with Pro-NAD+ Complex visibly firms and lifts skin.",
      "tagline": "toning + tightening body serum",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9213,
      "name": "age reversal eye complex",
      "priceCurrency": "$",
      "price": "84.00",
      "shopify_id": "5710549090456",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070275973272",
          "default": true,
          "default_fallback": false,
          "size": "0.5 oz",
          "price": "84.00",
          "sku": "111236"
        }
      ],
      "productUrl": "https://dermalogica.com/products/age-reversal-eye-complex",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/15-ml-0.5-oz-age-reversal-eye-complex_Size.png",
      "name_original": "age reversal eye complex",
      "mediumDescription": "Advanced, microencapsulated Retinol helps smooth away the signs of skin aging around the eyes.",
      "tagline": "potent retinol cream",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 50126,
      "name": "deep acne liquid patch",
      "priceCurrency": "$",
      "price": "35.00",
      "shopify_id": "7610318389400",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "42572941852824",
          "default": true,
          "default_fallback": false,
          "size": "0.5 oz",
          "price": "35.00",
          "sku": "111462"
        }
      ],
      "productUrl": "https://dermalogica.com/products/deep-acne-liquid-patch",
      "imageUrl": "https://dermalogica.widen.net/s/qvrgjxqzmq/9x16-deep-acne-liquid-patch-hero-1",
      "name_original": "deep acne liquid patch",
      "mediumDescription": "Invisible sulfur-based spot treatment transforms from liquid to patch to soothe, clear, and help prevent future breakouts",
      "tagline": "invisible acne treatment",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9277,
      "name": "calm water gel",
      "priceCurrency": "$",
      "price": "59.00",
      "shopify_id": "5710545223832",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070264864920",
          "default": true,
          "default_fallback": false,
          "size": "1.7 oz",
          "price": "59.00",
          "sku": "111268"
        }
      ],
      "productUrl": "https://dermalogica.com/products/calm-water-gel",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/50-ml-1.7-oz-calm-water-gel_Size.png",
      "name_original": "calm water gel",
      "mediumDescription": "Weightless water-gel moisturizer hydrates dry, sensitive skin. Refreshing gel formula transforms into a skin-quenching fluid upon application, forming a weightless barrier against environmental assault.",
      "tagline": "weightless water-gel moisturizer",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 25079,
      "name": "circular hydration serum",
      "priceCurrency": "$",
      "price": "28.00",
      "shopify_id": "7181179027608",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "44648311619736",
          "default": true,
          "default_fallback": false,
          "size": "0.34 oz",
          "price": "28.00",
          "sku": "411455"
        },
        {
          "shopify_variant_id": "41484270305432",
          "default": false,
          "default_fallback": false,
          "size": "1.0 oz",
          "price": "69.00",
          "sku": "111455"
        },
        {
          "shopify_variant_id": "45345141522584",
          "default": false,
          "default_fallback": false,
          "size": "2.0 oz",
          "price": "99.00",
          "sku": "811455"
        }
      ],
      "productUrl": "https://dermalogica.com/products/circular-hydration-serum",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/Circular_Hydration_Serum.png",
      "name_original": "circular hydration serum",
      "mediumDescription": "Kick-start skin’s hydration cycle: long-lasting serum immediately floods skin with hydration, replenishes from within, and helps prevent future hydration evaporation.",
      "tagline": "long-lasting hydration",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 15436,
      "name": "clear and brighten kit",
      "priceCurrency": "$",
      "price": "49.50",
      "shopify_id": "7571579437208",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "42465447575704",
          "default": true,
          "default_fallback": false,
          "size": "KIT",
          "price": "49.50",
          "sku": "111472"
        }
      ],
      "productUrl": "https://dermalogica.com/products/clear-brighten-kit",
      "imageUrl": "https://dermalogica.widen.net/content/w3yvll9tbd/jpeg/Clear%20%26%20Brighten%20Kit_SLEEVE_ANGLED.tif",
      "name_original": "clear and brighten kit",
      "mediumDescription": "This kit contains highly active ingredients to clear breakouts, smooth skin and brighten skin tone.",
      "tagline": "target breakouts + premature skin aging",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 37520,
      "name": "dynamic skin retinol serum",
      "priceCurrency": "$",
      "price": "39.00",
      "shopify_id": "7445212987544",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "45345174028440",
          "default": true,
          "default_fallback": false,
          "size": "0.34 oz",
          "price": "39.00",
          "sku": "411465"
        },
        {
          "shopify_variant_id": "42210000765080",
          "default": false,
          "default_fallback": false,
          "size": "1.0 oz",
          "price": "98.00",
          "sku": "111465"
        }
      ],
      "productUrl": "https://dermalogica.com/products/dynamic-skin-retinol-serum",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/Dynamic_Skin_Retinol_Serum.png",
      "name_original": "dynamic skin retinol serum",
      "mediumDescription": "3.5% high-dose retinol serum visibly reduces the 4 signs of skin aging in just 2 weeks .",
      "tagline": "high-dose wrinkle serum",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 72603,
      "name": "acne biotic moisturizer",
      "priceCurrency": "$",
      "price": "76.00",
      "shopify_id": "8713521004696",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "46560384778392",
          "default": true,
          "default_fallback": false,
          "size": "1.7 oz",
          "price": "76.00",
          "sku": "111506"
        }
      ],
      "productUrl": "https://dermalogica.com/products/acne-biotic-moisturizer",
      "imageUrl": "https://dermalogica.widen.net/content/o4v05pqbj4/png/Breakout%20Biotic%20Moisturize%20INT%20Front.png?color=ffffff00&u=dzfqbh&w=2048&h=2048&position=c&crop=false",
      "name_original": "breakout biotic moisturizer",
      "mediumDescription": "Daily gel targets and helps prevent breakouts. Lightweight moisturizer with a prebiotic blend hydrates and supports skin’s microbiome, helping aging skin look healthier, now and in the future.",
      "tagline": "daily breakout-targeting gel",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 15520,
      "name": "cooling aqua jelly",
      "priceCurrency": "$",
      "price": "26.00",
      "shopify_id": "6157012631704",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "37671486095512",
          "default": true,
          "default_fallback": false,
          "size": "2.0 oz",
          "price": "26.00",
          "sku": "111431"
        }
      ],
      "productUrl": "https://dermalogica.com/products/cooling-aqua-jelly",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/cooling_aqua_jelly.png",
      "name_original": "cooling aqua jelly",
      "mediumDescription": "A moisturizer for oily skin that gives all the dewy glow, none of the shine",
      "tagline": "weightless hydration for oily skin",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9160,
      "name": "clearing skin wash",
      "priceCurrency": "$",
      "price": "48.00",
      "shopify_id": "5710542635160",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070258540696",
          "default": true,
          "default_fallback": false,
          "size": "8.4 oz",
          "price": "48.00",
          "sku": "111345"
        },
        {
          "shopify_variant_id": "36070258573464",
          "default": false,
          "default_fallback": false,
          "size": "16.9 oz",
          "price": "72.00",
          "sku": "111347"
        }
      ],
      "productUrl": "https://dermalogica.com/products/clearing-skin-wash",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/clearing-skin-wash.png",
      "name_original": "clearing skin wash",
      "mediumDescription": "A foaming cleanser that is the perfect start to around-the-clock control of breakouts, comedones and excess surface oils.",
      "tagline": "breakout clearing foam",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9170,
      "name": "antioxidant hydramist",
      "priceCurrency": "$",
      "price": "15.00",
      "shopify_id": "5710542078104",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070254379160",
          "default": true,
          "default_fallback": false,
          "size": "1.0 oz",
          "price": "15.00",
          "sku": "102022"
        },
        {
          "shopify_variant_id": "36070254346392",
          "default": false,
          "default_fallback": false,
          "size": "5.1 oz",
          "price": "49.00",
          "sku": "102021"
        }
      ],
      "productUrl": "https://dermalogica.com/products/antioxidant-hydramist",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/150-ml-5.1-oz-antioxidant-hydramist_Size.png",
      "name_original": "antioxidant hydramist",
      "mediumDescription": "Refreshing antioxidant toner that helps firm and hydrate. Convenient mist-on formula supplements skin’s protective barrier by creating an active antioxidant shield to help fight free radical damage, and help prevent the signs of aging caused by Advanced Glycation End-products (AGEs) – a damaging by-product of sugar/protein reactions on the skin.",
      "tagline": "refreshing antioxidant shield",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 9129,
      "name": "ultracalming cleanser",
      "priceCurrency": "$",
      "price": "48.00",
      "shopify_id": "5710542471320",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "36070254968984",
          "default": false,
          "default_fallback": false,
          "size": "16.9 oz",
          "price": "72.00",
          "sku": "110542"
        },
        {
          "shopify_variant_id": "36070254936216",
          "default": true,
          "default_fallback": false,
          "size": "8.4 oz",
          "price": "48.00",
          "sku": "110541"
        }
      ],
      "productUrl": "https://dermalogica.com/products/ultracalming-cleanser",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/250-ml-8.4-oz-ultracalming-cleanser_Size.png",
      "name_original": "ultracalming cleanser",
      "mediumDescription": "Gentle cleansing gel/cream for reactive skin. This pH-balanced, non-foaming cleanser helps calm and cool reactive, sensitized or overprocessed skin.",
      "tagline": "gentle cleansing cream",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 45549,
      "name": "oil to foam total cleanser",
      "priceCurrency": "$",
      "price": "59.00",
      "shopify_id": "7610306560152",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "42572918882456",
          "default": true,
          "default_fallback": false,
          "size": "8.4 oz",
          "price": "59.00",
          "sku": "111474"
        }
      ],
      "productUrl": "https://dermalogica.com/products/oil-to-foam-total-cleanser",
      "imageUrl": "https://dermalogica.widen.net/content/dozqigeyad/jpeg/1x1%20Oil%20to%20Foam%20Cleanser%20Product%20Stylized%205365.tif",
      "name_original": "oil to foam total cleanser",
      "mediumDescription": "Transformative oil to foam cleanser removes make-up, sunscreen, and debris while cleansing skin in one step for ultra-clean, healthy-looking skin.",
      "tagline": "all in one cleanser",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 58419,
      "name": "phyto nature lifting eye cream",
      "priceCurrency": "$",
      "price": "118.00",
      "shopify_id": "7933065887896",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "43467335106712",
          "default": true,
          "default_fallback": false,
          "size": "0.5 oz",
          "price": "118.00",
          "sku": "111487"
        }
      ],
      "productUrl": "https://dermalogica.com/products/phyto-nature-lifting-eye-cream",
      "imageUrl": "https://dermalogica.widen.net/content/jbifitn1lf/png/Phyto%20Nature%20Lifting%20Eye%20Cream_FRONT.png",
      "name_original": "phyto nature lifting eye cream",
      "mediumDescription": "Transformative eye cream with potent peptides and phytoactives delivers a more lifted look.",
      "tagline": "firming + lifting eye treatment",
      "bullet_points_desc": null,
      "translated_name": false,
      "translated_med_descr": false,
      "translated_tagline": false,
      "translated_bullet_point_desc": false
    },
    {
      "pimcore_id": 42889,
      "name": "phyto nature oxygen cream",
      "priceCurrency": "$",
      "price": "129.00",
      "shopify_id": "7574132654232",
      "available": true,
      "variants": [
        {
          "shopify_variant_id": "42474079158424",
          "default": true,
          "default_fallback": false,
          "size": "1.7 oz",
          "price": "129.00",
          "sku": "111466"
        }
      ],
      "productUrl": "https://dermalogica.com/products/phyto-nature-oxygen-cream",
      "imageUrl": "https://dioxide.s3.us-west-1.amazonaws.com/Product_Images_Full_Size/Oxygen_Liquid_Cream_Front.png",
      "name_original": "phyto nature oxygen cream",
      "mediumDescription": "Lightweight liquid cream moisturizer utilizes oxygen-optimizing phytoactives to breathe new life into aging skin.",
      "tagline": "firming + lifting liquid moisturizer",
      "bullet_points_desc": null,
      "translated_name": true,
      "translated_med_descr": true,
      "translated_tagline": true,
      "translated_bullet_point_desc": false
    }
  ],
  "processing_time": 0.09663701057434082
}
```
