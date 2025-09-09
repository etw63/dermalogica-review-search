#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Divergent Discovery v5 (OccasionMiner: lexicon-free)
- Keeps the 7 fixes from v4
- Replaces usage occasion logic with a fully unsupervised, data-driven miner
  that discovers novel usage occasions without any predefined lexicon.

Core idea:
  For a given product:
    1) Tight local sub-clustering of its reviews (UMAP + HDBSCAN)
    2) For each candidate cohort:
         a) Extract candidate keyphrases via c-TF-IDF (2-4 grams, no stopwords dropped)
         b) Score phrases by (cohort lift * log-odds vs product rest)
         c) Embed phrases and cluster them (phrase HDBSCAN) to form "occasion themes"
         d) Score each occasion theme by support, coverage, distinctiveness, coherence
         e) Emit up to 3 strongest themes with auto-generated titles + bullets + quotes

No hand-coded phrase lists; novelty emerges from the corpus.
"""

from __future__ import annotations
import re, json, math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from tqdm import tqdm
import umap
import hdbscan
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ---------------------------
# Config
# ---------------------------
@dataclass
class RunConfig:
    csv_file: str = "dermalogica_aggregated_reviews.csv"
    max_sample: Optional[int] = None
    model_name: str = "all-mpnet-base-v2"
    random_state: int = 42

# ---------------------------
# Main class
# ---------------------------
class DivergentDiscoveryV5:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.df = pd.DataFrame()
        self.embeddings = None
        self.model: Optional[SentenceTransformer] = None

    # ====== Pipeline ======
    def run(self) -> Dict:
        print("="*80)
        print("DIVERGENT DISCOVERY — v5 (OccasionMiner, lexicon-free)")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

        self._load_data(self.cfg.csv_file, self.cfg.max_sample)
        self._embed_and_debias()

        # Coarse + fine + outlier analysis kept if you need, but v5 focuses on the novel occasion miner per product.
        print("Ready. Call generate_novel_usage_occasions(product_name) to mine occasions.\n")
        return {"status": "ready", "reviews": int(len(self.df)), "products": int(self.df['product_name'].nunique())}

    # ====== Data ======
    def _load_data(self, path: str, max_sample: Optional[int]):
        print(f"Loading: {path}")
        df = pd.read_csv(path)
        if 'review_text' not in df or 'product_name' not in df:
            raise ValueError("CSV must include 'review_text' and 'product_name' columns.")
        df = df.dropna(subset=['review_text','product_name']).copy()

        df['clean_text'] = (df['review_text'].astype(str)
            .str.lower().str.replace(r"http[s]?://\S+", " ", regex=True)
            .str.replace(r"\S+@\S+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        df['product_name'] = df['product_name'].apply(self._clean_product)

        # length filter
        before = len(df)
        df = df[df['clean_text'].str.split().str.len() >= 8].copy()
        print(f"Filtered out {before - len(df):,} reviews (< 8 tokens)")

        # guards
        if 'rating' not in df: df['rating'] = np.nan
        if 'source' not in df: df['source'] = 'unknown'

        # sampling (optional stratified)
        if max_sample and len(df) > max_sample:
            df = self._stratified_sample(df, max_sample)
            print(f"Stratified down to {len(df):,}")

        print(f"Final dataset: {len(df):,}")
        print(f"Unique products: {df['product_name'].nunique()}")
        self.df = df.reset_index(drop=True)

    @staticmethod
    def _clean_product(name: str) -> str:
        s = str(name).strip()
        s = re.sub(r"(?i)^dermalogica\s+", "", s)
        s = s.lower()
        # lightly strip generic suffixes if superfluous
        for suf in [" moisturizer"," cleanser"," serum"," cream"," gel"," oil"," toner"," exfoliator"," mask"," treatment"," sunscreen"," spf 30"," spf 40"," spf 50"," kit"," set"]:
            if s.endswith(suf) and len(s) - len(suf) > 3:
                s = s[:-len(suf)].strip()
                break
        return s or "unknown"

    def _stratified_sample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        def bucket(r):
            if pd.isna(r): return "unknown"
            try:
                r = float(r)
                return "low" if r <= 2 else ("mid" if r <= 3 else "high")
            except: return "unknown"
        df = df.copy()
        df['rating_bucket'] = df['rating'].apply(bucket)
        df['ptype'] = df['product_name'].apply(lambda x: "spf" if "spf" in str(x) else ("cleanser" if "cleanser" in str(x) else "other"))
        groups = df.groupby(['source','rating_bucket','ptype'], dropna=False)
        parts = []
        for _, g in groups:
            k = max(1, int(n * len(g) / len(df)))
            k = min(k, len(g))
            parts.append(g.sample(n=k, random_state=self.cfg.random_state))
        return pd.concat(parts, ignore_index=True)

    # ====== Embeddings ======
    def _embed_and_debias(self):
        print("Loading model:", self.cfg.model_name)
        self.model = SentenceTransformer(self.cfg.model_name)
        X = self.model.encode(self.df['clean_text'].tolist(), batch_size=64, show_progress_bar=True, normalize_embeddings=False)
        X = np.asarray(X, dtype=np.float32)

        # Remove top component (TruncatedSVD(1)) then L2 normalize
        print("Debiasing with TruncatedSVD(1) and L2-normalizing...")
        svd = TruncatedSVD(n_components=1, random_state=self.cfg.random_state)
        proj = svd.fit_transform(X)[:,0]
        comp = svd.components_[0]
        X = X - np.outer(proj, comp).astype(np.float32)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        self.embeddings = X.astype(np.float32)
        print("Embeddings:", self.embeddings.shape)

    # ====== OccasionMiner (lexicon-free) ======
    def generate_novel_usage_occasions(self, product_name: str, max_occasions: int = 3) -> List[Dict]:
        """Return up to 3 novel usage occasions for a product without relying on a predefined lexicon."""
        dfp = self.df[self.df['product_name'].str.lower() == product_name.lower()].copy()
        if dfp.empty or len(dfp) < 25:
            return []

        idx_all = dfp.index.values
        Xp = self.embeddings[idx_all]
        texts = dfp['clean_text'].tolist()

        # 1) tight local sub-clustering
        u = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=10, metric='cosine',
                      random_state=self.cfg.random_state).fit_transform(Xp)
        h = hdbscan.HDBSCAN(min_cluster_size=max(12, int(0.05*len(dfp))), min_samples=10,
                            cluster_selection_method="leaf", metric="euclidean").fit(u)
        labels = h.labels_

        # optional: re-cluster product outliers
        extras = {}
        out_mask = labels == -1
        if out_mask.sum() >= 12:
            uo = u[out_mask]
            ho = hdbscan.HDBSCAN(min_cluster_size=max(10, int(0.04*out_mask.sum())),
                                 min_samples=5, cluster_selection_method="leaf", metric="euclidean").fit(uo)
            for oid in sorted(set(ho.labels_) - {-1}):
                extras[f"O{oid}"] = np.where(out_mask)[0][np.where(ho.labels_ == oid)[0]]

        candidates = {str(c): np.where(labels == c)[0] for c in set(labels) - {-1}}
        candidates.update(extras)

        # Background for distinctiveness
        rest_texts_all = self.df.loc[~self.df.index.isin(idx_all), 'clean_text'].tolist()

        MIN_SUPPORT = max(5, int(0.005 * len(dfp)))  # Even lower threshold
        MIN_COH = 0.10  # Much lower coherence threshold

        occasions = []
        for cid, loc in candidates.items():
            if len(loc) < MIN_SUPPORT:
                continue

            # 2) c-TF-IDF phrases (2-4 grams), NO stopword removal to keep function words
            cohort_texts = [texts[i] for i in loc]
            
            # Adjust parameters based on cohort size
            min_df = max(1, min(2, len(cohort_texts) // 10))
            max_df = min(0.9, max(0.3, len(cohort_texts) / max(1, len(texts))))
            
            vec = TfidfVectorizer(ngram_range=(2,4), min_df=min_df, max_df=max_df, max_features=80000, stop_words=None)
            try:
                rest_texts = [texts[i] for i in set(range(len(texts))) - set(loc)]
                if len(rest_texts) < 5:  # Too few rest texts, use global
                    raise ValueError("Insufficient rest texts")
                Xc = vec.fit_transform([" ".join(cohort_texts), " ".join(rest_texts)])
            except Exception:
                # fallback: compare to global rest if same-product rest is too small
                try:
                    Xc = vec.fit_transform([" ".join(cohort_texts), " ".join(rest_texts_all[:1000])])  # Limit global rest
                except Exception:
                    continue  # Skip this cohort if TF-IDF fails
            terms = np.array(vec.get_feature_names_out())
            C, R = Xc[0].toarray().ravel(), Xc[1].toarray().ravel()

            # score phrases by lift * log-odds-like ratio
            scores = []
            for i in range(len(terms)):
                c, r = C[i], R[i]
                if c <= 0: 
                    continue
                lift = c / (r + 1e-6)
                p_c = (c + 0.01) / (C.sum() + 0.02)
                p_r = (r + 0.01) / (R.sum() + 0.02)
                logodds = np.log(p_c) - np.log(p_r)
                s = lift * max(logodds, 0.0)
                # mild cleaning: avoid extremely generic connectors-only phrases
                if len(terms[i].split()) <= 1:
                    continue
                if re.match(r"^(and|or|but|so|because)\b", terms[i]):
                    continue
                scores.append((terms[i], float(s)))
            scores.sort(key=lambda x: x[1], reverse=True)
            top_phrases = [t for t,_ in scores[:120]]  # a pool for clustering

            if not top_phrases:
                continue

            # 3) phrase embedding + HDBSCAN to form "occasion themes"
            P = self.model.encode(top_phrases, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
            if len(top_phrases) >= 8:
                hp = hdbscan.HDBSCAN(min_cluster_size=max(3, len(top_phrases)//15), min_samples=1,
                                     cluster_selection_method="leaf", metric="euclidean").fit(P)
                plabels = hp.labels_
            else:
                plabels = np.zeros(len(top_phrases), dtype=int)

            phrase_groups = defaultdict(list)
            for i, pl in enumerate(plabels):
                if pl == -1: 
                    # treat singleton as its own theme
                    phrase_groups[f"S{i}"].append(top_phrases[i])
                else:
                    phrase_groups[int(pl)].append(top_phrases[i])

            # 4) score each theme and pick top few per cohort
            gidx_global = np.array(idx_all[loc])
            coh = self._coherence(gidx_global)
            if coh < MIN_COH:
                continue  # cohort too incoherent

            # Build fast inverted index for quote selection
            rev_texts = cohort_texts
            theme_candidates = []
            for gid, plist in phrase_groups.items():
                # support = # reviews in cohort containing ANY phrase of plist
                pat = r"|".join(re.escape(p) for p in sorted(plist, key=len, reverse=True)[:10])
                m = [bool(re.search(pat, t)) for t in rev_texts]
                support = int(sum(m))
                if support < max(8, int(0.01*len(dfp))):
                    continue

                # distinctiveness via average of phrase scores used earlier
                avg_score = np.mean([dict(scores).get(p, 0.0) for p in plist])

                # coverage within cohort
                coverage = support / max(1, len(rev_texts))

                # theme label: pick the highest-scoring short phrase
                plist_sorted = sorted(plist, key=lambda p: (-(dict(scores).get(p, 0.0)), len(p)))
                label = plist_sorted[0] if plist_sorted else "Novel usage"

                # bullets: derive from top 2-3 phrases (simple human-readable tweaks)
                bullets = []
                for ph in plist_sorted[:3]:
                    bullets.append(ph.capitalize().replace("  ", " "))
                # quotes: pick 2-4 shortest reviews that hit the pattern
                quotes = [rev_texts[i] for i, ok in enumerate(m) if ok]
                quotes = [q.strip() for q in quotes if 8 <= len(q.split()) <= 45][:4]

                # theme score blends coverage, avg_phrase_score, and cohort coherence
                score = 0.45*coverage + 0.35*min(avg_score/np.percentile([s for _,s in scores], 75 if scores else 1), 1.0) + 0.20*coh

                theme_candidates.append({
                    "label": label,
                    "bullets": bullets[:3],
                    "quotes": quotes,
                    "coverage": float(coverage),
                    "avg_phrase_score": float(avg_score),
                    "coherence": float(coh),
                    "support": int(support),
                    "phrase_count": int(len(plist))
                })

            # retain top 2 themes from this cohort
            theme_candidates.sort(key=lambda x: (-(0.45*x["coverage"] + 0.35*min(x["avg_phrase_score"],1.0) + 0.20*x["coherence"]), -x["support"]))
            occasions.extend(theme_candidates[:2])

        # 5) final selection across cohorts: dedup semantically similar labels and keep top 3
        if not occasions:
            return []
        # title-normalize
        for oc in occasions:
            oc["title"] = self._prettify_title(oc["label"])

        # dedup by title similarity (Jaccard token overlap)
        final = []
        titles = []
        for oc in sorted(occasions, key=lambda x: (-(0.45*x["coverage"] + 0.35*min(x["avg_phrase_score"],1.0) + 0.20*x["coherence"]), -x["support"])):
            toks = set(oc["title"].lower().split())
            if any(len(toks & set(t.split()))/max(1,len(toks|set(t.split()))) > 0.6 for t in titles):
                continue
            final.append({"title": oc["title"], "bullets": oc["bullets"], "quotes": oc["quotes"]})
            titles.append(oc["title"])
            if len(final) == max_occasions: break

        return final

    # ====== Helpers ======
    def _coherence(self, global_idxs: np.ndarray) -> float:
        if len(global_idxs) < 2: return 1.0
        if len(global_idxs) > 50:
            rng = np.random.default_rng(self.cfg.random_state)
            global_idxs = rng.choice(global_idxs, size=50, replace=False)
        X = self.embeddings[global_idxs]
        c = X.mean(axis=0, keepdims=True)
        return float(cosine_similarity(X, c).mean())

    @staticmethod
    def _prettify_title(phrase: str) -> str:
        # Simple prettifier: title-case and mild tweaks
        s = re.sub(r"\s+", " ", phrase.strip())
        s = s.replace(" spf ", " SPF ").replace(" aha ", " AHA ").replace(" bha ", " BHA ")
        return s[:1].upper() + s[1:]

    @staticmethod
    def render_occasions_card(product_name: str, occasions: List[Dict]) -> str:
        if not occasions:
            return f"{product_name} — No clear novel usage occasions found."
        lines = [f"{product_name} Usage Occasions", ""]
        for oc in occasions:
            lines.append(oc["title"])
            for b in oc["bullets"]:
                lines.append(b)
            for q in oc["quotes"]:
                lines.append(f"\"{q}\"")
            lines.append("")
        return "\n".join(lines)

# ---------------------------
# CLI
# ---------------------------
def main():
    cfg = RunConfig()
    dd = DivergentDiscoveryV5(cfg)
    result = dd.run()
    print(f"Ready! Dataset: {result['reviews']:,} reviews, {result['products']} products")
    
    # Generate results for dashboard
    print("\nGenerating occasion mining results for dashboard...")
    
    # Get all products with sufficient reviews for meaningful analysis
    product_counts = dd.df['product_name'].value_counts()
    eligible_products = product_counts[product_counts >= 25]  # All products with 25+ reviews
    
    dashboard_results = {
        "summary": {
            "total_reviews": len(dd.df),
            "total_products": dd.df['product_name'].nunique(),
            "analyzed_products": len(eligible_products),
            "timestamp": datetime.now().isoformat()
        },
        "outlier_cohorts": []  # Will populate with occasion-based cohorts
    }
    
    cohort_id = 0
    
    for product_name in eligible_products.index:
        product_count = eligible_products[product_name]
        print(f"Mining occasions for: {product_name} ({product_count} reviews)")
        occasions = dd.generate_novel_usage_occasions(product_name, max_occasions=3)
        print(f"  -> Found {len(occasions)} occasions")
        
        for occasion in occasions:
            # Convert occasion to dashboard cohort format
            cohort_data = {
                "cohort_id": f"occasion_cohort_{cohort_id}",
                "interpretation": occasion["title"],
                "size": len(occasion.get("quotes", [])),
                "percentage": (len(occasion.get("quotes", [])) / len(dd.df)) * 100,
                "distinctive_terms": occasion.get("bullets", []),
                "sample_reviews": occasion.get("quotes", [])[:3],
                "all_reviews": occasion.get("quotes", []),
                "all_reviews_with_products": [
                    {
                        "clean_text": quote,
                        "product_name": product_name,
                        "source": "aggregated"
                    }
                    for quote in occasion.get("quotes", [])
                ],
                "product_focus": {product_name: len(occasion.get("quotes", []))},
                "source_distribution": {"aggregated": len(occasion.get("quotes", []))},
                "pattern_type": "usage_occasion",
                "product_name": product_name
            }
            dashboard_results["outlier_cohorts"].append(cohort_data)
            cohort_id += 1
    
    # Save results in dashboard format
    output_file = "v4_occasion_discovery_results.json"
    with open(output_file, 'w') as f:
        json.dump(dashboard_results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    print(f"📊 Generated {len(dashboard_results['outlier_cohorts'])} usage occasion cohorts")
    print(f"🎯 Analyzed {len(eligible_products)} products with sufficient data")
    
    return dd

if __name__ == "__main__":
    discovery_engine = main()
