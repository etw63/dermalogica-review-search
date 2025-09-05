#!/usr/bin/env python3
"""
Run this script ONCE to preprocess your entire dataset with usage patterns.
This creates the preprocessed CSV file that your main app will use.
"""

import pandas as pd
import numpy as np
import re
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter

class UsageTaxonomyPreprocessor:
    """Preprocess entire dataset to assign usage contexts and roles to every review"""
    
    def __init__(self, embeddings_model='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(embeddings_model)
        
        # Core usage taxonomy - starts with seed patterns
        self.usage_taxonomy = {
            'contexts': {
                'daily_routine': {
                    'patterns': [
                        r'\b(daily|every day|everyday|each day|routine|always)\b',
                        r'\b(morning|evening|night|am|pm)\b'
                    ],
                    'keywords': ['daily', 'routine', 'every day', 'morning', 'evening'],
                    'description': 'Regular daily skincare routine'
                },
                'shower_routine': {
                    'patterns': [
                        r'\b(shower|bath|washing face|in the shower)\b',
                        r'\b(keep.*shower|shower.*keep)\b'
                    ],
                    'keywords': ['shower', 'bath', 'in shower'],
                    'description': 'Used during shower or bath time'
                },
                'pre_makeup': {
                    'patterns': [
                        r'\b(before makeup|pre.?makeup|under makeup|makeup base)\b',
                        r'\b(glass skin|smooth.*makeup|prep.*makeup)\b'
                    ],
                    'keywords': ['before makeup', 'pre makeup', 'glass skin', 'makeup base'],
                    'description': 'Preparation before applying makeup'
                },
                'special_events': {
                    'patterns': [
                        r'\b(special|occasion|event|weekend|night out|date)\b',
                        r'\b(glow.*event|before.*event|facial.*event)\b'
                    ],
                    'keywords': ['special occasion', 'event', 'night out', 'glow'],
                    'description': 'Special occasions or events'
                },
                'pre_workout': {
                    'patterns': [
                        r'\b(before workout|pre.?workout|gym|exercise|sweat)\b',
                        r'\b(active|sport|run|before.*run)\b'
                    ],
                    'keywords': ['workout', 'gym', 'exercise', 'before run', 'sweat'],
                    'description': 'Before physical activity or exercise'
                },
                'travel': {
                    'patterns': [
                        r'\b(travel|trip|vacation|plane|hotel)\b',
                        r'\b(on the go|portable|pack)\b'
                    ],
                    'keywords': ['travel', 'trip', 'vacation', 'on the go'],
                    'description': 'During travel or on-the-go situations'
                },
                'sensitive_periods': {
                    'patterns': [
                        r'\b(pregnant|pregnancy|sensitive|reaction|irritat)\b',
                        r'\b(gentle|mild|safe|during pregnancy)\b'
                    ],
                    'keywords': ['pregnancy', 'sensitive', 'gentle', 'safe'],
                    'description': 'During sensitive periods (pregnancy, reactions)'
                }
            },
            'roles': {
                'replacement': {
                    'patterns': [
                        r'\b(only|main|primary|instead of|replace|switch)\b',
                        r'\b(go.?to|staple|must.?have|this is all)\b',
                        r'\b(don\'t need|no other|nothing else)\b'
                    ],
                    'keywords': ['only', 'main', 'primary', 'replace', 'go-to', 'staple'],
                    'description': 'Replaces other products entirely'
                },
                'consolidator': {
                    'patterns': [
                        r'\b(in one|all in one|combines|both|together)\b',
                        r'\b(moisturizer.*spf|spf.*moisturizer|don\'t need.*else)\b',
                        r'\b(simplifies|one step|everything)\b'
                    ],
                    'keywords': ['in one', 'all in one', 'combines', 'simplifies'],
                    'description': 'Combines multiple functions in one product'
                },
                'prep_enhancer': {
                    'patterns': [
                        r'\b(prep|prepare|prime|base|ready|before)\b',
                        r'\b(glow|bright|smooth|polish|enhance)\b',
                        r'\b(glass skin|luminous|radiant)\b'
                    ],
                    'keywords': ['prep', 'prime', 'glow', 'enhance', 'glass skin'],
                    'description': 'Prepares or enhances skin appearance'
                },
                'treatment': {
                    'patterns': [
                        r'\b(acne|breakout|pimple|clear|problem|fix)\b',
                        r'\b(aging|wrinkle|fine line|anti.?aging|repair)\b',
                        r'\b(dry|flaky|rough|heal|treat|cure)\b'
                    ],
                    'keywords': ['acne', 'aging', 'treat', 'fix', 'repair', 'clear'],
                    'description': 'Treats specific skin problems or concerns'
                }
            }
        }
        
        self.context_embeddings = {}
        self.role_embeddings = {}
        
    def _create_embeddings_for_taxonomy(self):
        """Create embeddings for existing taxonomy patterns"""
        print("Creating embeddings for taxonomy patterns...")
        
        for context_name, context_info in self.usage_taxonomy['contexts'].items():
            text = f"{context_info['description']} {' '.join(context_info['keywords'])}"
            embedding = self.model.encode([text])[0]
            self.context_embeddings[context_name] = embedding
            
        for role_name, role_info in self.usage_taxonomy['roles'].items():
            text = f"{role_info['description']} {' '.join(role_info['keywords'])}"
            embedding = self.model.encode([text])[0]
            self.role_embeddings[role_name] = embedding
    
    def _match_context(self, review_text):
        """Find best matching context for a review"""
        text_lower = review_text.lower()
        context_scores = {}
        
        for context_name, context_info in self.usage_taxonomy['contexts'].items():
            score = 0
            
            # Check regex patterns
            for pattern in context_info['patterns']:
                if re.search(pattern, text_lower):
                    score += 2
            
            # Check keywords
            for keyword in context_info['keywords']:
                if keyword.lower() in text_lower:
                    score += 1
            
            if score > 0:
                context_scores[context_name] = score
        
        # Semantic similarity fallback
        if not context_scores:
            review_embedding = self.model.encode([review_text])[0]
            
            for context_name, context_embedding in self.context_embeddings.items():
                similarity = cosine_similarity([review_embedding], [context_embedding])[0][0]
                if similarity > 0.3:
                    context_scores[context_name] = similarity
        
        if context_scores:
            best_context = max(context_scores, key=context_scores.get)
            confidence = context_scores[best_context]
            return best_context, confidence
        
        return None, 0.0
    
    def _match_role(self, review_text):
        """Find best matching role for a review"""
        text_lower = review_text.lower()
        role_scores = {}
        
        for role_name, role_info in self.usage_taxonomy['roles'].items():
            score = 0
            
            for pattern in role_info['patterns']:
                if re.search(pattern, text_lower):
                    score += 2
            
            for keyword in role_info['keywords']:
                if keyword.lower() in text_lower:
                    score += 1
            
            if score > 0:
                role_scores[role_name] = score
        
        # Semantic similarity fallback
        if not role_scores:
            review_embedding = self.model.encode([review_text])[0]
            
            for role_name, role_embedding in self.role_embeddings.items():
                similarity = cosine_similarity([review_embedding], [role_embedding])[0][0]
                if similarity > 0.3:
                    role_scores[role_name] = similarity
        
        if role_scores:
            best_role = max(role_scores, key=role_scores.get)
            confidence = role_scores[best_role]
            return best_role, confidence
            
        return None, 0.0
    
    def assign_or_create_usage(self, review_text, product_name=None):
        """Assign usage context and role to a review"""
        
        context, context_confidence = self._match_context(review_text)
        role, role_confidence = self._match_role(review_text)
        
        # Default assignments for low confidence
        if not context or context_confidence < 0.4:
            context = 'general_context'
            context_confidence = 0.1
        
        if not role or role_confidence < 0.4:
            role = 'general_role'
            role_confidence = 0.1
        
        return {
            'context': context,
            'context_confidence': context_confidence,
            'role': role,
            'role_confidence': role_confidence,
            'product_name': product_name
        }
    
    def preprocess_dataset(self, csv_file, output_file=None):
        """Process entire dataset and assign usage patterns to every review"""
        
        print(f"Loading dataset from {csv_file}...")
        df = pd.read_csv(csv_file)
        
        self._create_embeddings_for_taxonomy()
        
        print(f"Processing {len(df)} reviews...")
        
        usage_assignments = []
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"Processed {idx}/{len(df)} reviews...")
            
            review_text = str(row.get('review_text', ''))
            product_name = str(row.get('clean_product_name', ''))
            
            if len(review_text) < 10:
                usage_assignment = {
                    'context': 'insufficient_text',
                    'context_confidence': 0.0,
                    'role': 'insufficient_text',
                    'role_confidence': 0.0,
                    'product_name': product_name
                }
            else:
                usage_assignment = self.assign_or_create_usage(review_text, product_name)
            
            usage_assignments.append(usage_assignment)
        
        # Add usage assignments to dataframe
        df['usage_context'] = [ua['context'] for ua in usage_assignments]
        df['context_confidence'] = [ua['context_confidence'] for ua in usage_assignments]
        df['usage_role'] = [ua['role'] for ua in usage_assignments]
        df['role_confidence'] = [ua['role_confidence'] for ua in usage_assignments]
        
        # Save processed dataset
        if output_file is None:
            output_file = csv_file.replace('.csv', '_with_usage_patterns.csv')
        
        df.to_csv(output_file, index=False)
        print(f"Saved processed dataset to {output_file}")
        
        # Generate summary report
        self._generate_preprocessing_report(df, usage_assignments)
        
        return df
    
    def _generate_preprocessing_report(self, df, usage_assignments):
        """Generate a summary report of the preprocessing results"""
        
        print("\n=== PREPROCESSING REPORT ===")
        print(f"Total reviews processed: {len(df)}")
        
        context_counts = Counter([ua['context'] for ua in usage_assignments])
        print(f"\nUsage Contexts Found:")
        for context, count in context_counts.most_common():
            percentage = count / len(df) * 100
            print(f"  {context}: {count} reviews ({percentage:.1f}%)")
        
        role_counts = Counter([ua['role'] for ua in usage_assignments])
        print(f"\nUsage Roles Found:")
        for role, count in role_counts.most_common():
            percentage = count / len(df) * 100
            print(f"  {role}: {count} reviews ({percentage:.1f}%)")
        
        context_confidences = [ua['context_confidence'] for ua in usage_assignments]
        role_confidences = [ua['role_confidence'] for ua in usage_assignments]
        
        print(f"\nConfidence Scores:")
        print(f"  Context - Mean: {np.mean(context_confidences):.2f}")
        print(f"  Role - Mean: {np.mean(role_confidences):.2f}")


if __name__ == "__main__":
    print("Starting usage pattern preprocessing...")
    
    # Initialize preprocessor
    preprocessor = UsageTaxonomyPreprocessor()
    
    # Process the entire dataset
    df_processed = preprocessor.preprocess_dataset("dermalogica_aggregated_reviews.csv")
    
    print("\n✅ Preprocessing complete!")
    print("Your dataset now has usage_context and usage_role columns")
    print("You can now use the processed file in your main app")
