"""
NLP Processor Module
Uses NLP to extract keywords and generate word clouds from job descriptions.
"""

import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
from typing import List, Dict, Optional
import os

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. Some features may be limited.")


class NLPProcessor:
    """Processes job descriptions using NLP techniques."""
    
    def __init__(self, output_dir: str = "output"):
        """Initialize NLP processor."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        if NLTK_AVAILABLE:
            try:
                self.stop_words = set(stopwords.words('english'))
                self.lemmatizer = WordNetLemmatizer()
            except LookupError:
                print("Warning: NLTK data not downloaded. Run: python -m nltk.downloader punkt stopwords wordnet")
                self.stop_words = set()
                self.lemmatizer = None
        else:
            # Basic stop words list
            self.stop_words = {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
                'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if',
                'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her',
                'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more',
                'very', 'after', 'words', 'long', 'than', 'first', 'been', 'call',
                'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get',
                'come', 'made', 'may', 'part'
            }
            self.lemmatizer = None
    
    def preprocess_text(self, text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> List[str]:
        """Preprocess text: tokenize, remove stopwords, lemmatize."""
        if pd.isna(text) or not isinstance(text, str):
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        if NLTK_AVAILABLE:
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords
            if remove_stopwords:
                tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
            
            # Lemmatize
            if lemmatize and self.lemmatizer:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        else:
            # Simple tokenization
            tokens = text.split()
            if remove_stopwords:
                tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        return tokens
    
    def extract_keywords(self, texts: List[str], top_n: int = 50) -> List[tuple]:
        """Extract top keywords from a collection of texts."""
        all_tokens = []
        
        for text in texts:
            if pd.notna(text):
                tokens = self.preprocess_text(str(text))
                all_tokens.extend(tokens)
        
        # Count frequencies
        word_freq = Counter(all_tokens)
        
        # Get top N keywords
        top_keywords = word_freq.most_common(top_n)
        
        return top_keywords
    
    def extract_nouns(self, texts: List[str], top_n: int = 30) -> List[tuple]:
        """Extract nouns from texts (requires NLTK)."""
        if not NLTK_AVAILABLE:
            print("Warning: NLTK not available. Falling back to general keyword extraction.")
            return self.extract_keywords(texts, top_n)
        
        all_nouns = []
        
        for text in texts:
            if pd.notna(text):
                tokens = word_tokenize(str(text).lower())
                tagged = pos_tag(tokens)
                nouns = [word for word, pos in tagged if pos.startswith('NN') and word not in self.stop_words and len(word) > 2]
                all_nouns.extend(nouns)
        
        noun_freq = Counter(all_nouns)
        return noun_freq.most_common(top_n)
    
    def extract_skills_from_text(self, texts: List[str], known_skills: Optional[List[str]] = None) -> Dict[str, int]:
        """Extract technical skills mentioned in job descriptions."""
        if known_skills is None:
            known_skills = [
                'python', 'java', 'javascript', 'react', 'node', 'sql', 'aws',
                'docker', 'kubernetes', 'tensorflow', 'pytorch', 'git', 'mongodb',
                'postgresql', 'redis', 'kafka', 'spark', 'machine learning',
                'deep learning', 'data analysis', 'rest', 'graphql', 'typescript',
                'vue', 'angular', 'ci/cd', 'linux', 'azure', 'gcp', 'scala', 'go',
                'ruby', 'php', 'html', 'css', 'sass', 'less', 'jquery', 'django',
                'flask', 'express', 'spring', 'hibernate', 'elasticsearch'
            ]
        
        skill_counts = Counter()
        
        for text in texts:
            if pd.notna(text):
                text_lower = str(text).lower()
                for skill in known_skills:
                    if skill.lower() in text_lower:
                        skill_counts[skill] += 1
        
        return dict(skill_counts)
    
    def generate_wordcloud(self, texts: List[str], title: str = "Word Cloud", 
                          width: int = 800, height: int = 400, save: bool = True) -> WordCloud:
        """Generate a word cloud from texts."""
        # Combine all texts
        combined_text = ' '.join([str(text) for text in texts if pd.notna(text)])
        
        # Preprocess
        tokens = self.preprocess_text(combined_text)
        processed_text = ' '.join(tokens)
        
        if not processed_text:
            print("Warning: No text to generate word cloud from.")
            return None
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color='white',
            max_words=100,
            colormap='viridis',
            relative_scaling=0.5,
            random_state=42
        ).generate(processed_text)
        
        # Display or save
        plt.figure(figsize=(width/100, height/100))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16, pad=20)
        
        if save:
            filename = os.path.join(self.output_dir, f'wordcloud_{title.lower().replace(" ", "_")}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved word cloud to {filename}")
        
        plt.close()
        return wordcloud
    
    def generate_keyword_wordcloud(self, keywords: List[tuple], title: str = "Keyword Word Cloud",
                                   width: int = 800, height: int = 400, save: bool = True) -> WordCloud:
        """Generate word cloud from keyword frequency list."""
        if not keywords:
            print("Warning: No keywords provided.")
            return None
        
        # Create dictionary from keyword tuples
        word_freq = {word: freq for word, freq in keywords}
        
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color='white',
            max_words=100,
            colormap='plasma',
            relative_scaling=0.5,
            random_state=42
        ).generate_from_frequencies(word_freq)
        
        plt.figure(figsize=(width/100, height/100))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16, pad=20)
        
        if save:
            filename = os.path.join(self.output_dir, f'wordcloud_{title.lower().replace(" ", "_")}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved keyword word cloud to {filename}")
        
        plt.close()
        return wordcloud
    
    def analyze_job_descriptions(self, df: pd.DataFrame, description_col: str = 'description') -> Dict:
        """Comprehensive NLP analysis of job descriptions."""
        if description_col not in df.columns:
            print(f"Warning: Column '{description_col}' not found. Using all text columns.")
            # Try to combine all text columns
            texts = df.select_dtypes(include=['object']).fillna('').agg(' '.join, axis=1).tolist()
        else:
            texts = df[description_col].fillna('').tolist()
        
        analysis = {
            'keywords': self.extract_keywords(texts),
            'nouns': self.extract_nouns(texts) if NLTK_AVAILABLE else [],
            'skills': self.extract_skills_from_text(texts)
        }
        
        return analysis
    
    def process_and_visualize(self, df: pd.DataFrame, description_col: str = 'description'):
        """Process job descriptions and create visualizations."""
        analysis = self.analyze_job_descriptions(df, description_col)
        
        # Generate word clouds
        if description_col in df.columns:
            texts = df[description_col].fillna('').tolist()
            self.generate_wordcloud(texts, title="Job Descriptions Word Cloud")
            
            if analysis['keywords']:
                self.generate_keyword_wordcloud(
                    analysis['keywords'][:50],
                    title="Top Keywords Word Cloud"
                )
        
        return analysis


if __name__ == "__main__":
    processor = NLPProcessor()
    # Example usage
    # texts = ["Looking for a Python developer with React experience.", "Java developer needed."]
    # keywords = processor.extract_keywords(texts)
    # processor.generate_keyword_wordcloud(keywords)

