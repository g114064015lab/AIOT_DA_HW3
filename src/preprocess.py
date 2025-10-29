"""Text preprocessing utilities for spam classification."""
import re
from typing import Optional, Dict, Any

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


def download_nltk_data():
    """Download required NLTK data files."""
    for resource in ['punkt', 'stopwords', 'wordnet']:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning: Failed to download NLTK resource {resource}: {e}")


def clean_text(text: str,
              remove_urls: bool = True,
              remove_numbers: bool = True,
              remove_punct: bool = True) -> str:
    """Clean text by removing URLs, numbers, punctuation."""
    if not isinstance(text, str):
        return ""
    
    text = text.lower().strip()
    
    if remove_urls:
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    if remove_numbers:
        # Replace numbers with token
        text = re.sub(r'\d+', ' <NUM> ', text)
    
    if remove_punct:
        # Remove punctuation but keep word boundaries
        text = re.sub(r'[^\w\s<>]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_text(text: str,
                   config: Optional[Dict[str, Any]] = None) -> str:
    """Full preprocessing pipeline including cleaning, tokenization, etc."""
    if config is None:
        config = {}
    
    # Get config values with defaults
    do_clean = config.get('clean', True)
    do_lemmatize = config.get('lemmatize', False)
    do_remove_stops = config.get('remove_stopwords', True)
    
    if do_clean:
        text = clean_text(
            text,
            remove_urls=config.get('remove_urls', True),
            remove_numbers=config.get('remove_numbers', True),
            remove_punct=config.get('remove_punct', True)
        )
    
    if do_lemmatize or do_remove_stops:
        # Tokenize
        tokens = word_tokenize(text)
        
        if do_remove_stops:
            # Remove stopwords
            stops = set(stopwords.words('english'))
            tokens = [t for t in tokens if t not in stops]
        
        if do_lemmatize:
            # Lemmatize
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(t) for t in tokens]
        
        # Rejoin tokens
        text = ' '.join(tokens)
    
    return text


def build_vectorizer(max_features: Optional[int] = None,
                    max_df: float = 0.9,
                    min_df: int = 3,
                    ngram_range: tuple = (1, 2),
                    use_char_ngrams: bool = False) -> TfidfVectorizer:
    """Build a TF-IDF vectorizer with configurable parameters."""
    params = {
        'max_features': max_features,
        'max_df': max_df,
        'min_df': min_df,
        'ngram_range': ngram_range,
        'stop_words': 'english',  # Additional stopword removal at vectorizer level
    }
    # If character n-grams are explicitly requested, switch analyzer,
    # otherwise use word n-grams (default) which typically works well for SMS.
    if use_char_ngrams:
        params.update({
            'analyzer': 'char_wb',
            'ngram_range': (3, 5),  # Common character ngram range
        })

    return TfidfVectorizer(**params)


# Download NLTK data on import
download_nltk_data()