"""
Shared tokenizer for BitsGPT TF-IDF index.
Both ingest.py and rag.py must import stem_tokenize from here so that
pickle can resolve the function reference when loading the saved vectorizer.
"""

import re
import nltk
from nltk.stem import PorterStemmer

nltk.download("punkt_tab", quiet=True)

_stemmer = PorterStemmer()


def stem_tokenize(text: str) -> list:
    """Stemmed unigrams + bigrams used for TF-IDF indexing and querying."""
    raw = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    stemmed = [_stemmer.stem(t) for t in raw]
    features = stemmed[:]
    for i in range(len(stemmed) - 1):
        features.append(f"{stemmed[i]} {stemmed[i + 1]}")
    return features
