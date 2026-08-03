"""
mock_embedder_for_testing.py
------------------------------
YEH SIRF SANDBOX TESTING KE LIYE HAI (jaha Hugging Face access nahi hai).
Tumhare LOCAL MACHINE pe tum asli sentence-transformers use karoge
(schema_retriever.py mein already configured hai).

Yeh ek simple TF-IDF based "fake embedder" hai jo same interface follow karta
hai jaisा SentenceTransformer karta hai, taaki hum poora pipeline test kar sakein
bina internet ke. PRODUCTION mein yeh mat use karna - accuracy kam hogi.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class MockEmbedder:
    """SentenceTransformer jaisa interface, lekin TF-IDF use karta hai (offline)."""

    def __init__(self, model_name=None):
        self.vectorizer = TfidfVectorizer()
        self._fitted = False

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        if not self._fitted:
            # Pehli baar fit karo (table names pe)
            self.vectorizer.fit(texts)
            self._fitted = True
            vectors = self.vectorizer.transform(texts).toarray()
        else:
            try:
                vectors = self.vectorizer.transform(texts).toarray()
            except Exception:
                vectors = np.zeros((len(texts), len(self.vectorizer.vocabulary_)))

        return vectors
