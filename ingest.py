"""
BitsGPT — Ingestion Pipeline
Parses the BPHC knowledge base and builds a TF-IDF index for fast retrieval.

Run once:
    python ingest.py

Re-run whenever knowledge_base.md is updated.
"""

import re
import json
import hashlib
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).parent
KB_PATH = BASE_DIR / "knowledge_base.md"
DATA_DIR = BASE_DIR / "data"
INDEX_PATH = DATA_DIR / "tfidf_index.pkl"
CHUNKS_PATH = DATA_DIR / "chunks.json"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 80


def kb_chunks(md_text: str) -> list:
    """Split knowledge base into semantic chunks using # CHUNK headers."""
    parts = re.split(r"\n(?=# CHUNK \d{3})", md_text)
    out = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.match(r"# CHUNK (\d{3}) — (.+)", part)
        num = m.group(1) if m else "000"
        title = m.group(2).strip() if m else "preamble"
        words = part.split()
        if len(words) <= CHUNK_SIZE * 2:
            cid = f"kb_{num}"
            out.append(
                {
                    "id": cid,
                    "text": part,
                    "metadata": {
                        "source": "knowledge_base",
                        "doc_type": "curated",
                        "chunk_num": num,
                        "chunk_title": title,
                    },
                }
            )
        else:
            start, idx = 0, 0
            while start < len(words):
                end = min(start + CHUNK_SIZE, len(words))
                chunk_text = " ".join(words[start:end])
                if len(chunk_text.strip()) > 60:
                    cid = hashlib.md5(
                        f"kb_{num}_{idx}_{chunk_text[:60]}".encode()
                    ).hexdigest()
                    out.append(
                        {
                            "id": cid,
                            "text": chunk_text,
                            "metadata": {
                                "source": "knowledge_base",
                                "doc_type": "curated",
                                "chunk_num": num,
                                "chunk_title": title,
                            },
                        }
                    )
                start += CHUNK_SIZE - CHUNK_OVERLAP
                idx += 1
    return out


def ingest():
    print("=" * 55)
    print("  BitsGPT — Building Knowledge Index")
    print("=" * 55)

    if not KB_PATH.exists():
        raise FileNotFoundError(f"Knowledge base not found: {KB_PATH}")

    DATA_DIR.mkdir(exist_ok=True)

    print(f"\n[1/3] Parsing {KB_PATH.name}...")
    kb_text = KB_PATH.read_text(encoding="utf-8")
    chunks = kb_chunks(kb_text)
    print(f"      {len(chunks)} semantic chunks extracted")

    print("\n[2/3] Building TF-IDF index...")
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=16384,
        sublinear_tf=True,
        min_df=1,
        analyzer="word",
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"      Vocabulary : {len(vectorizer.vocabulary_)} terms")
    print(f"      Matrix     : {tfidf_matrix.shape}")

    print("\n[3/3] Saving index...")
    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": tfidf_matrix}, f)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Done — {len(chunks)} chunks indexed")
    print(f"  {INDEX_PATH}")
    print(f"  {CHUNKS_PATH}\n")
    return len(chunks)


if __name__ == "__main__":
    ingest()
