"""
BitsGPT — RAG Query Pipeline
TF-IDF retrieval → keyword rerank → Groq LLM (llama-3.3-70b)

Usage:
    python rag.py "What is the minimum CGPA at BPHC?"
    python rag.py --interactive
    GROQ_API_KEY=gsk_... python rag.py "How does PS-1 work?"
"""

import os
import re
import sys
import json
import pickle
import argparse
import textwrap
from pathlib import Path
from typing import List, Dict, Optional, Generator

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

BASE_DIR = Path(__file__).parent
INDEX_PATH = BASE_DIR / "data" / "tfidf_index.pkl"
CHUNKS_PATH = BASE_DIR / "data" / "chunks.json"

GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K_RETRIEVE = 20
TOP_K_FINAL = 7
MAX_CTX_WORDS = 2800

SYSTEM_PROMPT = """You are BitsGPT — the official AI assistant for BITS Pilani Hyderabad Campus (BPHC).
You answer questions about academics, student life, clubs, fests, hostels, fees, the Students' Union constitution, campus procedures, and anything a BPHC student might ask.

STRICT RULES:
1. Answer ONLY from the provided CONTEXT. Never hallucinate facts, numbers, or names.
2. If context is insufficient, say so clearly and direct the student to the right office (SWD, EC, CRC, AUGSD, etc.).
3. Be concise. Use bullet points or numbered lists for multi-part answers.
4. For constitutional/procedural thresholds (⅔ majority, ⅓ quorum), state them exactly.
5. For academic rules (CGPA, credits, deadlines, fees), always give the exact figure from context.
6. Friendly, helpful tone — like a knowledgeable 4th-year BITSian senior.
7. Never invent contact numbers, dates, or fees not in the context.
"""


class BPHCRetriever:
    def __init__(self):
        if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
            raise FileNotFoundError(
                "Index not found. Run `python ingest.py` first to build the index."
            )
        with open(INDEX_PATH, "rb") as f:
            idx = pickle.load(f)
        self.vectorizer = idx["vectorizer"]
        self.tfidf_matrix = idx["matrix"]
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            self.chunks = json.load(f)

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVE) -> List[Dict]:
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.tfidf_matrix)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_idx:
            if scores[i] > 0.01:
                c = dict(self.chunks[i])
                c["tfidf_score"] = float(scores[i])
                results.append(c)
        return results

    def rerank(self, query: str, candidates: List[Dict], top_k: int = TOP_K_FINAL) -> List[Dict]:
        q_tokens = set(re.findall(r"\b\w+\b", query.lower()))
        for c in candidates:
            text_tokens = set(re.findall(r"\b\w+\b", c["text"].lower()))
            overlap = len(q_tokens & text_tokens) / max(len(q_tokens), 1)
            kb_boost = 1.3 if c["metadata"].get("doc_type") == "curated" else 1.0
            c["final_score"] = (c["tfidf_score"] * kb_boost) + (overlap * 0.15)
        ranked = sorted(candidates, key=lambda x: x["final_score"], reverse=True)
        return ranked[:top_k]

    def get_context(self, query: str):
        candidates = self.retrieve(query)
        top_chunks = self.rerank(query, candidates)
        ctx_parts, word_count, used = [], 0, []
        for c in top_chunks:
            words = c["text"].split()
            if word_count + len(words) > MAX_CTX_WORDS:
                remaining = MAX_CTX_WORDS - word_count
                if remaining > 80:
                    c = dict(c)
                    c["text"] = " ".join(words[:remaining]) + " ..."
                else:
                    break
            ctx_parts.append(c["text"])
            word_count += len(c["text"].split())
            used.append(c)
        return "\n\n---\n\n".join(ctx_parts), used


class GroqLLM:
    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise ValueError(
                "GROQ_API_KEY not set.\n"
                "  Set it: export GROQ_API_KEY=gsk_...\n"
                "  Get a free key: https://console.groq.com/"
            )
        self.client = Groq(api_key=key)

    def answer(self, query: str, context: str, stream: bool = False):
        user_msg = (
            f"CONTEXT (from BPHC official documents):\n\n{context}\n\n"
            f"---\n\nQUESTION: {query}\n\n"
            "Answer strictly from the context above."
        )
        return self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.15,
            max_tokens=1024,
            stream=stream,
        )


class BitsGPT:
    def __init__(self, api_key: Optional[str] = None):
        self.retriever = BPHCRetriever()
        self.llm = GroqLLM(api_key)

    def ask(self, query: str, verbose: bool = False) -> Dict:
        context, chunks = self.retriever.get_context(query)
        if not context.strip():
            return {
                "query": query,
                "answer": "Couldn't find relevant info. Try rephrasing, or contact SWD/AUGSD directly.",
                "sources": [],
            }
        resp = self.llm.answer(query, context)
        answer = resp.choices[0].message.content.strip()

        if verbose:
            print("\n── RETRIEVED CONTEXT ──")
            print(textwrap.shorten(context, 1500, placeholder=" ..."))
            print("\n── SCORES ──")
            for c in chunks:
                print(
                    f"  [{c.get('final_score', 0):.3f}] "
                    f"{c['metadata']['source']} — {c['metadata'].get('chunk_title', '')}"
                )
            print()

        seen, sources = set(), []
        for c in chunks:
            src = c["metadata"].get("source", "?")
            title = c["metadata"].get("chunk_title", "")
            label = f"{src}|{title}"
            if label not in seen:
                seen.add(label)
                sources.append(
                    {
                        "source": src,
                        "chunk_title": title,
                        "score": round(c.get("final_score", 0), 3),
                    }
                )
        return {"query": query, "answer": answer, "sources": sources}

    def stream_ask(self, query: str) -> Generator:
        """Yields (type, content) tuples — type ∈ 'chunk' | 'done'."""
        context, chunks = self.retriever.get_context(query)
        if not context.strip():
            yield ("done", {"answer": "No relevant info found.", "sources": []})
            return
        stream = self.llm.answer(query, context, stream=True)
        full_text = ""
        for part in stream:
            delta = part.choices[0].delta.content or ""
            if delta:
                full_text += delta
                yield ("chunk", delta)
        seen, sources = set(), []
        for c in chunks:
            src = c["metadata"].get("source", "?")
            title = c["metadata"].get("chunk_title", "")
            label = f"{src}|{title}"
            if label not in seen:
                seen.add(label)
                sources.append(
                    {
                        "source": src,
                        "chunk_title": title,
                        "score": round(c.get("final_score", 0), 3),
                    }
                )
        yield ("done", {"answer": full_text, "sources": sources})


def main():
    p = argparse.ArgumentParser(description="BitsGPT — BPHC AI assistant (CLI)")
    p.add_argument("query", nargs="?", help="Question to ask")
    p.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    p.add_argument("-v", "--verbose", action="store_true", help="Show retrieval scores")
    p.add_argument("--api-key", help="Groq API key")
    p.add_argument("--json", action="store_true", help="JSON output")
    args = p.parse_args()

    if not args.query and not args.interactive:
        p.print_help()
        sys.exit(1)

    bot = BitsGPT(api_key=args.api_key)

    if args.interactive:
        print("\n" + "═" * 58)
        print("  BitsGPT — BPHC Knowledge Assistant")
        print("  Powered by Groq (llama-3.3-70b) + TF-IDF RAG")
        print("  Type 'quit' to exit")
        print("═" * 58 + "\n")
        while True:
            try:
                q = input("You › ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not q:
                continue
            if q.lower() in ("quit", "exit", "bye"):
                print("Bye!")
                break
            print("\nBitsGPT › ", end="", flush=True)
            for typ, content in bot.stream_ask(q):
                if typ == "chunk":
                    print(content, end="", flush=True)
                elif typ == "done":
                    srcs = content.get("sources", [])
                    if srcs:
                        print(
                            "\n\n  Sources: "
                            + " · ".join(
                                s["source"]
                                + (" — " + s["chunk_title"] if s["chunk_title"] else "")
                                for s in srcs[:3]
                            )
                        )
            print("\n")
    else:
        result = bot.ask(args.query, verbose=args.verbose)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"\nAnswer:\n{result['answer']}")
            if result["sources"]:
                print("\nSources:")
                for s in result["sources"]:
                    t = f" — {s['chunk_title']}" if s["chunk_title"] else ""
                    print(f"  [{s['score']:.3f}] {s['source']}{t}")


if __name__ == "__main__":
    main()
