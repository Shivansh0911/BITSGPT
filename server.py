"""
BitsGPT — FastAPI Server
Serves the chat API with SSE streaming and the static web UI.

Run:
    uvicorn server:app --reload --port 8000
    # Then open http://localhost:8000
"""

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag import BitsGPT

app = FastAPI(
    title="BitsGPT",
    description="AI assistant for BITS Pilani Hyderabad Campus",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_bot: BitsGPT | None = None


def get_bot() -> BitsGPT:
    global _bot
    if _bot is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. "
                "Get a free key at https://console.groq.com/ and set it before starting the server."
            )
        _bot = BitsGPT(api_key=api_key)
    return _bot


@app.on_event("startup")
async def startup():
    try:
        get_bot()
        print("[BitsGPT] RAG pipeline loaded successfully.")
    except Exception as e:
        print(f"[BitsGPT] WARNING: Could not pre-load bot — {e}")
        print("[BitsGPT] The bot will attempt to load on the first request.")


class ChatRequest(BaseModel):
    message: str
    stream: bool = True


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    try:
        bot = get_bot()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if req.stream:
        def generate():
            try:
                for typ, content in bot.stream_ask(req.message):
                    if typ == "chunk":
                        yield f"data: {json.dumps({'type': 'chunk', 'text': content})}\n\n"
                    elif typ == "done":
                        yield f"data: {json.dumps({'type': 'done', 'sources': content.get('sources', [])})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        try:
            result = bot.ask(req.message)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return JSONResponse(result)


@app.get("/api/health")
async def health():
    data_dir = Path(__file__).parent / "data"
    index_ready = (
        (data_dir / "tfidf_index.pkl").exists()
        and (data_dir / "chunks.json").exists()
    )
    return {
        "status": "ok",
        "model": "llama-3.3-70b-versatile",
        "index_ready": index_ready,
        "groq_key_set": bool(os.getenv("GROQ_API_KEY")),
    }


@app.get("/api/suggestions")
async def suggestions():
    return {
        "questions": [
            "What is the minimum CGPA required to stay enrolled?",
            "How does PS-1 and PS-2 work?",
            "What clubs can I join at BPHC?",
            "What are the fee components for 2023 batch?",
            "How does a no-confidence motion against the President work?",
            "What food outlets are at Connaught Place?",
            "What is ATMOS and when does it happen?",
            "How do I apply for a minor degree?",
            "What does 'lite' mean in BITS jargon?",
            "What hostels are available and what do rooms include?",
        ]
    }


static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
