"""
Single-command launcher for Adaptive-RAG (local dev & cloud deployment).
Run locally: python run.py
"""

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    is_cloud = "PORT" in os.environ
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0" if is_cloud else "127.0.0.1",
        port=port,
        reload=not is_cloud,
    )
