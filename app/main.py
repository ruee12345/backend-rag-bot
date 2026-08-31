from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="HR Compliance RAG Bot")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routes
try:
    from app.api import documents, rag, auth
    app.include_router(documents.router)
    app.include_router(rag.router)
    app.include_router(auth.router)
except ImportError as e:
    print(f"Warning: Could not import routes: {e}")

@app.get("/")
def root():
    return {"status": "HR Compliance RAG Bot is running"}

@app.get("/api/health")
def health():
    return {"status": "healthy", "service": "hr-compliance-backend"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
