import os

class Settings:
    groq_api_key: str = os.environ.get("GROQ_API_KEY", "")
    secret_key: str = os.environ.get("SECRET_KEY", "your-secret-key-here")
    upload_folder: str = os.environ.get("UPLOAD_FOLDER", "uploads")
    vector_store_path: str = os.environ.get("VECTOR_STORE_PATH", "data/vector_store")
    
settings = Settings()
