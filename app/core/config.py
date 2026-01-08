from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Vector Store
    vector_store_path: str = "./data/vector_store"
    
    # Document Processing - INCREASED OVERLAP
    chunk_size: int = 500
    chunk_overlap: int = 200  # Increased from 100
    
    upload_folder: str = "./data/documents"
    
    # Voyage AI
    voyage_api_key: str = "pa-iznSYTT-ntR0a_wgXP29udIem0x9WvVt0U6cKb6BAFX"
    
    # OpenAI (optional fallback)
    openai_api_key: str = ""
    
    # Authentication
    secret_key: str = "your-super-secret-jwt-key-change-this-in-production"
    debug: bool = True
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    class Config:
        env_file = ".env"

settings = Settings()
