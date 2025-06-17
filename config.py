import os
from dotenv import load_dotenv

# Make sure we load environment variables here too
load_dotenv()

# Print the environment variables as they're loaded into Config
print("\n==== CONFIG.PY ENVIRONMENT LOADING ====")
print(f"FIREBASE_STORAGE_BUCKET from env: {os.environ.get('FIREBASE_STORAGE_BUCKET', 'Not set in env')}")
print(f"GOOGLE_APPLICATION_CREDENTIALS from env: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'Not set in env')}")
print("==== END CONFIG.PY ====\n")

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-for-learning-goals'
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max upload size
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    GOOGLE_APPLICATION_CREDENTIALS = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    FIREBASE_STORAGE_BUCKET = os.environ.get('FIREBASE_STORAGE_BUCKET')
    UPLOAD_FOLDER = 'instance/uploads'
    ALLOWED_EXTENSIONS = {'pdf'}
    
    # Domain and deployment configuration
    CUSTOM_DOMAIN = os.environ.get('CUSTOM_DOMAIN', 'mathmatic.org')
    FORCE_HTTPS = os.environ.get('FORCE_HTTPS', 'true').lower() == 'true'
    
    # CORS settings for custom domain
    CORS_ORIGINS = [
        'https://mathmatic.org',
        'https://www.mathmatic.org',
        'http://localhost:3000',  # For development
        'http://127.0.0.1:3000'   # For development
    ] 