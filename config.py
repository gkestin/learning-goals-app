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
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    GOOGLE_APPLICATION_CREDENTIALS = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    FIREBASE_STORAGE_BUCKET = os.environ.get('FIREBASE_STORAGE_BUCKET')
    UPLOAD_FOLDER = os.path.join('instance', 'uploads')
    ALLOWED_EXTENSIONS = {'pdf'} 