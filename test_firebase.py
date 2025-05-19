import os
import firebase_admin
from firebase_admin import credentials, storage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get credentials path
cred_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
print(f"Using credentials from: {cred_path}")

# Test different bucket formats
bucket_formats = [
    "learninggoals2",  # Raw project name
    "learninggoals2.appspot.com",  # Traditional format
    "learninggoals2.firebasestorage.app",  # Storage URL format
]

for format in bucket_formats:
    print(f"\n\n======= TESTING BUCKET FORMAT: {format} =======")
    
    # Reset Firebase app
    if firebase_admin._apps:
        for app in list(firebase_admin._apps.keys()):
            firebase_admin.delete_app(firebase_admin._apps[app])
    
    try:
        print(f"Initializing Firebase with bucket: {format}")
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {
            'storageBucket': format
        })
        
        bucket = storage.bucket()
        print(f"✅ SUCCESS! Bucket initialized: {bucket.name}")
        
        # Test listing files in bucket
        print("Testing bucket access by listing files:")
        blobs = list(bucket.list_blobs(max_results=5))
        print(f"Found {len(blobs)} files in bucket")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")

print("\n\nTest complete") 