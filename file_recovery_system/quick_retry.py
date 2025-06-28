#!/usr/bin/env python3

import os
import sys
import time

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def connect_to_firebase():
    """Connect to Firebase services"""
    # Set up environment
    os.environ['USE_MOCK_SERVICES'] = 'False'  # Force real services
    
    # Import and initialize Firebase
    import firebase_admin
    from firebase_admin import credentials, firestore, storage
    
    cred_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if not cred_path or not os.path.exists(cred_path):
        print(f"❌ Credentials file not found: {cred_path}")
        return None, None
    
    # Initialize Firebase Admin if not already done
    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {
            'storageBucket': os.environ.get('FIREBASE_STORAGE_BUCKET')
        })
    
    # Get Firestore and Storage clients
    db = firestore.client()
    bucket = storage.bucket()
    
    print(f"✅ Connected to Firebase project: {os.environ.get('FIREBASE_STORAGE_BUCKET')}")
    return db, bucket

def upload_to_firebase_storage(local_path, storage_path, bucket):
    """Upload file to Firebase Storage with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            blob = bucket.blob(storage_path)
            blob.upload_from_filename(local_path)
            print(f"✅ Uploaded to Firebase Storage: {storage_path}")
            return True
        except Exception as e:
            print(f"❌ Upload attempt {attempt + 1} failed for {storage_path}: {e}")
            if attempt < max_retries - 1:
                print(f"⏳ Waiting 5 seconds before retry...")
                time.sleep(5)
            else:
                print(f"❌ All retry attempts failed for {storage_path}")
                return False

def find_file_in_recovery(filename):
    """Find a file in the recovery directory"""
    recovery_path = os.path.expanduser("~/Downloads/recovery_files_growbot")
    
    # Search recursively for the file
    for root, dirs, files in os.walk(recovery_path):
        if filename in files:
            return os.path.join(root, filename)
    
    return None

def quick_retry():
    """Quick retry of the specific files that failed"""
    
    # Known failed files from the previous run (based on timeout errors)
    failed_files = [
        "Lec1A_post_class.pdf",
        "Lecture12.pdf", 
        "Lec9B_post_class.pdf"
    ]
    
    print(f"🔄 Quick retry of {len(failed_files)} specific failed files...")
    
    # Connect to Firebase
    db, bucket = connect_to_firebase()
    if not bucket:
        print("❌ Could not connect to Firebase Storage")
        return
    
    success_count = 0
    
    for i, filename in enumerate(failed_files):
        print(f"\n[{i+1}/{len(failed_files)}] Retrying: {filename}")
        
        # Find the source file
        source_path = find_file_in_recovery(filename)
        if not source_path:
            print(f"  ❌ Source file not found in recovery folder")
            continue
        
        print(f"  📁 Source: {source_path}")
        
        # Generate storage path (assuming standard naming convention)
        if "JCz (UCLA)" in source_path:
            creator = "JCz (UCLA)"
        elif "Marty Weissman" in source_path:
            creator = "Marty Weissman"
        else:
            creator = "Unknown"
        
        # Clean filename for storage path
        clean_filename = filename.replace(' ', '_').replace('(', '').replace(')', '')
        storage_path = f"pdfs/{creator}_{filename}_{clean_filename}"
        
        print(f"  🎯 Target: {storage_path}")
        
        # Check if file already exists
        blob = bucket.blob(storage_path)
        if blob.exists():
            print(f"  ✅ File already exists in storage, skipping")
            success_count += 1
            continue
        
        # Upload with retry logic
        if upload_to_firebase_storage(source_path, storage_path, bucket):
            # Verify the file can be previewed
            bucket_name = bucket.name
            encoded_path = storage_path.replace('/', '%2F')
            preview_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o/{encoded_path}?alt=media"
            print(f"  📄 Preview URL: {preview_url}")
            print(f"  ✅ Successfully processed {filename}")
            success_count += 1
        else:
            print(f"  ❌ Failed to upload {filename}")
        
        # Small delay between uploads
        time.sleep(2)
    
    print(f"\n🎯 Quick retry complete!")
    print(f"  ✅ Successfully processed: {success_count}/{len(failed_files)}")
    print(f"  📊 Success rate: {success_count/len(failed_files)*100:.1f}%")
    
    if success_count == len(failed_files):
        print(f"🎉 All failed files successfully uploaded!")
    else:
        failed_count = len(failed_files) - success_count
        print(f"⚠️  {failed_count} files still failed")

if __name__ == "__main__":
    print("🔄 Quick Retry of Failed File Uploads")
    print("=" * 45)
    
    # Set environment variables
    os.environ['FIREBASE_STORAGE_BUCKET'] = 'learninggoals2.firebasestorage.app'
    os.environ['USE_MOCK_SERVICES'] = 'False'
    
    quick_retry() 