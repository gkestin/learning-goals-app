#!/usr/bin/env python3

import os
import sys
import time
from collections import defaultdict

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

def convert_to_pdf(input_path, output_path):
    """Convert Word or PowerPoint file to PDF using LibreOffice"""
    try:
        import subprocess
        import shutil
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use LibreOffice headless mode to convert
        cmd = [
            'soffice',
            '--headless',
            '--convert-to', 'pdf',
            '--outdir', os.path.dirname(output_path),
            input_path
        ]
        
        print(f"🔄 Converting {os.path.basename(input_path)} to PDF...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            # LibreOffice creates a PDF with the same base name
            input_base = os.path.splitext(os.path.basename(input_path))[0]
            created_pdf = os.path.join(os.path.dirname(output_path), f"{input_base}.pdf")
            
            if os.path.exists(created_pdf):
                # Rename to the desired output filename
                if created_pdf != output_path:
                    shutil.move(created_pdf, output_path)
                return True
            else:
                print(f"❌ PDF not created: {created_pdf}")
                return False
        else:
            print(f"❌ LibreOffice conversion failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ Conversion timeout for {os.path.basename(input_path)}")
        return False
    except Exception as e:
        print(f"❌ Conversion error for {os.path.basename(input_path)}: {e}")
        return False

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

def retry_failed_files():
    """Find and retry files that should exist but don't in Firebase Storage"""
    
    print("🔍 Finding files that failed to upload...")
    
    # Connect to Firebase
    db, bucket = connect_to_firebase()
    if not bucket:
        print("❌ Could not connect to Firebase Storage")
        return
    
    # Get all legitimate matches
    from final_file_recovery import get_all_legitimate_matches
    all_matches = get_all_legitimate_matches()
    
    print(f"📊 Checking {len(all_matches)} legitimate matches for missing files...")
    
    failed_files = []
    
    # Check which files are missing from storage
    for match in all_matches:
        db_ref = match['db_ref']
        file_info = match['file_info']
        storage_path = db_ref['storage_path']
        
        # Check if file exists in storage
        blob = bucket.blob(storage_path)
        if not blob.exists():
            failed_files.append({
                'filename': file_info['filename'],
                'storage_path': storage_path,
                'source_path': file_info['full_path'],
                'strategy': match['strategy'],
                'confidence': match['confidence']
            })
    
    if not failed_files:
        print("🎉 All files are already uploaded successfully!")
        return
    
    print(f"🔄 Found {len(failed_files)} files that need to be uploaded...")
    
    success_count = 0
    total_files = len(failed_files)
    
    for i, failed_file in enumerate(failed_files):
        filename = failed_file['filename']
        storage_path = failed_file['storage_path']
        source_path = failed_file['source_path']
        
        print(f"\n[{i+1}/{total_files}] Uploading: {filename}")
        print(f"  Strategy: {failed_file['strategy']} (confidence: {failed_file['confidence']:.2f})")
        
        if not os.path.exists(source_path):
            print(f"  ❌ Source file not found: {source_path}")
            continue
        
        print(f"  📁 Source: {source_path}")
        print(f"  🎯 Target: {storage_path}")
        
        # Check if conversion is needed
        needs_conversion = False
        upload_path = source_path
        
        if source_path.endswith(('.docx', '.pptx')) and (
            storage_path.endswith('.docx.pdf') or storage_path.endswith('.pptx.pdf')
        ):
            # Need to convert to PDF
            import tempfile
            temp_dir = tempfile.mkdtemp()
            temp_pdf_path = os.path.join(temp_dir, f"converted_{i}.pdf")
            
            if convert_to_pdf(source_path, temp_pdf_path):
                upload_path = temp_pdf_path
                needs_conversion = True
                print(f"  🔄 Converted to PDF: {temp_pdf_path}")
            else:
                print(f"  ❌ Conversion failed for {filename}")
                continue
        
        # Upload with retry logic
        if upload_to_firebase_storage(upload_path, storage_path, bucket):
            # Verify the file can be previewed
            bucket_name = bucket.name
            encoded_path = storage_path.replace('/', '%2F')
            preview_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o/{encoded_path}?alt=media"
            print(f"  📄 Preview URL: {preview_url}")
            print(f"  ✅ Successfully processed {filename}")
            success_count += 1
        else:
            print(f"  ❌ Failed to upload {filename}")
        
        # Clean up temp file if created
        if needs_conversion and os.path.exists(upload_path):
            os.remove(upload_path)
        
        # Small delay between uploads
        time.sleep(1)
    
    print(f"\n🎯 Retry complete!")
    print(f"  ✅ Successfully processed: {success_count}/{total_files}")
    print(f"  📊 Success rate: {success_count/total_files*100:.1f}%")
    
    if success_count == total_files:
        print(f"🎉 All failed files successfully uploaded!")
    else:
        failed_count = total_files - success_count
        print(f"⚠️  {failed_count} files still failed - may need manual investigation")

if __name__ == "__main__":
    print("🔄 Retrying Failed File Uploads")
    print("=" * 40)
    
    # Set environment variables
    os.environ['FIREBASE_STORAGE_BUCKET'] = 'learninggoals2.firebasestorage.app'
    os.environ['USE_MOCK_SERVICES'] = 'False'
    
    retry_failed_files() 