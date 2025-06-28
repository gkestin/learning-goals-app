#!/usr/bin/env python3

import os
import sys
import shutil
import subprocess
from pathlib import Path
from collections import defaultdict
import tempfile
import time

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Global variable to store soffice path
SOFFICE_PATH = None

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

def get_database_file_references():
    """Get all file references from the database"""
    db, bucket = connect_to_firebase()
    if not db:
        return []
    
    print("🔍 Getting file references from database...")
    
    docs_ref = db.collection('documents')
    docs = docs_ref.stream()
    
    file_references = []
    for doc in docs:
        doc_data = doc.to_dict()
        storage_path = doc_data.get('storage_path', '')
        
        if storage_path:
            file_references.append({
                'doc_id': doc.id,
                'storage_path': storage_path,
                'original_filename': doc_data.get('original_filename', ''),
                'creator': doc_data.get('creator', ''),
                'name': doc_data.get('name', '')
            })
    
    print(f"📊 Found {len(file_references)} file references in database")
    return file_references

def build_recovery_file_index():
    """Build an index of all files in the recovery folder"""
    recovery_path = os.path.expanduser("~/Downloads/recovery_files_growbot")
    print(f"🔍 Indexing recovery folder: {recovery_path}")
    
    # Multiple indexes for different matching strategies
    by_exact_name = {}      # Exact filename match
    by_base_name = {}       # Filename without extension
    by_fuzzy_name = {}      # Simplified name (lowercase, no spaces/special chars)
    
    total_files = 0
    
    for root, dirs, files in os.walk(recovery_path):
        for file in files:
            if file.startswith('.'):  # Skip hidden files
                continue
            
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, recovery_path)
            
            file_info = {
                'full_path': full_path,
                'relative_path': relative_path,
                'filename': file,
                'extension': os.path.splitext(file)[1].lower(),
                'base_name': os.path.splitext(file)[0]
            }
            
            # Index by exact filename
            if file not in by_exact_name:
                by_exact_name[file] = []
            by_exact_name[file].append(file_info)
            
            # Index by base name (without extension)
            base_name = os.path.splitext(file)[0]
            if base_name not in by_base_name:
                by_base_name[base_name] = []
            by_base_name[base_name].append(file_info)
            
            # Index by fuzzy name (simplified)
            fuzzy_name = simplify_filename(base_name)
            if fuzzy_name not in by_fuzzy_name:
                by_fuzzy_name[fuzzy_name] = []
            by_fuzzy_name[fuzzy_name].append(file_info)
            
            total_files += 1
    
    print(f"📊 Indexed {total_files} files")
    print(f"📊 {len(by_exact_name)} unique exact names")
    print(f"📊 {len(by_base_name)} unique base names")
    print(f"📊 {len(by_fuzzy_name)} unique fuzzy names")
    
    return by_exact_name, by_base_name, by_fuzzy_name

def simplify_filename(filename):
    """Simplify filename for fuzzy matching"""
    import re
    # Convert to lowercase, remove spaces and special characters
    simplified = re.sub(r'[^a-z0-9]', '', filename.lower().replace(' ', ''))
    return simplified

def find_matching_file(db_ref, by_exact_name, by_base_name, by_fuzzy_name):
    """Find matching file in recovery folder using multiple strategies"""
    storage_path = db_ref['storage_path']
    original_filename = db_ref['original_filename']
    
    # Extract expected filename from storage path
    expected_filename = os.path.basename(storage_path)
    
    # Determine target file extension based on expected filename
    if expected_filename.endswith('.docx.pdf'):
        target_base = expected_filename[:-8]  # Remove .docx.pdf
        target_exts = ['.docx']
        final_ext = '.docx.pdf'
    elif expected_filename.endswith('.pptx.pdf'):
        target_base = expected_filename[:-8]  # Remove .pptx.pdf
        target_exts = ['.pptx']
        final_ext = '.pptx.pdf'
    elif expected_filename.endswith('.pdf'):
        target_base = expected_filename[:-4]  # Remove .pdf
        target_exts = ['.pdf']
        final_ext = '.pdf'
    else:
        target_base = os.path.splitext(expected_filename)[0]
        target_exts = ['.pdf', '.docx', '.pptx']
        final_ext = '.pdf'
    
    # Strategy 1: Try exact original filename match
    if original_filename:
        orig_base = os.path.splitext(original_filename)[0]
        if orig_base in by_base_name:
            for file_info in by_base_name[orig_base]:
                if file_info['extension'] in target_exts:
                    return file_info, 'original_filename', final_ext
    
    # Strategy 2: Try exact expected base name
    if target_base in by_base_name:
        for file_info in by_base_name[target_base]:
            if file_info['extension'] in target_exts:
                return file_info, 'expected_base', final_ext
    
    # Strategy 3: Try fuzzy matching on original filename
    if original_filename:
        orig_fuzzy = simplify_filename(os.path.splitext(original_filename)[0])
        if orig_fuzzy in by_fuzzy_name:
            for file_info in by_fuzzy_name[orig_fuzzy]:
                if file_info['extension'] in target_exts:
                    return file_info, 'fuzzy_original', final_ext
    
    # Strategy 4: Try fuzzy matching on expected base
    target_fuzzy = simplify_filename(target_base)
    if target_fuzzy in by_fuzzy_name:
        for file_info in by_fuzzy_name[target_fuzzy]:
            if file_info['extension'] in target_exts:
                return file_info, 'fuzzy_expected', final_ext
    
    return None, None, None

def convert_to_pdf(input_path, output_path):
    """Convert Word or PowerPoint file to PDF using LibreOffice"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use LibreOffice headless mode to convert
        global SOFFICE_PATH
        soffice_cmd = SOFFICE_PATH or 'soffice'
        cmd = [
            soffice_cmd,
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
                print(f"✅ Successfully converted to {os.path.basename(output_path)}")
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
    """Upload file to Firebase Storage"""
    try:
        blob = bucket.blob(storage_path)
        blob.upload_from_filename(local_path)
        print(f"✅ Uploaded to Firebase Storage: {storage_path}")
        return True
    except Exception as e:
        print(f"❌ Upload failed for {storage_path}: {e}")
        return False

def process_file_recovery(dry_run=True):
    """Main recovery process"""
    print("🚀 Starting file recovery process...")
    
    # Get database references
    file_references = get_database_file_references()
    if not file_references:
        print("❌ No file references found in database")
        return
    
    # Build recovery file index
    by_exact_name, by_base_name, by_fuzzy_name = build_recovery_file_index()
    
    # Connect to Firebase Storage
    db, bucket = connect_to_firebase()
    if not bucket:
        print("❌ Could not connect to Firebase Storage")
        return
    
    # Process each file reference
    matches = []
    missing = []
    
    print(f"\n🔍 Matching {len(file_references)} database references to recovery files...")
    
    for i, db_ref in enumerate(file_references):
        if i % 50 == 0:
            print(f"Progress: {i}/{len(file_references)}")
        
        file_info, match_strategy, final_ext = find_matching_file(
            db_ref, by_exact_name, by_base_name, by_fuzzy_name
        )
        
        if file_info:
            matches.append({
                'db_ref': db_ref,
                'file_info': file_info,
                'match_strategy': match_strategy,
                'final_ext': final_ext
            })
        else:
            missing.append(db_ref)
    
    print(f"\n🎯 Matching results:")
    print(f"  ✅ Found matches: {len(matches)}")
    print(f"  ❌ Missing files: {len(missing)}")
    
    if dry_run:
        print(f"\n🔍 DRY RUN - Showing what would be processed:")
        
        # Show sample matches
        print(f"\n📋 Sample matches (showing first 10):")
        for match in matches[:10]:
            db_ref = match['db_ref']
            file_info = match['file_info']
            print(f"  DB: {db_ref['storage_path']}")
            print(f"  Recovery: {file_info['relative_path']} -> {match['final_ext']}")
            print(f"  Strategy: {match['match_strategy']}")
            print()
        
        # Show conversion needs
        conversions_needed = [m for m in matches if not m['file_info']['filename'].endswith('.pdf')]
        print(f"📄 Files needing conversion: {len(conversions_needed)}")
        
        # Show missing files
        if missing:
            print(f"\n❌ Sample missing files:")
            for miss in missing[:5]:
                print(f"  - {miss['storage_path']}")
        
        print(f"\n💡 To run actual recovery, call with dry_run=False")
        return matches, missing
    
    # Actual processing
    print(f"\n🔄 Processing {len(matches)} files...")
    
    success_count = 0
    error_count = 0
    temp_dir = tempfile.mkdtemp(prefix="file_recovery_")
    
    try:
        for i, match in enumerate(matches):
            db_ref = match['db_ref']
            file_info = match['file_info']
            final_ext = match['final_ext']
            
            print(f"\n[{i+1}/{len(matches)}] Processing: {file_info['filename']}")
            
            source_path = file_info['full_path']
            storage_path = db_ref['storage_path']
            
            # Determine if conversion is needed
            if file_info['extension'] == '.pdf':
                # PDF file - copy directly
                temp_pdf_path = os.path.join(temp_dir, f"temp_{i}.pdf")
                shutil.copy2(source_path, temp_pdf_path)
                upload_path = temp_pdf_path
            else:
                # Word/PowerPoint file - convert to PDF
                temp_pdf_path = os.path.join(temp_dir, f"converted_{i}.pdf")
                if convert_to_pdf(source_path, temp_pdf_path):
                    upload_path = temp_pdf_path
                else:
                    print(f"❌ Skipping {file_info['filename']} due to conversion failure")
                    error_count += 1
                    continue
            
            # Upload to Firebase Storage
            if upload_to_firebase_storage(upload_path, storage_path, bucket):
                success_count += 1
                print(f"✅ Successfully processed {file_info['filename']}")
            else:
                error_count += 1
                print(f"❌ Upload failed for {file_info['filename']}")
            
            # Add small delay to avoid overwhelming the system
            time.sleep(0.1)
    
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"\n🎯 Recovery complete!")
    print(f"  ✅ Successfully processed: {success_count}")
    print(f"  ❌ Errors: {error_count}")
    print(f"  📊 Success rate: {success_count/(success_count+error_count)*100:.1f}%")
    
    return matches, missing

def check_prerequisites():
    """Check if all required tools are available"""
    print("🔧 Checking prerequisites...")
    
    # Check LibreOffice
    soffice_paths = [
        'soffice',  # In PATH
        '/usr/local/bin/soffice',  # Homebrew location
        '/Applications/LibreOffice.app/Contents/MacOS/soffice'  # Direct app location
    ]
    
    soffice_found = None
    for soffice_path in soffice_paths:
        try:
            result = subprocess.run([soffice_path, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ LibreOffice found at: {soffice_path}")
                soffice_found = soffice_path
                global SOFFICE_PATH
                SOFFICE_PATH = soffice_path
                break
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    
    if not soffice_found:
        print("❌ LibreOffice not found. Please install it:")
        print("   macOS: brew install --cask libreoffice")
        print("   or download from: https://www.libreoffice.org/download/")
        return False
    
    # Check credentials
    cred_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if not cred_path or not os.path.exists(cred_path):
        print(f"❌ Firebase credentials not found: {cred_path}")
        return False
    else:
        print("✅ Firebase credentials found")
    
    # Check recovery folder
    recovery_path = os.path.expanduser("~/Downloads/recovery_files_growbot")
    if not os.path.exists(recovery_path):
        print(f"❌ Recovery folder not found: {recovery_path}")
        return False
    else:
        print("✅ Recovery folder found")
    
    return True

if __name__ == "__main__":
    print("🚀 File Recovery Script")
    print("=" * 50)
    
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        sys.exit(1)
    
    # Run dry run first
    print("\n🔍 Running dry run analysis...")
    matches, missing = process_file_recovery(dry_run=True)
    
    if not matches:
        print("❌ No matches found. Cannot proceed.")
        sys.exit(1)
    
    # Ask for confirmation
    print(f"\n❓ Ready to process {len(matches)} files.")
    response = input("Continue with actual recovery? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        print("\n🚀 Starting actual file recovery...")
        process_file_recovery(dry_run=False)
    else:
        print("❌ Recovery cancelled by user.") 