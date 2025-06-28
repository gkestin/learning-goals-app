#!/usr/bin/env python3

import os
import sys
from collections import defaultdict, Counter
import re

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def connect_to_real_firebase():
    """Connect to the real Firebase database (not mock)"""
    # Set up environment
    os.environ['USE_MOCK_SERVICES'] = 'False'  # Force real services
    
    # Import and initialize Firebase
    from app.firebase_service import db, bucket, init_app
    from flask import Flask
    
    app = Flask(__name__)
    app.config.from_object('config.Config')
    
    # Force initialization of real services
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

def analyze_database_files():
    """Analyze all file references in the database"""
    db, bucket = connect_to_real_firebase()
    if not db:
        return
    
    print("🔍 Analyzing file references in database...")
    
    # Get all documents from the database
    docs_ref = db.collection('documents')
    docs = docs_ref.stream()
    
    file_references = []
    storage_paths = []
    creators = set()
    
    doc_count = 0
    for doc in docs:
        doc_count += 1
        doc_data = doc.to_dict()
        
        storage_path = doc_data.get('storage_path', '')
        original_filename = doc_data.get('original_filename', '')
        creator = doc_data.get('creator', '')
        name = doc_data.get('name', '')
        
        if storage_path:
            storage_paths.append(storage_path)
            file_references.append({
                'doc_id': doc.id,
                'storage_path': storage_path,
                'original_filename': original_filename,
                'creator': creator,
                'name': name
            })
            
        if creator:
            creators.add(creator)
    
    print(f"📊 Found {doc_count} documents in database")
    print(f"📊 Found {len(file_references)} documents with storage paths")
    print(f"📊 Found {len(creators)} unique creators")
    
    # Analyze file patterns
    print("\n🔍 Analyzing file naming patterns...")
    
    # Group by file extensions referenced in storage_path
    extension_counts = Counter()
    duplicated_paths = []
    pdf_extensions = []  # Files that should be .docx.pdf or .pptx.pdf
    
    for ref in file_references:
        path = ref['storage_path']
        
        # Check for duplicated filenames in path
        if '/' in path:
            path_parts = path.split('/')
            filename = path_parts[-1]
            if filename in path_parts[:-1]:
                duplicated_paths.append(ref)
        
        # Analyze extensions
        if path.endswith('.pdf'):
            extension_counts['.pdf'] += 1
            # Check if it's a converted file
            if path.endswith('.docx.pdf'):
                extension_counts['.docx.pdf'] += 1
                pdf_extensions.append(ref)
            elif path.endswith('.pptx.pdf'):
                extension_counts['.pptx.pdf'] += 1
                pdf_extensions.append(ref)
        else:
            # Extract extension
            ext = os.path.splitext(path)[1] if '.' in path else '[no extension]'
            extension_counts[ext] += 1
    
    print(f"\n📈 File extension counts:")
    for ext, count in extension_counts.most_common():
        print(f"  {ext}: {count}")
    
    print(f"\n🔧 Found {len(duplicated_paths)} files with duplicated paths")
    if duplicated_paths:
        print("Sample duplicated paths:")
        for ref in duplicated_paths[:5]:
            print(f"  - {ref['storage_path']}")
    
    print(f"\n📄 Found {len(pdf_extensions)} files that were converted from Word/PowerPoint:")
    if pdf_extensions:
        print("Sample converted files:")
        for ref in pdf_extensions[:10]:
            print(f"  - {ref['storage_path']} (original: {ref['original_filename']})")
    
    # Show samples by creator
    print(f"\n👥 Sample files by creator:")
    creator_samples = defaultdict(list)
    for ref in file_references:
        creator_samples[ref['creator']].append(ref)
    
    for creator in sorted(creators)[:5]:  # Show first 5 creators
        samples = creator_samples[creator][:3]  # Show 3 samples per creator
        print(f"\n  {creator} ({len(creator_samples[creator])} files):")
        for ref in samples:
            print(f"    - {ref['storage_path']}")
    
    return file_references, duplicated_paths, pdf_extensions

def find_matching_files_in_recovery(file_references):
    """Find matching files in the recovery folder"""
    recovery_path = os.path.expanduser("~/Downloads/recovery_files_growbot")
    
    print(f"\n🔍 Scanning recovery folder: {recovery_path}")
    
    # Build a map of all files in recovery folder
    recovery_files = {}
    total_files = 0
    
    for root, dirs, files in os.walk(recovery_path):
        for file in files:
            if file.startswith('.'):  # Skip hidden files
                continue
            
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, recovery_path)
            
            # Store by just the filename for easier matching
            base_name = os.path.splitext(file)[0]  # Name without extension
            full_name = file  # Name with extension
            
            if base_name not in recovery_files:
                recovery_files[base_name] = []
            recovery_files[base_name].append({
                'full_path': full_path,
                'relative_path': relative_path,
                'filename': full_name,
                'extension': os.path.splitext(file)[1].lower()
            })
            total_files += 1
    
    print(f"📊 Found {total_files} files in recovery folder")
    print(f"📊 Found {len(recovery_files)} unique base names")
    
    # Match database references to recovery files
    matches = []
    missing = []
    
    for ref in file_references:
        storage_path = ref['storage_path']
        original_filename = ref['original_filename']
        
        # Extract the expected filename from storage path
        expected_filename = os.path.basename(storage_path)
        
        # Try different matching strategies
        matched = False
        
        # Strategy 1: Try to match the exact expected filename
        expected_base = os.path.splitext(expected_filename)[0]
        
        # Handle .docx.pdf and .pptx.pdf cases
        if expected_filename.endswith('.docx.pdf'):
            # Look for .docx file
            search_base = expected_base[:-4]  # Remove .pdf part
            target_ext = '.docx'
        elif expected_filename.endswith('.pptx.pdf'):
            # Look for .pptx file
            search_base = expected_base[:-4]  # Remove .pdf part
            target_ext = '.pptx'
        elif expected_filename.endswith('.pdf'):
            # Look for .pdf file
            search_base = expected_base
            target_ext = '.pdf'
        else:
            search_base = expected_base
            target_ext = None
        
        # Strategy 2: Try matching by original filename
        if not matched and original_filename:
            orig_base = os.path.splitext(original_filename)[0]
            if orig_base in recovery_files:
                for recovery_file in recovery_files[orig_base]:
                    if target_ext is None or recovery_file['extension'] == target_ext:
                        matches.append({
                            'database_ref': ref,
                            'recovery_file': recovery_file,
                            'match_strategy': 'original_filename'
                        })
                        matched = True
                        break
        
        # Strategy 3: Try matching by search_base
        if not matched and search_base in recovery_files:
            for recovery_file in recovery_files[search_base]:
                if target_ext is None or recovery_file['extension'] == target_ext:
                    matches.append({
                        'database_ref': ref,
                        'recovery_file': recovery_file,
                        'match_strategy': 'storage_path_base'
                    })
                    matched = True
                    break
        
        if not matched:
            missing.append(ref)
    
    print(f"\n🎯 Matching results:")
    print(f"  ✅ Found matches: {len(matches)}")
    print(f"  ❌ Missing files: {len(missing)}")
    
    # Show sample matches
    print(f"\n📋 Sample matches:")
    for match in matches[:10]:
        db_ref = match['database_ref']
        rec_file = match['recovery_file']
        print(f"  DB: {db_ref['storage_path']}")
        print(f"  Recovery: {rec_file['relative_path']} ({match['match_strategy']})")
        print()
    
    if missing:
        print(f"\n❌ Sample missing files:")
        for miss in missing[:5]:
            print(f"  - {miss['storage_path']} (original: {miss['original_filename']})")
    
    return matches, missing

if __name__ == "__main__":
    print("🚀 Starting database file analysis...")
    
    try:
        file_references, duplicated_paths, pdf_extensions = analyze_database_files()
        matches, missing = find_matching_files_in_recovery(file_references)
        
        print(f"\n🎯 Summary:")
        print(f"  Total database references: {len(file_references)}")
        print(f"  Duplicated path issues: {len(duplicated_paths)}")
        print(f"  Converted files (.docx.pdf/.pptx.pdf): {len(pdf_extensions)}")
        print(f"  Files found in recovery: {len(matches)}")
        print(f"  Files missing from recovery: {len(missing)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc() 