#!/usr/bin/env python3

import os
import sys
import time
from collections import defaultdict

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def connect_to_firebase():
    """Connect to Firebase services"""
    # Set up environment to force real services
    os.environ['USE_MOCK_SERVICES'] = 'False'
    
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

def get_all_documents_from_firestore(db):
    """Get all documents from the Firestore 'documents' collection"""
    print("🔍 Fetching all documents from Firestore...")
    
    try:
        docs = db.collection('documents').stream()
        documents = []
        
        for doc in docs:
            doc_data = doc.to_dict()
            doc_data['id'] = doc.id
            documents.append(doc_data)
        
        print(f"📊 Found {len(documents)} documents in Firestore")
        return documents
        
    except Exception as e:
        print(f"❌ Error fetching documents from Firestore: {e}")
        return []

def check_file_exists_in_storage(storage_path, bucket):
    """Check if a file exists in Firebase Storage"""
    try:
        blob = bucket.blob(storage_path)
        return blob.exists()
    except Exception as e:
        print(f"❌ Error checking storage for {storage_path}: {e}")
        return False

def analyze_storage_paths(documents):
    """Analyze storage path patterns to understand naming conventions"""
    print("\n🔍 Analyzing storage path patterns...")
    
    path_patterns = defaultdict(int)
    creators = defaultdict(int)
    extensions = defaultdict(int)
    duplicated_names = []
    
    for doc in documents:
        storage_path = doc.get('storage_path', '')
        creator = doc.get('creator', 'Unknown')
        original_filename = doc.get('original_filename', '')
        name = doc.get('name', '')
        
        creators[creator] += 1
        
        if storage_path:
            # Extract pattern
            if storage_path.startswith('pdfs/'):
                path_parts = storage_path[5:].split('_', 2)  # Remove 'pdfs/' and split into max 3 parts
                if len(path_parts) >= 3:
                    # Format: pdfs/{creator}_{filename}_{clean_filename}
                    creator_part, filename_part, clean_filename_part = path_parts[0], path_parts[1], path_parts[2]
                    
                    # Check for duplicated names (the issue you mentioned)
                    if filename_part == clean_filename_part.split('.')[0]:  # Compare without extension
                        duplicated_names.append({
                            'doc_id': doc['id'],
                            'storage_path': storage_path,
                            'creator': creator,
                            'filename_part': filename_part,
                            'clean_filename_part': clean_filename_part,
                            'original_filename': original_filename
                        })
                    
                    path_patterns[f"pdfs/{creator_part}_{{filename}}_{{clean_filename}}"] += 1
            
            # Extract extension
            if '.' in storage_path:
                ext = os.path.splitext(storage_path)[1]
                extensions[ext] += 1
    
    print(f"\n📈 Storage Path Analysis:")
    print(f"  Total documents: {len(documents)}")
    print(f"  Documents with storage_path: {sum(1 for doc in documents if doc.get('storage_path'))}")
    print(f"  Documents without storage_path: {sum(1 for doc in documents if not doc.get('storage_path'))}")
    
    print(f"\n👥 Documents by Creator:")
    for creator, count in sorted(creators.items(), key=lambda x: x[1], reverse=True):
        print(f"  {creator}: {count}")
    
    print(f"\n📄 File Extensions in Storage:")
    for ext, count in sorted(extensions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ext}: {count}")
    
    print(f"\n🔄 Duplicated Names Found: {len(duplicated_names)}")
    if duplicated_names:
        print("  Sample duplicated names:")
        for dup in duplicated_names[:5]:
            print(f"    {dup['storage_path']}")
            print(f"      → filename_part: '{dup['filename_part']}'")
            print(f"      → clean_filename_part: '{dup['clean_filename_part']}'")
            print(f"      → original: '{dup['original_filename']}'")
    
    return duplicated_names

def check_missing_files(documents, bucket):
    """Check which documents have missing files in Firebase Storage"""
    print("\n🔍 Checking for missing files in Firebase Storage...")
    
    missing_files = []
    existing_files = []
    no_storage_path = []
    
    total_docs = len(documents)
    batch_size = 50
    
    for i, doc in enumerate(documents):
        if i % batch_size == 0:
            print(f"  Progress: {i}/{total_docs} ({i/total_docs*100:.1f}%)")
        
        storage_path = doc.get('storage_path', '')
        
        if not storage_path:
            no_storage_path.append(doc)
            continue
        
        # Check if file exists in storage
        exists = check_file_exists_in_storage(storage_path, bucket)
        
        if exists:
            existing_files.append(doc)
        else:
            missing_files.append(doc)
        
        # Small delay to avoid overwhelming Firebase
        if i % 10 == 0:
            time.sleep(0.1)
    
    print(f"  Progress: {total_docs}/{total_docs} (100.0%)")
    
    return missing_files, existing_files, no_storage_path

def generate_detailed_report(missing_files, existing_files, no_storage_path, duplicated_names):
    """Generate a detailed report of the findings"""
    total_docs = len(missing_files) + len(existing_files) + len(no_storage_path)
    
    print(f"\n" + "="*80)
    print(f"📊 DETAILED STORAGE VERIFICATION REPORT")
    print(f"="*80)
    
    print(f"\n📈 Summary Statistics:")
    print(f"  Total documents in Firestore: {total_docs}")
    print(f"  ✅ Files exist in storage: {len(existing_files)} ({len(existing_files)/total_docs*100:.1f}%)")
    print(f"  ❌ Files missing from storage: {len(missing_files)} ({len(missing_files)/total_docs*100:.1f}%)")
    print(f"  ⚠️  Documents without storage_path: {len(no_storage_path)} ({len(no_storage_path)/total_docs*100:.1f}%)")
    print(f"  🔄 Documents with duplicated names: {len(duplicated_names)}")
    
    # Group missing files by creator
    missing_by_creator = defaultdict(list)
    for doc in missing_files:
        creator = doc.get('creator', 'Unknown')
        missing_by_creator[creator].append(doc)
    
    print(f"\n❌ COMPLETE LIST OF ALL MISSING FILES ({len(missing_files)} total):")
    print("="*60)
    
    # Show ALL missing files by creator
    for creator, docs in sorted(missing_by_creator.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\n📂 {creator}: {len(docs)} missing files")
        print("-" * 50)
        
        # Show ALL missing files for this creator
        for i, doc in enumerate(docs, 1):
            print(f"  {i:2d}. {doc.get('name', 'Unnamed')} (ID: {doc['id']})")
            print(f"      Storage: {doc.get('storage_path', 'No path')}")
            print(f"      Original: {doc.get('original_filename', 'No original name')}")
            if i < len(docs):  # Don't add extra line after last item
                print()
    
    # Group documents without storage_path by creator
    if no_storage_path:
        no_path_by_creator = defaultdict(list)
        for doc in no_storage_path:
            creator = doc.get('creator', 'Unknown')
            no_path_by_creator[creator].append(doc)
        
        print(f"\n⚠️  Documents Without Storage Path by Creator:")
        for creator, docs in sorted(no_path_by_creator.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"  {creator}: {len(docs)} documents")
            
            # Show sample documents
            for doc in docs[:2]:
                print(f"    - {doc.get('name', 'Unnamed')} (ID: {doc['id']})")
            
            if len(docs) > 2:
                print(f"    ... and {len(docs) - 2} more")
    
    # Success rate comparison with previous recovery report
    if len(missing_files) > 0:
        print(f"\n🔍 Comparison with Previous Recovery Report:")
        print(f"  Previous report: 628 files successfully uploaded (90.9%)")
        print(f"  Current check: {len(existing_files)} files exist in storage ({len(existing_files)/total_docs*100:.1f}%)")
        print(f"  Difference: {len(missing_files)} files appear to be missing")
    
    return {
        'total_docs': total_docs,
        'existing_files': len(existing_files),
        'missing_files': len(missing_files),
        'no_storage_path': len(no_storage_path),
        'missing_by_creator': missing_by_creator,
        'duplicated_names': len(duplicated_names)
    }

def save_report_to_file(missing_files, existing_files, no_storage_path, duplicated_names):
    """Save detailed report to a file"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"COMPLETE_missing_files_report_{timestamp}.md"
    
    with open(filename, 'w') as f:
        f.write(f"# COMPLETE Missing Files Report\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        total_docs = len(missing_files) + len(existing_files) + len(no_storage_path)
        
        f.write(f"## Summary\n\n")
        f.write(f"- **Total documents**: {total_docs}\n")
        f.write(f"- **Files exist in storage**: {len(existing_files)} ({len(existing_files)/total_docs*100:.1f}%)\n")
        f.write(f"- **Files missing from storage**: {len(missing_files)} ({len(missing_files)/total_docs*100:.1f}%)\n")
        f.write(f"- **Documents without storage_path**: {len(no_storage_path)} ({len(no_storage_path)/total_docs*100:.1f}%)\n")
        f.write(f"- **Documents with duplicated names**: {len(duplicated_names)}\n\n")
        
        if missing_files:
            f.write(f"## COMPLETE List of All Missing Files ({len(missing_files)} total)\n\n")
            
            missing_by_creator = defaultdict(list)
            for doc in missing_files:
                creator = doc.get('creator', 'Unknown')
                missing_by_creator[creator].append(doc)
            
            for creator, docs in sorted(missing_by_creator.items(), key=lambda x: len(x[1]), reverse=True):
                f.write(f"### {creator} ({len(docs)} missing)\n\n")
                for i, doc in enumerate(docs, 1):
                    f.write(f"{i:2d}. **{doc.get('name', 'Unnamed')}** (ID: `{doc['id']}`)\n")
                    f.write(f"    - Storage: `{doc.get('storage_path', 'No path')}`\n")
                    f.write(f"    - Original: `{doc.get('original_filename', 'No original name')}`\n")
                    f.write(f"\n")
                f.write(f"\n")
        
        if no_storage_path:
            f.write(f"## Documents Without Storage Path ({len(no_storage_path)} total)\n\n")
            
            no_path_by_creator = defaultdict(list)
            for doc in no_storage_path:
                creator = doc.get('creator', 'Unknown')
                no_path_by_creator[creator].append(doc)
            
            for creator, docs in sorted(no_path_by_creator.items(), key=lambda x: len(x[1]), reverse=True):
                f.write(f"### {creator} ({len(docs)} documents)\n\n")
                for doc in docs:
                    f.write(f"- **{doc.get('name', 'Unnamed')}** (ID: `{doc['id']}`)\n")
                f.write(f"\n")
        
        if duplicated_names:
            f.write(f"## Documents with Duplicated Names ({len(duplicated_names)} total)\n\n")
            for dup in duplicated_names:
                f.write(f"- **{dup['storage_path']}**\n")
                f.write(f"  - Creator: {dup['creator']}\n")
                f.write(f"  - Filename part: `{dup['filename_part']}`\n")
                f.write(f"  - Clean filename part: `{dup['clean_filename_part']}`\n")
                f.write(f"  - Original: `{dup['original_filename']}`\n")
    
    print(f"\n📄 Complete detailed report saved to: {filename}")
    return filename

def main():
    print("🔍 Firebase Storage Verification Tool - COMPLETE MISSING FILES LIST")
    print("="*80)
    print("This script checks all documents in Firestore against Firebase Storage")
    print("and lists EVERY SINGLE missing file by name")
    print("="*80)
    
    # Connect to Firebase
    db, bucket = connect_to_firebase()
    if not db or not bucket:
        print("❌ Could not connect to Firebase. Please check your credentials.")
        return
    
    # Get all documents from Firestore
    documents = get_all_documents_from_firestore(db)
    if not documents:
        print("❌ No documents found in Firestore.")
        return
    
    # Analyze storage path patterns
    duplicated_names = analyze_storage_paths(documents)
    
    # Check for missing files
    missing_files, existing_files, no_storage_path = check_missing_files(documents, bucket)
    
    # Generate report
    report = generate_detailed_report(missing_files, existing_files, no_storage_path, duplicated_names)
    
    # Save detailed report to file
    report_file = save_report_to_file(missing_files, existing_files, no_storage_path, duplicated_names)
    
    print(f"\n✅ Storage verification complete!")
    print(f"📊 {report['existing_files']}/{report['total_docs']} files exist in storage")
    
    if report['missing_files'] > 0:
        print(f"⚠️  {report['missing_files']} files are missing from storage")
        print(f"📄 See {report_file} for the complete list of ALL missing files")
    else:
        print(f"🎉 All files are present in storage!")

if __name__ == "__main__":
    main() 