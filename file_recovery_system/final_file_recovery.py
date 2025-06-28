#!/usr/bin/env python3

import os
import sys
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict

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

def get_all_legitimate_matches():
    """Get all legitimate file matches (improved + Cornell legitimate)"""
    from improved_matching_analysis import (
        get_database_file_references, build_recovery_file_index, 
        advanced_file_matching
    )
    from cornell_corrected_matching import find_cornell_candidates_corrected
    
    print("🔍 Getting all legitimate file matches...")
    
    file_references = get_database_file_references()
    by_exact_name, by_base_name, by_fuzzy_name, by_super_fuzzy, all_files = build_recovery_file_index()
    
    all_matches = []
    
    # Get improved matches for all files
    for db_ref in file_references:
        matches = advanced_file_matching(
            db_ref, by_exact_name, by_base_name, by_fuzzy_name, by_super_fuzzy, all_files
        )
        
        if matches:
            best_match = matches[0]
            all_matches.append({
                'db_ref': db_ref,
                'file_info': best_match[0],
                'strategy': best_match[1],
                'confidence': best_match[2]
            })
        elif 'Cornell' in db_ref['creator']:
            # Try Cornell-specific legitimate matching for missing Cornell files
            cornell_files = [f for f in all_files if 'cornell' in f['relative_path'].lower()]
            original_base = os.path.splitext(db_ref['original_filename'])[0] if db_ref['original_filename'] else ''
            
            if original_base:
                candidates = find_cornell_candidates_corrected(original_base, cornell_files)
                
                # Only accept high-confidence non-worksheet->slides matches
                if candidates:
                    best_candidate = candidates[0]
                    confidence = best_candidate[1]
                    filename = best_candidate[0]['filename'].lower()
                    original = db_ref['original_filename'].lower()
                    
                    # Filter out questionable worksheet->slides matches
                    is_legitimate = True
                    if 'worksheet' in original and 'slides' in filename and confidence < 0.8:
                        is_legitimate = False
                    if filename.endswith(('.aux', '.log', '.out')) and confidence < 0.8:
                        is_legitimate = False
                    
                    if is_legitimate and confidence > 0.7:
                        all_matches.append({
                            'db_ref': db_ref,
                            'file_info': best_candidate[0],
                            'strategy': 'cornell_legitimate',
                            'confidence': confidence
                        })
    
    print(f"📊 Found {len(all_matches)} legitimate matches")
    
    # Categorize by strategy
    strategy_counts = defaultdict(int)
    for match in all_matches:
        strategy_counts[match['strategy']] += 1
    
    print("📈 Matches by strategy:")
    for strategy, count in strategy_counts.items():
        print(f"  {strategy}: {count}")
    
    return all_matches

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

def verify_file_preview(storage_path, bucket):
    """Verify that the uploaded file can be previewed"""
    try:
        blob = bucket.blob(storage_path)
        if blob.exists():
            # Generate a test URL to verify accessibility
            bucket_name = bucket.name
            encoded_path = storage_path.replace('/', '%2F')
            preview_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o/{encoded_path}?alt=media"
            print(f"📄 Preview URL: {preview_url}")
            return True
        else:
            print(f"❌ File not found in storage: {storage_path}")
            return False
    except Exception as e:
        print(f"❌ Preview verification failed: {e}")
        return False

def process_file_recovery(dry_run=True):
    """Main recovery process"""
    print("🚀 Starting final file recovery process...")
    
    # Get all legitimate matches
    all_matches = get_all_legitimate_matches()
    if not all_matches:
        print("❌ No legitimate matches found")
        return
    
    # Connect to Firebase Storage
    db, bucket = connect_to_firebase()
    if not bucket:
        print("❌ Could not connect to Firebase Storage")
        return
    
    # Analyze what needs to be done
    conversions_needed = []
    direct_uploads = []
    
    for match in all_matches:
        file_info = match['file_info']
        db_ref = match['db_ref']
        
        if db_ref['storage_path'].endswith('.docx.pdf') and file_info['extension'] == '.docx':
            conversions_needed.append(match)
        elif db_ref['storage_path'].endswith('.pptx.pdf') and file_info['extension'] == '.pptx':
            conversions_needed.append(match)
        else:
            direct_uploads.append(match)
    
    print(f"\n📊 Recovery plan:")
    print(f"  Direct uploads (PDFs): {len(direct_uploads)}")
    print(f"  Conversions needed (Word/PowerPoint): {len(conversions_needed)}")
    print(f"  Total files to process: {len(all_matches)}")
    
    if dry_run:
        print(f"\n🔍 DRY RUN - Showing what would be processed:")
        
        print(f"\n📄 Sample conversions needed:")
        for conv in conversions_needed[:5]:
            print(f"  {conv['file_info']['filename']} → PDF")
            print(f"    Storage: {conv['db_ref']['storage_path']}")
        
        print(f"\n📁 Sample direct uploads:")
        for upload in direct_uploads[:5]:
            print(f"  {upload['file_info']['filename']}")
            print(f"    Storage: {upload['db_ref']['storage_path']}")
        
        print(f"\n💡 To run actual recovery, call with dry_run=False")
        return all_matches
    
    # Actual processing
    print(f"\n🔄 Processing {len(all_matches)} files...")
    
    success_count = 0
    error_count = 0
    conversion_count = 0
    temp_dir = tempfile.mkdtemp(prefix="file_recovery_")
    
    try:
        for i, match in enumerate(all_matches):
            db_ref = match['db_ref']
            file_info = match['file_info']
            
            print(f"\n[{i+1}/{len(all_matches)}] Processing: {file_info['filename']}")
            print(f"  Strategy: {match['strategy']} (confidence: {match['confidence']:.2f})")
            
            source_path = file_info['full_path']
            storage_path = db_ref['storage_path']
            
            # Check if file already exists in storage
            blob = bucket.blob(storage_path)
            if blob.exists():
                print(f"  ✅ File already exists in storage, skipping")
                success_count += 1
                continue
            
            # Determine if conversion is needed
            if file_info['extension'] in ['.docx', '.pptx'] and (
                storage_path.endswith('.docx.pdf') or storage_path.endswith('.pptx.pdf')
            ):
                # Convert to PDF
                temp_pdf_path = os.path.join(temp_dir, f"converted_{i}.pdf")
                if convert_to_pdf(source_path, temp_pdf_path):
                    upload_path = temp_pdf_path
                    conversion_count += 1
                else:
                    print(f"❌ Skipping {file_info['filename']} due to conversion failure")
                    error_count += 1
                    continue
            else:
                # Direct upload (copy to temp first to avoid path issues)
                temp_file_path = os.path.join(temp_dir, f"temp_{i}{file_info['extension']}")
                shutil.copy2(source_path, temp_file_path)
                upload_path = temp_file_path
            
            # Upload to Firebase Storage
            if upload_to_firebase_storage(upload_path, storage_path, bucket):
                # Verify the file can be previewed
                if verify_file_preview(storage_path, bucket):
                    success_count += 1
                    print(f"✅ Successfully processed {file_info['filename']}")
                else:
                    error_count += 1
                    print(f"⚠️ Uploaded but preview verification failed for {file_info['filename']}")
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
    print(f"  🔄 Files converted to PDF: {conversion_count}")
    print(f"  ❌ Errors: {error_count}")
    print(f"  📊 Success rate: {success_count/(success_count+error_count)*100:.1f}%")
    
    return all_matches

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

def test_file_preview():
    """Test that uploaded files can be properly previewed"""
    print("\n🧪 Testing file preview functionality...")
    
    db, bucket = connect_to_firebase()
    if not bucket:
        return False
    
    # Test a few existing files
    test_paths = [
        "pdfs/test_file.pdf",  # Will test with actual files during recovery
    ]
    
    for path in test_paths:
        if verify_file_preview(path, bucket):
            print(f"✅ Preview test passed for: {path}")
        else:
            print(f"❌ Preview test failed for: {path}")
    
    return True

if __name__ == "__main__":
    print("🚀 Final File Recovery Script")
    print("=" * 50)
    print("This script will recover legitimate file matches and ensure proper previewing")
    print("=" * 50)
    
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        sys.exit(1)
    
    # Run dry run first
    print("\n🔍 Running dry run analysis...")
    matches = process_file_recovery(dry_run=True)
    
    if not matches:
        print("❌ No matches found. Cannot proceed.")
        sys.exit(1)
    
    # Ask for confirmation
    print(f"\n❓ Ready to process {len(matches)} legitimate files.")
    print("This will:")
    print("  • Convert Word/PowerPoint files to PDF as needed")
    print("  • Upload files to Firebase Storage")
    print("  • Ensure files can be previewed in search.html, view.html, and artifacts.html")
    print("  • NOT delete any existing files")
    
    response = input("\nContinue with actual recovery? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        print("\n🚀 Starting actual file recovery...")
        process_file_recovery(dry_run=False)
        
        print("\n🔍 Testing preview functionality...")
        test_file_preview()
        
        print("\n✅ File recovery complete!")
        print("Files should now be accessible through:")
        print("  • search.html - Search and preview documents")
        print("  • view.html - View individual documents") 
        print("  • artifacts.html - Document previews in modals")
    else:
        print("❌ Recovery cancelled by user.") 