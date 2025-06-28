#!/usr/bin/env python3

import os
import sys
import re
from collections import defaultdict, Counter

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def connect_to_real_firebase():
    """Connect to the real Firebase database (not mock)"""
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
    
    return db, bucket

def get_missing_files():
    """Get files that are still missing after improved matching"""
    # Import the analysis functions from the other script
    from improved_matching_analysis import (
        get_database_file_references, build_recovery_file_index, 
        advanced_file_matching, extract_filename_parts
    )
    
    file_references = get_database_file_references()
    by_exact_name, by_base_name, by_fuzzy_name, by_super_fuzzy, all_files = build_recovery_file_index()
    
    still_missing = []
    
    for db_ref in file_references:
        matches = advanced_file_matching(
            db_ref, by_exact_name, by_base_name, by_fuzzy_name, by_super_fuzzy, all_files
        )
        
        if not matches:
            still_missing.append(db_ref)
    
    return still_missing, all_files

def analyze_missing_patterns(missing_files, all_recovery_files):
    """Analyze patterns in missing files"""
    print(f"🔍 Analyzing {len(missing_files)} missing files...")
    
    # Pattern analysis
    patterns = {
        'by_creator': defaultdict(list),
        'by_extension': defaultdict(list),
        'by_filename_length': defaultdict(list),
        'by_path_structure': defaultdict(list),
        'with_special_chars': [],
        'with_numbers': [],
        'with_parentheses': [],
        'with_spaces': [],
        'potential_duplicates': [],
        'short_names': [],
        'long_names': []
    }
    
    for miss in missing_files:
        storage_path = miss['storage_path']
        original_filename = miss['original_filename']
        creator = miss['creator']
        
        # Group by creator
        patterns['by_creator'][creator].append(miss)
        
        # Group by extension
        filename = os.path.basename(storage_path)
        if filename.endswith('.docx.pdf'):
            patterns['by_extension']['docx.pdf'].append(miss)
        elif filename.endswith('.pptx.pdf'):
            patterns['by_extension']['pptx.pdf'].append(miss)
        elif filename.endswith('.pdf'):
            patterns['by_extension']['pdf'].append(miss)
        else:
            patterns['by_extension']['other'].append(miss)
        
        # Analyze filename characteristics
        base_name = os.path.splitext(original_filename)[0] if original_filename else os.path.splitext(filename)[0]
        
        if len(base_name) < 10:
            patterns['short_names'].append(miss)
        elif len(base_name) > 50:
            patterns['long_names'].append(miss)
        
        if re.search(r'[^a-zA-Z0-9\s\-_.]', base_name):
            patterns['with_special_chars'].append(miss)
        
        if re.search(r'\d', base_name):
            patterns['with_numbers'].append(miss)
        
        if '(' in base_name or ')' in base_name:
            patterns['with_parentheses'].append(miss)
        
        if ' ' in base_name:
            patterns['with_spaces'].append(miss)
        
        # Check for potential duplicates in path
        path_parts = storage_path.split('/')
        if len(path_parts) > 2:
            filename_part = path_parts[-1]
            for part in path_parts[:-1]:
                if filename_part in part or part in filename_part:
                    patterns['potential_duplicates'].append(miss)
                    break
    
    # Print analysis
    print(f"\n📊 Missing Files Analysis:")
    print(f"=" * 50)
    
    print(f"\n👥 By Creator:")
    for creator, files in sorted(patterns['by_creator'].items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {creator}: {len(files)} files")
        if len(files) <= 3:  # Show all if 3 or fewer
            for f in files:
                print(f"    - {f['original_filename'] or os.path.basename(f['storage_path'])}")
        else:  # Show first 3
            for f in files[:3]:
                print(f"    - {f['original_filename'] or os.path.basename(f['storage_path'])}")
            print(f"    ... and {len(files) - 3} more")
    
    print(f"\n📄 By Extension:")
    for ext, files in patterns['by_extension'].items():
        print(f"  {ext}: {len(files)} files")
        if files:
            sample = files[0]
            print(f"    Sample: {sample['original_filename'] or os.path.basename(sample['storage_path'])}")
    
    print(f"\n🔤 Filename Characteristics:")
    print(f"  Short names (<10 chars): {len(patterns['short_names'])}")
    print(f"  Long names (>50 chars): {len(patterns['long_names'])}")
    print(f"  With special characters: {len(patterns['with_special_chars'])}")
    print(f"  With numbers: {len(patterns['with_numbers'])}")
    print(f"  With parentheses: {len(patterns['with_parentheses'])}")
    print(f"  With spaces: {len(patterns['with_spaces'])}")
    print(f"  Potential path duplicates: {len(patterns['potential_duplicates'])}")
    
    return patterns

def manual_search_for_missing(missing_files, all_recovery_files):
    """Manually search for some missing files to understand why they're not found"""
    print(f"\n🔍 Manual search for missing files:")
    print(f"=" * 50)
    
    # Take first 10 missing files and try to find them manually
    for i, miss in enumerate(missing_files[:10]):
        print(f"\n[{i+1}] Searching for: {miss['original_filename'] or os.path.basename(miss['storage_path'])}")
        
        # Extract search terms
        original = miss['original_filename']
        storage_name = os.path.basename(miss['storage_path'])
        
        search_terms = []
        if original:
            search_terms.append(os.path.splitext(original)[0])
        search_terms.append(os.path.splitext(storage_name)[0])
        
        # Remove duplicates
        search_terms = list(set(search_terms))
        
        found_candidates = []
        
        for search_term in search_terms:
            if len(search_term) < 3:  # Skip very short terms
                continue
                
            # Search for partial matches
            for file_info in all_recovery_files:
                file_base = file_info['base_name']
                
                # Try different matching strategies
                if search_term.lower() in file_base.lower():
                    similarity = len(search_term) / len(file_base) if len(file_base) > 0 else 0
                    found_candidates.append((file_info, similarity, 'contains'))
                elif file_base.lower() in search_term.lower():
                    similarity = len(file_base) / len(search_term) if len(search_term) > 0 else 0
                    found_candidates.append((file_info, similarity, 'contained_in'))
                
                # Try fuzzy matching on words
                search_words = search_term.lower().split()
                file_words = file_base.lower().split()
                
                common_words = set(search_words) & set(file_words)
                if len(common_words) > 0 and len(search_words) > 1:
                    similarity = len(common_words) / max(len(search_words), len(file_words))
                    if similarity > 0.3:
                        found_candidates.append((file_info, similarity, 'word_overlap'))
        
        # Sort candidates by similarity
        found_candidates.sort(key=lambda x: x[1], reverse=True)
        
        if found_candidates:
            print(f"  Found {len(found_candidates)} potential matches:")
            for candidate, similarity, method in found_candidates[:3]:  # Show top 3
                print(f"    {similarity:.2f} ({method}): {candidate['relative_path']}")
        else:
            print(f"  No potential matches found")

def suggest_additional_strategies(missing_files):
    """Suggest additional strategies for finding missing files"""
    print(f"\n💡 Suggested Additional Strategies:")
    print(f"=" * 50)
    
    # Analyze the missing files to suggest strategies
    strategies = []
    
    # Check for files with very specific patterns
    cornell_files = [f for f in missing_files if 'Cornell' in f['creator']]
    if cornell_files:
        strategies.append(f"1. Cornell-specific search: {len(cornell_files)} files from Cornell might have different naming conventions")
    
    # Check for files with class/week patterns
    class_pattern_files = [f for f in missing_files if re.search(r'(class|week|lecture|lab)\s*\d+', f['original_filename'] or f['storage_path'], re.IGNORECASE)]
    if class_pattern_files:
        strategies.append(f"2. Class/Week pattern matching: {len(class_pattern_files)} files follow class/week numbering patterns")
    
    # Check for files with date patterns
    date_pattern_files = [f for f in missing_files if re.search(r'\d{1,2}[-/]\d{1,2}|\d{4}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec', f['original_filename'] or f['storage_path'], re.IGNORECASE)]
    if date_pattern_files:
        strategies.append(f"3. Date pattern matching: {len(date_pattern_files)} files contain dates")
    
    # Check for very short filenames
    short_files = [f for f in missing_files if len(os.path.splitext(f['original_filename'] or os.path.basename(f['storage_path']))[0]) < 8]
    if short_files:
        strategies.append(f"4. Short filename expansion: {len(short_files)} files have very short names that might need context-based matching")
    
    # Check for files with version numbers or copies
    version_files = [f for f in missing_files if re.search(r'\([\d\s]+\)|copy|version|v\d+', f['original_filename'] or f['storage_path'], re.IGNORECASE)]
    if version_files:
        strategies.append(f"5. Version/copy handling: {len(version_files)} files appear to be copies or versions")
    
    for strategy in strategies:
        print(f"  {strategy}")
    
    if not strategies:
        print("  No obvious patterns detected. Manual review may be needed.")

if __name__ == "__main__":
    print("🚀 Missing Files Pattern Analysis")
    print("=" * 50)
    
    try:
        missing_files, all_recovery_files = get_missing_files()
        
        print(f"📊 Found {len(missing_files)} missing files to analyze")
        
        patterns = analyze_missing_patterns(missing_files, all_recovery_files)
        manual_search_for_missing(missing_files, all_recovery_files)
        suggest_additional_strategies(missing_files)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc() 