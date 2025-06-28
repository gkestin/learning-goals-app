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
    
    print(f"✅ Connected to Firebase project: {os.environ.get('FIREBASE_STORAGE_BUCKET')}")
    return db, bucket

def get_database_file_references():
    """Get all file references from the database"""
    db, bucket = connect_to_real_firebase()
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
    """Build a comprehensive index of all files in the recovery folder"""
    recovery_path = os.path.expanduser("~/Downloads/recovery_files_growbot")
    print(f"🔍 Indexing recovery folder: {recovery_path}")
    
    # Multiple indexes for different matching strategies
    by_exact_name = {}      # Exact filename match
    by_base_name = {}       # Filename without extension
    by_fuzzy_name = {}      # Simplified name (lowercase, no spaces/special chars)
    by_super_fuzzy = {}     # Even more aggressive simplification
    
    all_files = []
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
            
            all_files.append(file_info)
            
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
            
            # Index by super fuzzy name (very aggressive)
            super_fuzzy_name = super_simplify_filename(base_name)
            if super_fuzzy_name not in by_super_fuzzy:
                by_super_fuzzy[super_fuzzy_name] = []
            by_super_fuzzy[super_fuzzy_name].append(file_info)
            
            total_files += 1
    
    print(f"📊 Indexed {total_files} files")
    print(f"📊 {len(by_exact_name)} unique exact names")
    print(f"📊 {len(by_base_name)} unique base names")
    print(f"📊 {len(by_fuzzy_name)} unique fuzzy names")
    print(f"📊 {len(by_super_fuzzy)} unique super fuzzy names")
    
    return by_exact_name, by_base_name, by_fuzzy_name, by_super_fuzzy, all_files

def simplify_filename(filename):
    """Simplify filename for fuzzy matching"""
    import re
    # Convert to lowercase, remove spaces and special characters
    simplified = re.sub(r'[^a-z0-9]', '', filename.lower().replace(' ', ''))
    return simplified

def super_simplify_filename(filename):
    """Even more aggressive filename simplification"""
    import re
    # Remove common words, numbers, and keep only letters
    simplified = filename.lower()
    # Remove common words
    common_words = ['the', 'and', 'or', 'of', 'to', 'in', 'for', 'with', 'on', 'at', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall']
    for word in common_words:
        simplified = simplified.replace(word, '')
    # Remove all non-letters
    simplified = re.sub(r'[^a-z]', '', simplified)
    return simplified

def extract_filename_parts(storage_path, original_filename):
    """Extract meaningful parts from storage path and original filename"""
    parts = {
        'storage_filename': os.path.basename(storage_path),
        'original_filename': original_filename,
        'storage_base': os.path.splitext(os.path.basename(storage_path))[0],
        'original_base': os.path.splitext(original_filename)[0] if original_filename else '',
    }
    
    # Handle duplicated filenames in storage path
    if '/' in storage_path:
        path_parts = storage_path.split('/')
        filename = path_parts[-1]
        # Check if filename appears earlier in path (duplicated name bug)
        for i, part in enumerate(path_parts[:-1]):
            if filename in part or part in filename:
                parts['has_duplication'] = True
                parts['clean_filename'] = filename
                break
    
    # Handle .docx.pdf and .pptx.pdf extensions
    storage_filename = parts['storage_filename']
    if storage_filename.endswith('.docx.pdf'):
        parts['target_extension'] = '.docx'
        parts['clean_base'] = storage_filename[:-8]  # Remove .docx.pdf
        parts['expected_original'] = parts['clean_base'] + '.docx'
    elif storage_filename.endswith('.pptx.pdf'):
        parts['target_extension'] = '.pptx'
        parts['clean_base'] = storage_filename[:-8]  # Remove .pptx.pdf
        parts['expected_original'] = parts['clean_base'] + '.pptx'
    elif storage_filename.endswith('.pdf'):
        parts['target_extension'] = '.pdf'
        parts['clean_base'] = storage_filename[:-4]  # Remove .pdf
        parts['expected_original'] = parts['clean_base'] + '.pdf'
    
    return parts

def advanced_file_matching(db_ref, by_exact_name, by_base_name, by_fuzzy_name, by_super_fuzzy, all_files):
    """Advanced file matching with multiple strategies"""
    storage_path = db_ref['storage_path']
    original_filename = db_ref['original_filename']
    
    # Extract parts for analysis
    parts = extract_filename_parts(storage_path, original_filename)
    
    matches = []
    
    # Strategy 1: Exact original filename match
    if original_filename and original_filename in by_exact_name:
        for file_info in by_exact_name[original_filename]:
            matches.append((file_info, 'exact_original', 1.0))
    
    # Strategy 2: Exact base name match on original
    if parts['original_base'] and parts['original_base'] in by_base_name:
        for file_info in by_base_name[parts['original_base']]:
            if parts.get('target_extension') and file_info['extension'] == parts['target_extension']:
                matches.append((file_info, 'base_original_correct_ext', 0.95))
            else:
                matches.append((file_info, 'base_original', 0.8))
    
    # Strategy 3: Match using clean base name from storage path
    if parts.get('clean_base') and parts['clean_base'] in by_base_name:
        for file_info in by_base_name[parts['clean_base']]:
            if parts.get('target_extension') and file_info['extension'] == parts['target_extension']:
                matches.append((file_info, 'clean_base_correct_ext', 0.9))
            else:
                matches.append((file_info, 'clean_base', 0.7))
    
    # Strategy 4: Match using expected original filename
    if parts.get('expected_original') and parts['expected_original'] in by_exact_name:
        for file_info in by_exact_name[parts['expected_original']]:
            matches.append((file_info, 'expected_original', 0.95))
    
    # Strategy 5: Fuzzy matching on original filename
    if parts['original_base']:
        orig_fuzzy = simplify_filename(parts['original_base'])
        if orig_fuzzy in by_fuzzy_name:
            for file_info in by_fuzzy_name[orig_fuzzy]:
                if parts.get('target_extension') and file_info['extension'] == parts['target_extension']:
                    matches.append((file_info, 'fuzzy_original_correct_ext', 0.85))
                else:
                    matches.append((file_info, 'fuzzy_original', 0.6))
    
    # Strategy 6: Fuzzy matching on clean base
    if parts.get('clean_base'):
        clean_fuzzy = simplify_filename(parts['clean_base'])
        if clean_fuzzy in by_fuzzy_name:
            for file_info in by_fuzzy_name[clean_fuzzy]:
                if parts.get('target_extension') and file_info['extension'] == parts['target_extension']:
                    matches.append((file_info, 'fuzzy_clean_correct_ext', 0.8))
                else:
                    matches.append((file_info, 'fuzzy_clean', 0.5))
    
    # Strategy 7: Super fuzzy matching
    if parts['original_base']:
        super_fuzzy = super_simplify_filename(parts['original_base'])
        if len(super_fuzzy) > 3 and super_fuzzy in by_super_fuzzy:  # Only if meaningful
            for file_info in by_super_fuzzy[super_fuzzy]:
                if parts.get('target_extension') and file_info['extension'] == parts['target_extension']:
                    matches.append((file_info, 'super_fuzzy_correct_ext', 0.7))
                else:
                    matches.append((file_info, 'super_fuzzy', 0.4))
    
    # Strategy 8: Partial string matching for very similar names
    search_terms = []
    if parts['original_base']:
        search_terms.append(parts['original_base'])
    if parts.get('clean_base'):
        search_terms.append(parts['clean_base'])
    
    for search_term in search_terms:
        if len(search_term) > 5:  # Only for meaningful terms
            for file_info in all_files:
                file_base = file_info['base_name']
                # Check if search term is substantially contained in filename
                if search_term.lower() in file_base.lower() or file_base.lower() in search_term.lower():
                    # Calculate similarity
                    similarity = calculate_similarity(search_term, file_base)
                    if similarity > 0.6:
                        if parts.get('target_extension') and file_info['extension'] == parts['target_extension']:
                            matches.append((file_info, f'partial_match_correct_ext', similarity * 0.9))
                        else:
                            matches.append((file_info, f'partial_match', similarity * 0.6))
    
    # Remove duplicates and sort by confidence
    unique_matches = {}
    for file_info, strategy, confidence in matches:
        key = file_info['full_path']
        if key not in unique_matches or unique_matches[key][2] < confidence:
            unique_matches[key] = (file_info, strategy, confidence)
    
    # Sort by confidence
    sorted_matches = sorted(unique_matches.values(), key=lambda x: x[2], reverse=True)
    
    return sorted_matches

def calculate_similarity(str1, str2):
    """Calculate similarity between two strings using Levenshtein distance"""
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    distance = levenshtein_distance(str1.lower(), str2.lower())
    max_len = max(len(str1), len(str2))
    return 1 - (distance / max_len) if max_len > 0 else 0

def analyze_missing_files():
    """Analyze files that weren't matched and try improved matching"""
    print("🚀 Advanced file matching analysis...")
    
    # Get database references and recovery files
    file_references = get_database_file_references()
    by_exact_name, by_base_name, by_fuzzy_name, by_super_fuzzy, all_files = build_recovery_file_index()
    
    print(f"\n🔍 Analyzing {len(file_references)} database references with improved matching...")
    
    matched = []
    still_missing = []
    
    for i, db_ref in enumerate(file_references):
        if i % 50 == 0:
            print(f"Progress: {i}/{len(file_references)}")
        
        matches = advanced_file_matching(
            db_ref, by_exact_name, by_base_name, by_fuzzy_name, by_super_fuzzy, all_files
        )
        
        if matches:
            # Take the best match
            best_match = matches[0]
            matched.append({
                'db_ref': db_ref,
                'file_info': best_match[0],
                'strategy': best_match[1],
                'confidence': best_match[2],
                'all_matches': matches[:3]  # Keep top 3 for analysis
            })
        else:
            still_missing.append(db_ref)
    
    print(f"\n🎯 Improved matching results:")
    print(f"  ✅ Found matches: {len(matched)}")
    print(f"  ❌ Still missing: {len(still_missing)}")
    print(f"  📈 Improvement: +{len(matched) - 433} additional matches")
    
    # Show sample improved matches
    print(f"\n📋 Sample matches by strategy:")
    strategy_samples = defaultdict(list)
    for match in matched:
        strategy_samples[match['strategy']].append(match)
    
    for strategy, matches in strategy_samples.items():
        if len(matches) > 0:
            sample = matches[0]
            print(f"\n  {strategy} (confidence: {sample['confidence']:.2f}):")
            print(f"    DB: {sample['db_ref']['storage_path']}")
            print(f"    Recovery: {sample['file_info']['relative_path']}")
            if sample['db_ref']['original_filename']:
                print(f"    Original: {sample['db_ref']['original_filename']}")
    
    # Analyze files that need conversion
    conversions_needed = []
    for match in matched:
        file_info = match['file_info']
        db_ref = match['db_ref']
        
        if db_ref['storage_path'].endswith('.docx.pdf') and file_info['extension'] == '.docx':
            conversions_needed.append(match)
        elif db_ref['storage_path'].endswith('.pptx.pdf') and file_info['extension'] == '.pptx':
            conversions_needed.append(match)
    
    print(f"\n📄 Files needing conversion: {len(conversions_needed)}")
    if conversions_needed:
        print("Sample conversions needed:")
        for conv in conversions_needed[:3]:
            print(f"  {conv['file_info']['filename']} -> PDF")
    
    # Analyze still missing files
    if still_missing:
        print(f"\n❌ Still missing files analysis:")
        missing_patterns = defaultdict(list)
        for miss in still_missing:
            storage_path = miss['storage_path']
            if storage_path.endswith('.docx.pdf'):
                missing_patterns['docx_pdf'].append(miss)
            elif storage_path.endswith('.pptx.pdf'):
                missing_patterns['pptx_pdf'].append(miss)
            elif storage_path.endswith('.pdf'):
                missing_patterns['pdf'].append(miss)
            else:
                missing_patterns['other'].append(miss)
        
        for pattern, files in missing_patterns.items():
            print(f"  {pattern}: {len(files)} files")
            if files:
                print(f"    Sample: {files[0]['storage_path']}")
    
    return matched, still_missing

if __name__ == "__main__":
    print("🚀 Improved File Matching Analysis")
    print("=" * 50)
    
    try:
        matched, still_missing = analyze_missing_files()
        
        print(f"\n🎯 Final Summary:")
        print(f"  Total database references: 691")
        print(f"  Previous matches: 433")
        print(f"  New matches found: {len(matched)}")
        print(f"  Still missing: {len(still_missing)}")
        print(f"  Overall success rate: {len(matched)/691*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc() 