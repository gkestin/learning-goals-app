#!/usr/bin/env python3

import os
import sys
import re
from collections import defaultdict

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def cornell_specific_matching():
    """Cornell-specific file matching with pattern adaptations"""
    from improved_matching_analysis import (
        get_database_file_references, build_recovery_file_index, 
        advanced_file_matching, calculate_similarity
    )
    
    file_references = get_database_file_references()
    by_exact_name, by_base_name, by_fuzzy_name, by_super_fuzzy, all_files = build_recovery_file_index()
    
    # Filter for Cornell files that are currently missing
    cornell_missing = []
    for db_ref in file_references:
        if 'Cornell' in db_ref['creator']:
            matches = advanced_file_matching(
                db_ref, by_exact_name, by_base_name, by_fuzzy_name, by_super_fuzzy, all_files
            )
            if not matches:
                cornell_missing.append(db_ref)
    
    print(f"🎯 Cornell-specific matching for {len(cornell_missing)} missing files")
    
    # Build Cornell-specific indexes
    cornell_files = [f for f in all_files if 'cornell' in f['relative_path'].lower()]
    print(f"📊 Found {len(cornell_files)} files in Cornell directories")
    
    new_matches = []
    
    for db_ref in cornell_missing:
        original_filename = db_ref['original_filename']
        storage_path = db_ref['storage_path']
        
        if not original_filename:
            continue
        
        print(f"\n🔍 Searching for: {original_filename}")
        
        # Extract key components
        original_base = os.path.splitext(original_filename)[0]
        
        # Cornell-specific transformations
        candidates = find_cornell_candidates(original_base, cornell_files)
        
        if candidates:
            best_candidate = candidates[0]
            print(f"  ✅ Found: {best_candidate[0]['relative_path']} (confidence: {best_candidate[1]:.2f})")
            new_matches.append({
                'db_ref': db_ref,
                'file_info': best_candidate[0],
                'strategy': 'cornell_specific',
                'confidence': best_candidate[1]
            })
        else:
            print(f"  ❌ No Cornell match found")
    
    return new_matches

def find_cornell_candidates(search_term, cornell_files):
    """Find Cornell file candidates using specific patterns"""
    candidates = []
    
    # Cornell-specific transformations
    transformations = [
        # Worksheet -> Slides transformation
        (r'(.+)-worksheet', r'\1-slides'),
        (r'(.+)_worksheet', r'\1_slides'),
        (r'(.+)-class(\d+)_worksheet', r'\1-class\2-slides'),
        (r'(.+)-class(\d+)-worksheet', r'\1-class\2-slides'),
        
        # ReadingGuides transformation
        (r'ReadingGuides-(.+)', r'\1'),
        
        # Homework transformations
        (r'HW(\d+) - (.+)-hw(\d+)', r'hw\3'),
        (r'(.+)-hw(\d+)', r'hw\2'),
        
        # Class transformations
        (r'Class (\d+) - (.+)-class(\d+)_worksheet', r'Class \1 - \2-class\3-slides'),
        (r'Class (\d+) - (.+)-class(\d+)-worksheet', r'Class \1 - \2-class\3-slides'),
        
        # Prelim transformations
        (r'Prelim (\d+)-prelim(\d+)', r'prelim\2'),
        
        # Extra problems
        (r'ExtraProblems&Plots-(.+)', r'\1'),
    ]
    
    # Apply transformations
    search_variants = [search_term]
    for pattern, replacement in transformations:
        if re.search(pattern, search_term, re.IGNORECASE):
            variant = re.sub(pattern, replacement, search_term, flags=re.IGNORECASE)
            search_variants.append(variant)
    
    # Search for each variant
    for variant in search_variants:
        for file_info in cornell_files:
            file_base = file_info['base_name']
            
            # Exact match
            if variant.lower() == file_base.lower():
                candidates.append((file_info, 1.0, 'exact_transformed'))
            
            # Partial match
            elif variant.lower() in file_base.lower():
                similarity = len(variant) / len(file_base) if len(file_base) > 0 else 0
                candidates.append((file_info, similarity * 0.9, 'contains_transformed'))
            
            elif file_base.lower() in variant.lower():
                similarity = len(file_base) / len(variant) if len(variant) > 0 else 0
                candidates.append((file_info, similarity * 0.8, 'contained_in_transformed'))
    
    # Also try word-based matching
    search_words = set(re.findall(r'\w+', search_term.lower()))
    for file_info in cornell_files:
        file_words = set(re.findall(r'\w+', file_info['base_name'].lower()))
        
        if search_words and file_words:
            common_words = search_words & file_words
            if len(common_words) > 0:
                similarity = len(common_words) / max(len(search_words), len(file_words))
                if similarity > 0.4:  # Only if substantial overlap
                    candidates.append((file_info, similarity * 0.7, 'word_overlap'))
    
    # Remove duplicates and sort by confidence
    unique_candidates = {}
    for file_info, confidence, method in candidates:
        key = file_info['full_path']
        if key not in unique_candidates or unique_candidates[key][1] < confidence:
            unique_candidates[key] = (file_info, confidence, method)
    
    # Sort by confidence
    sorted_candidates = sorted(unique_candidates.values(), key=lambda x: x[1], reverse=True)
    
    return sorted_candidates

def test_cornell_patterns():
    """Test Cornell pattern matching on known examples"""
    print("🧪 Testing Cornell pattern transformations:")
    
    test_cases = [
        "Class 26 - Review-class26_worksheet.pdf",
        "ReadingGuides-ReadingGuide_4-2.pdf", 
        "Prelim 2-prelim2.pdf",
        "Class 17 - 2.7-class17_worksheet.pdf",
        "HW10 - 3.1, 3.2-hw10-solutions.pdf",
        "HW8-hw8.pdf",
        "Class 14 - 2.4-class14-worksheet.pdf",
        "HW5 - 1.6, 1.7-hw5.pdf",
        "ExtraProblems&Plots-Questions_4-1.pdf"
    ]
    
    from improved_matching_analysis import build_recovery_file_index
    by_exact_name, by_base_name, by_fuzzy_name, by_super_fuzzy, all_files = build_recovery_file_index()
    cornell_files = [f for f in all_files if 'cornell' in f['relative_path'].lower()]
    
    for test_case in test_cases:
        print(f"\n🔍 Testing: {test_case}")
        base_name = os.path.splitext(test_case)[0]
        candidates = find_cornell_candidates(base_name, cornell_files)
        
        if candidates:
            best = candidates[0]
            print(f"  ✅ Best match: {best[0]['filename']} (confidence: {best[1]:.2f})")
            print(f"     Path: {best[0]['relative_path']}")
        else:
            print(f"  ❌ No matches found")

if __name__ == "__main__":
    print("🚀 Cornell-Specific File Matching")
    print("=" * 50)
    
    try:
        # First test the patterns
        test_cornell_patterns()
        
        print(f"\n" + "=" * 50)
        
        # Then run the actual matching
        new_matches = cornell_specific_matching()
        
        print(f"\n🎯 Cornell-specific matching results:")
        print(f"  ✅ Additional matches found: {len(new_matches)}")
        
        if new_matches:
            print(f"\n📋 Sample new matches:")
            for match in new_matches[:5]:
                print(f"  DB: {match['db_ref']['original_filename']}")
                print(f"  Recovery: {match['file_info']['relative_path']}")
                print(f"  Confidence: {match['confidence']:.2f}")
                print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc() 