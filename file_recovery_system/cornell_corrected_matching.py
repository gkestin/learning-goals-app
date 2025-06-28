#!/usr/bin/env python3

import os
import sys
import re
from collections import defaultdict

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def cornell_corrected_matching():
    """Cornell-specific file matching WITHOUT worksheet->slides transformations"""
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
    
    print(f"🎯 Cornell-corrected matching for {len(cornell_missing)} missing files")
    print("(Excluding worksheet→slides transformations)")
    
    # Build Cornell-specific indexes
    cornell_files = [f for f in all_files if 'cornell' in f['relative_path'].lower()]
    print(f"📊 Found {len(cornell_files)} files in Cornell directories")
    
    new_matches = []
    still_missing = []
    
    for db_ref in cornell_missing:
        original_filename = db_ref['original_filename']
        storage_path = db_ref['storage_path']
        
        if not original_filename:
            still_missing.append(db_ref)
            continue
        
        print(f"\n🔍 Searching for: {original_filename}")
        
        # Extract key components
        original_base = os.path.splitext(original_filename)[0]
        
        # Cornell-specific transformations (EXCLUDING worksheet->slides)
        candidates = find_cornell_candidates_corrected(original_base, cornell_files)
        
        if candidates:
            best_candidate = candidates[0]
            print(f"  ✅ Found: {best_candidate[0]['relative_path']} (confidence: {best_candidate[1]:.2f})")
            new_matches.append({
                'db_ref': db_ref,
                'file_info': best_candidate[0],
                'strategy': 'cornell_corrected',
                'confidence': best_candidate[1]
            })
        else:
            print(f"  ❌ No Cornell match found")
            still_missing.append(db_ref)
    
    return new_matches, still_missing

def find_cornell_candidates_corrected(search_term, cornell_files):
    """Find Cornell file candidates using corrected patterns (no worksheet->slides)"""
    candidates = []
    
    # Cornell-specific transformations (CORRECTED - no worksheet->slides)
    transformations = [
        # ReadingGuides transformation (this is legitimate)
        (r'ReadingGuides-(.+)', r'\1'),
        
        # Homework transformations (these are legitimate path/naming differences)
        (r'HW(\d+) - (.+)-hw(\d+)', r'hw\3'),
        (r'(.+)-hw(\d+)', r'hw\2'),
        
        # Prelim transformations (these are legitimate path differences)
        (r'Prelim (\d+)-prelim(\d+)', r'prelim\2'),
        
        # Extra problems (legitimate path differences)
        (r'ExtraProblems&Plots-(.+)', r'\1'),
        
        # Admin/Other prefix removal (legitimate)
        (r'Admin-(.+)', r'\1'),
        (r'Other Handouts-(.+)', r'\1'),
        (r'Miscellaneous-(.+)', r'\1'),
        (r'Practice-(.+)', r'\1'),
        (r'UpdatedRecitationWorksheets-(.+)', r'\1'),
        
        # HW with reading guide reference (legitimate)
        (r'HW\d+ - .+-(.+)', r'\1'),
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
    
    # Also try word-based matching (but be more conservative)
    search_words = set(re.findall(r'\w+', search_term.lower()))
    # Remove common words that might cause false matches
    common_words = {'class', 'week', 'hw', 'prelim', 'review', 'worksheet', 'slides', 'solutions'}
    search_words = search_words - common_words
    
    if len(search_words) >= 2:  # Only if we have at least 2 meaningful words
        for file_info in cornell_files:
            file_words = set(re.findall(r'\w+', file_info['base_name'].lower()))
            file_words = file_words - common_words
            
            if search_words and file_words:
                common_words_found = search_words & file_words
                if len(common_words_found) >= 2:  # Need at least 2 word matches
                    similarity = len(common_words_found) / max(len(search_words), len(file_words))
                    if similarity > 0.5:  # Higher threshold
                        candidates.append((file_info, similarity * 0.6, 'word_overlap_conservative'))
    
    # Remove duplicates and sort by confidence
    unique_candidates = {}
    for file_info, confidence, method in candidates:
        key = file_info['full_path']
        if key not in unique_candidates or unique_candidates[key][1] < confidence:
            unique_candidates[key] = (file_info, confidence, method)
    
    # Sort by confidence
    sorted_candidates = sorted(unique_candidates.values(), key=lambda x: x[1], reverse=True)
    
    return sorted_candidates

def analyze_worksheet_vs_slides_issue(cornell_missing):
    """Analyze how many missing files are worksheet vs slides issues"""
    worksheet_files = []
    other_files = []
    
    for db_ref in cornell_missing:
        original = db_ref['original_filename'] or ''
        if 'worksheet' in original.lower():
            worksheet_files.append(db_ref)
        else:
            other_files.append(db_ref)
    
    print(f"\n📊 Analysis of Cornell missing files:")
    print(f"  Worksheet files (potentially missing content): {len(worksheet_files)}")
    print(f"  Other files (likely naming/path issues): {len(other_files)}")
    
    if worksheet_files:
        print(f"\n📝 Sample worksheet files:")
        for wf in worksheet_files[:5]:
            print(f"    - {wf['original_filename']}")
    
    if other_files:
        print(f"\n📁 Sample other files:")
        for of in other_files[:5]:
            print(f"    - {of['original_filename']}")
    
    return worksheet_files, other_files

def test_corrected_patterns():
    """Test corrected Cornell pattern matching"""
    print("🧪 Testing CORRECTED Cornell pattern transformations:")
    
    # Separate legitimate transformations from worksheet->slides issues
    legitimate_cases = [
        "ReadingGuides-ReadingGuide_4-2.pdf",  # Path prefix issue
        "Prelim 2-prelim2.pdf",                # Path prefix issue  
        "HW10 - 3.1, 3.2-hw10-solutions.pdf",  # Path/naming issue
        "HW8-hw8.pdf",                         # Path/naming issue
        "ExtraProblems&Plots-Questions_4-1.pdf", # Path prefix issue
        "Admin-syllabus.pdf",                   # Path prefix issue
        "Practice-24prelim_1.pdf"              # Path prefix issue
    ]
    
    worksheet_cases = [
        "Class 26 - Review-class26_worksheet.pdf",    # Different content type
        "Class 17 - 2.7-class17_worksheet.pdf",       # Different content type
        "Class 14 - 2.4-class14-worksheet.pdf"        # Different content type
    ]
    
    from improved_matching_analysis import build_recovery_file_index
    by_exact_name, by_base_name, by_fuzzy_name, by_super_fuzzy, all_files = build_recovery_file_index()
    cornell_files = [f for f in all_files if 'cornell' in f['relative_path'].lower()]
    
    print(f"\n✅ Testing legitimate transformations:")
    for test_case in legitimate_cases:
        print(f"\n🔍 Testing: {test_case}")
        base_name = os.path.splitext(test_case)[0]
        candidates = find_cornell_candidates_corrected(base_name, cornell_files)
        
        if candidates:
            best = candidates[0]
            print(f"  ✅ Best match: {best[0]['filename']} (confidence: {best[1]:.2f})")
        else:
            print(f"  ❌ No matches found")
    
    print(f"\n❌ Testing worksheet cases (should NOT match slides):")
    for test_case in worksheet_cases:
        print(f"\n🔍 Testing: {test_case}")
        base_name = os.path.splitext(test_case)[0]
        candidates = find_cornell_candidates_corrected(base_name, cornell_files)
        
        if candidates:
            best = candidates[0]
            if 'slides' in best[0]['filename'].lower():
                print(f"  ⚠️  Found slides (should be avoided): {best[0]['filename']}")
            else:
                print(f"  ✅ Found non-slides: {best[0]['filename']} (confidence: {best[1]:.2f})")
        else:
            print(f"  ✅ Correctly no match found (worksheet content missing)")

if __name__ == "__main__":
    print("🚀 Cornell-Corrected File Matching")
    print("=" * 50)
    
    try:
        # First test the corrected patterns
        test_corrected_patterns()
        
        print(f"\n" + "=" * 50)
        
        # Get the original missing files for analysis
        from improved_matching_analysis import (
            get_database_file_references, build_recovery_file_index, 
            advanced_file_matching
        )
        
        file_references = get_database_file_references()
        by_exact_name, by_base_name, by_fuzzy_name, by_super_fuzzy, all_files = build_recovery_file_index()
        
        cornell_missing = []
        for db_ref in file_references:
            if 'Cornell' in db_ref['creator']:
                matches = advanced_file_matching(
                    db_ref, by_exact_name, by_base_name, by_fuzzy_name, by_super_fuzzy, all_files
                )
                if not matches:
                    cornell_missing.append(db_ref)
        
        # Analyze worksheet vs other issues
        worksheet_files, other_files = analyze_worksheet_vs_slides_issue(cornell_missing)
        
        # Run the corrected matching
        new_matches, still_missing = cornell_corrected_matching()
        
        print(f"\n🎯 Cornell-corrected matching results:")
        print(f"  ✅ Legitimate matches found: {len(new_matches)}")
        print(f"  ❌ Still missing (likely worksheet content): {len(still_missing)}")
        
        print(f"\n📊 Updated totals:")
        print(f"  Previous improved matches: 597")
        print(f"  Cornell legitimate matches: +{len(new_matches)}")
        print(f"  New total matches: {597 + len(new_matches)} out of 691")
        print(f"  Success rate: {(597 + len(new_matches))/691*100:.1f}%")
        print(f"  Remaining missing: {691 - 597 - len(new_matches)} files")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc() 