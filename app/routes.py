import os
import json
import time
import uuid
import csv
import io
from flask import Blueprint, render_template, request, redirect, url_for, jsonify, current_app, flash, session, Response, stream_with_context
from app.models import Document
from app.pdf_utils import allowed_file, save_pdf, extract_text_from_pdf
from app.openai_service import extract_learning_goals, DEFAULT_SYSTEM_MESSAGE
from app.firebase_service import upload_pdf_to_storage, save_document, get_document, search_documents, get_learning_goal_suggestions, delete_document, move_storage_file, smart_resolve_storage_path
from datetime import datetime
import datetime as dt

main = Blueprint('main', __name__)

# Temporary storage for embeddings data (in production, use Redis or similar)
embeddings_storage = {}

@main.route('/', methods=['GET'])
def index():
    """Render the main upload page"""
    return render_template('index.html', default_instructions=DEFAULT_SYSTEM_MESSAGE)

@main.route('/api/generate-upload-url', methods=['POST'])
def generate_upload_url():
    """Generate a signed URL for direct upload to Cloud Storage"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        content_type = data.get('contentType', 'application/pdf')
        file_size = data.get('fileSize', 0)  # Optional file size for validation
        
        if not filename or not allowed_file(filename, current_app.config['ALLOWED_EXTENSIONS']):
            return jsonify({'success': False, 'message': 'Invalid file type'}), 400
        
        # Check file size if provided (100MB limit)
        MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
        if file_size > MAX_FILE_SIZE:
            return jsonify({
                'success': False, 
                'message': f'File size ({file_size / (1024*1024):.1f}MB) exceeds maximum allowed size (100MB)'
            }), 413
        
        # Import here to avoid circular imports
        from app.firebase_service import generate_signed_upload_url
        
        # Generate signed URL for upload
        result = generate_signed_upload_url(filename, content_type)
        
        return jsonify({
            'success': True,
            'uploadUrl': result['upload_url'],
            'downloadUrl': result['download_url'],
            'storagePath': result['storage_path']
        })
        
    except Exception as e:
        print(f"Error generating upload URL: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@main.route('/api/upload-file', methods=['POST'])
def upload_file_to_storage():
    """Handle file upload via server to avoid CORS issues"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file provided'}), 400
        
        file = request.files['file']
        storage_path = request.form.get('storagePath')
        
        if not file or not storage_path:
            return jsonify({'success': False, 'message': 'File and storage path required'}), 400
        
        if not allowed_file(file.filename, current_app.config['ALLOWED_EXTENSIONS']):
            return jsonify({'success': False, 'message': 'Invalid file type'}), 400
        
        # Import here to avoid circular imports
        from app.firebase_service import bucket, using_mock
        
        if using_mock:
            print(f"⚠️ MOCK STORAGE: Pretending to upload {file.filename} to {storage_path}")
            return jsonify({
                'success': True,
                'storagePath': storage_path,
                'message': 'Mock upload successful'
            })
        
        print(f"✅ REAL STORAGE: Uploading {file.filename} to {storage_path}")
        
        # Upload file directly to storage
        blob = bucket.blob(storage_path)
        blob.upload_from_file(file.stream, content_type=file.content_type or 'application/pdf')
        
        print(f"Successfully uploaded {file.filename} to {storage_path}")
        
        return jsonify({
            'success': True,
            'storagePath': storage_path,
            'message': 'Upload successful'
        })
        
    except Exception as e:
        print(f"Error uploading file: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@main.route('/upload', methods=['POST'])
def upload_file():
    """Handle multiple PDF uploads and processing - DEPRECATED: Use client-side processing instead"""
    # This route is deprecated in favor of client-side PDF processing
    # Keeping for backward compatibility but redirecting to new flow
    flash('Please use the new client-side processing flow')
    return redirect(url_for('main.index'))

@main.route('/api/process-uploaded-files', methods=['POST'])
def process_uploaded_files():
    """Process files that were uploaded directly to Cloud Storage - DEPRECATED"""
    # This route is deprecated in favor of client-side processing
    return jsonify({
        'success': False, 
        'message': 'This endpoint is deprecated. Use client-side processing instead.'
    }), 410

@main.route('/edit', methods=['GET'])
def edit_learning_goals():
    """Render the page for editing learning goals for multiple documents"""
    # Check if we have session data from the new client-side processing flow
    if 'processed_files' not in session:
        # Try to create session data from browser if available
        # This will be populated by JavaScript on the client side
        flash('No file data found. Please select files and extract learning goals first.')
        return redirect(url_for('main.index'))
    
    processed_files = session.get('processed_files', [])
    
    # Add file size information for display if not already present
    for file_data in processed_files:
        if 'file_size' not in file_data:
            file_data['file_size'] = 0  # Default if not available
    
    return render_template('edit.html', 
                           processed_files=processed_files,
                           using_session_data=True)

@main.route('/save', methods=['POST'])
def save_document_data():
    """Save multiple document data to Firebase"""
    # Get form data for global metadata
    global_creator = request.form.get('global_creator', '')
    global_course_name = request.form.get('global_course_name', '')
    global_institution = request.form.get('global_institution', '')
    global_doc_type = request.form.get('global_doc_type', '')
    global_notes = request.form.get('global_notes', '')
    
    # Get uploaded files data (new flow)
    uploaded_files_json = request.form.get('uploaded_files')
    if uploaded_files_json:
        try:
            uploaded_files = json.loads(uploaded_files_json)
        except json.JSONDecodeError:
            return jsonify({'success': False, 'message': 'Invalid upload data'}), 400
    else:
        uploaded_files = []
    
    # Get processed files from session (contains learning goals and metadata)
    processed_files = session.get('processed_files', [])
    
    if not processed_files:
        return jsonify({'success': False, 'message': 'No file data found in session'}), 400
    
    if len(uploaded_files) != len(processed_files):
        return jsonify({
            'success': False, 
            'message': f'Mismatch between uploaded files ({len(uploaded_files)}) and processed files ({len(processed_files)})'
        }), 400
    
    results = []
    success_count = 0
    
    for index, (upload_result, file_data) in enumerate(zip(uploaded_files, processed_files)):
        try:
            # Get specific document metadata or use global values
            doc_id = request.form.get(f'doc_id_{index}', str(index))
            name = request.form.get(f'document_name_{index}', file_data['original_filename'])
            creator = request.form.get(f'creator_{index}', global_creator)
            course_name = request.form.get(f'course_name_{index}', global_course_name)
            institution = request.form.get(f'institution_{index}', global_institution)
            doc_type = request.form.get(f'doc_type_{index}', global_doc_type)
            notes = request.form.get(f'notes_{index}', global_notes)
            
            # Get learning goals for this document
            learning_goals = request.form.getlist(f'learning_goals_{index}[]')
            
            original_filename = file_data['original_filename']
            storage_path = upload_result['storagePath']
            
            # Move from temp location to final location if needed
            # Fix filename duplication by using name only if it's different from original_filename
            if name and name != original_filename:
                final_storage_path = f"pdfs/{creator}_{name}_{original_filename}"
            else:
                final_storage_path = f"pdfs/{creator}_{original_filename}"
            
            if storage_path != final_storage_path:
                # Import here to avoid circular imports
                from app.firebase_service import move_storage_file
                move_result = move_storage_file(storage_path, final_storage_path)
                final_storage_path = move_result['storage_path']
                public_url = move_result.get('public_url')
            else:
                # File is already in the right location, generate public URL
                from app.firebase_service import bucket, using_mock
                if using_mock:
                    public_url = f"https://mock-storage.example.com/{final_storage_path}"
                else:
                    bucket_name = bucket.name
                    encoded_path = final_storage_path.replace('/', '%2F')
                    public_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o/{encoded_path}?alt=media"
            
            # Create document object
            doc = Document(
                name=name or original_filename,
                original_filename=original_filename,
                creator=creator,
                course_name=course_name,
                institution=institution,
                doc_type=doc_type,
                notes=notes,
                learning_goals=learning_goals,
                storage_path=final_storage_path,
                public_url=public_url,
                lo_extraction_prompt=file_data.get('lo_extraction_prompt', '')
            )
            
            # Save to Firestore
            doc_id = save_document(doc)
            
            success_count += 1
            results.append({
                'success': True,
                'document_id': doc_id,
                'name': doc.name,
                'storage_path': doc.storage_path
            })
            
        except Exception as e:
            print(f"Error saving document {index}: {e}")
            results.append({
                'success': False,
                'error': str(e),
                'name': file_data.get('original_filename', f'Document {index}')
            })
    
    # Clear session data after successful processing
    if success_count > 0:
        session.pop('processed_files', None)
    
    return jsonify({
        'success': success_count > 0,
        'success_count': success_count,
        'total': len(processed_files),
        'results': results,
        'message': f'Successfully saved {success_count} of {len(processed_files)} documents'
    })

@main.route('/search')
def search_page():
    """Render the search page"""
    return render_template('search.html')

@main.route('/api/search')
def api_search():
    """API endpoint for searching documents"""
    query = request.args.get('q', '')
    # Set a high limit to fetch a large number of documents
    # This will be controlled by the firebase_service.py default limit
    if not query:
        print("Searching for all documents in Firestore...")
        results = search_documents() 
    else:
        print(f"Searching for documents matching: {query}")
        search_terms = query.split()
        results = search_documents(search_terms)
    
    # Convert to JSON-serializable format
    docs = []
    for doc in results:
        docs.append({
            'id': doc.id,
            'name': doc.name,
            'creator': doc.creator,
            'course_name': doc.course_name,
            'institution': doc.institution,
            'doc_type': doc.doc_type,
            'notes': doc.notes,
            'learning_goals': doc.learning_goals,
            'storage_path': doc.storage_path
        })
    
    print(f"Found {len(docs)} documents in Firestore")
    return jsonify(docs)

@main.route('/api/suggest')
def api_suggest():
    """API endpoint for autocomplete suggestions"""
    query = request.args.get('q', '')
    if len(query) < 3:
        return jsonify([])
        
    print(f"Getting learning goal suggestions for: {query}")
    suggestions = get_learning_goal_suggestions(query)
    print(f"Found {len(suggestions)} suggestions")
    return jsonify(suggestions)

@main.route('/view/<document_id>')
def view_document(document_id):
    """View a single document"""
    print(f"Retrieving document {document_id} from Firestore")
    doc = get_document(document_id)
    if not doc:
        flash('Document not found in Firestore')
        return redirect(url_for('main.search_page'))
    
    # Generate Firebase Storage URL for debugging
    if doc.storage_path:
        bucket_name = current_app.config.get('FIREBASE_STORAGE_BUCKET')
        encoded_path = doc.storage_path.replace('/', '%2F')
        firebase_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o/{encoded_path}?alt=media"
        print(f"DEBUG - Firebase Storage URL: {firebase_url}")
        
    return render_template('view.html', document=doc)

@main.route('/debug_url/<document_id>')
def debug_storage_url(document_id):
    """Debug endpoint to get the Firebase Storage URL for a document"""
    doc = get_document(document_id)
    if not doc:
        return jsonify({'error': 'Document not found'})
        
    # Generate the URL
    bucket_name = current_app.config.get('FIREBASE_STORAGE_BUCKET')
    
    # Original format for traditional Cloud Storage
    traditional_url = f"https://storage.googleapis.com/{bucket_name}/{doc.storage_path}"
    
    # Firebase Storage format
    encoded_path = doc.storage_path.replace('/', '%2F')
    firebase_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o/{encoded_path}?alt=media"
    
    return jsonify({
        'document_id': doc.id,
        'document_name': doc.name,
        'storage_path': doc.storage_path,
        'bucket_name': bucket_name,
        'traditional_url': traditional_url,
        'firebase_url': firebase_url
    })

@main.route('/debug_firebase')
def debug_firebase():
    """Debug page for Firebase Storage issues"""
    import firebase_admin
    from firebase_admin import storage
    
    # Get current configuration
    bucket_name = current_app.config.get('FIREBASE_STORAGE_BUCKET')
    cred_path = current_app.config.get('GOOGLE_APPLICATION_CREDENTIALS')
    
    # Get all documents to test
    docs = search_documents(limit=5)
    
    result = {
        'bucket_name': bucket_name,
        'credentials_path': cred_path,
        'credentials_exist': os.path.exists(cred_path) if cred_path else False,
        'firebase_initialized': bool(firebase_admin._apps),
        'documents': []
    }
    
    # Get bucket info
    bucket = storage.bucket()
    result['bucket_info'] = {
        'name': bucket.name,
        'path': bucket.path,
        'user_project': bucket.user_project
    }
    
    # Test retrieving and signing each document
    for doc in docs:
        if not doc.storage_path:
            continue
            
        doc_info = {
            'id': doc.id,
            'name': doc.name,
            'storage_path': doc.storage_path
        }
        
        # Test direct access URL
        encoded_path = doc.storage_path.replace('/', '%2F')
        doc_info['direct_url'] = f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o/{encoded_path}?alt=media"
        
        # Try to get signed URL
        try:
            blob = bucket.blob(doc.storage_path)
            import datetime as dt
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=dt.timedelta(minutes=30),
                method="GET"
            )
            doc_info['signed_url'] = signed_url
            doc_info['signed_url_working'] = True
        except Exception as e:
            doc_info['signed_url_error'] = str(e)
            doc_info['signed_url_working'] = False
            
        result['documents'].append(doc_info)
    
    return jsonify(result)

@main.route('/api/delete/<document_id>', methods=['POST'])
def api_delete_document(document_id):
    """API endpoint to delete a document"""
    print(f"Deleting document with ID: {document_id}")
    
    try:
        # Attempt to delete the document
        success = delete_document(document_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Document deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to delete document'
            }), 404
    except Exception as e:
        print(f"Error in delete API: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@main.route('/api/delete-user-documents/<creator>', methods=['POST'])
def api_delete_user_documents(creator):
    """API endpoint to delete all documents by a specific user/creator"""
    print(f"Deleting all documents for creator: {creator}")
    
    try:
        # First, get all documents by this creator
        all_documents = search_documents(limit=1000)  # Get all documents
        user_documents = [doc for doc in all_documents if doc.creator == creator]
        
        if not user_documents:
            return jsonify({
                'success': False,
                'message': f'No documents found for creator: {creator}'
            }), 404
        
        # Delete each document
        deleted_count = 0
        failed_deletions = []
        
        for doc in user_documents:
            try:
                success = delete_document(doc.id)
                if success:
                    deleted_count += 1
                else:
                    failed_deletions.append(doc.name)
            except Exception as e:
                print(f"Error deleting document {doc.id}: {e}")
                failed_deletions.append(doc.name)
        
        if deleted_count == len(user_documents):
            return jsonify({
                'success': True,
                'message': f'Successfully deleted all {deleted_count} documents for {creator}',
                'deleted_count': deleted_count
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Deleted {deleted_count} of {len(user_documents)} documents. Failed to delete: {", ".join(failed_deletions)}',
                'deleted_count': deleted_count,
                'failed_count': len(failed_deletions)
            }), 500
            
    except Exception as e:
        print(f"Error in batch delete API: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@main.route('/cluster')
def cluster_page():
    """Render the clustering sandbox page"""
    return render_template('cluster.html')

@main.route('/api/goals-overview')
def api_goals_overview():
    """API endpoint to get overview of available learning goals"""
    try:
        # Get course filter from query parameters
        course_filter = request.args.get('course')
        
        # Get all learning goals from Firebase
        all_documents = search_documents(limit=1000)  # Get all documents
        
        # Extract all learning goals with course filtering
        all_goals = []
        documents_count = 0
        unique_creators = set()
        unique_courses = set()
        
        for doc in all_documents:
            # Apply course filter if specified
            if course_filter and doc.course_name != course_filter:
                continue
                
            if doc.learning_goals:  # Only count documents with learning goals
                documents_count += 1
                all_goals.extend(doc.learning_goals)
                if doc.creator:
                    unique_creators.add(doc.creator)
                if doc.course_name:
                    unique_courses.add(doc.course_name)
        
        return jsonify({
            'success': True,
            'total_goals': len(all_goals),
            'total_documents': documents_count,
            'unique_creators': len(unique_creators),
            'unique_courses': len(unique_courses),
            'unique_goals': len(set(all_goals)),  # Count unique goals
            'avg_goals_per_doc': round(len(all_goals) / documents_count, 1) if documents_count > 0 else 0
        })
        
    except Exception as e:
        print(f"Error getting goals overview: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@main.route('/api/available-courses')
def api_available_courses():
    """API endpoint to get available courses for filtering"""
    try:
        # Get all documents from Firebase
        all_documents = search_documents(limit=1000)
        
        # Extract unique course names (excluding empty ones)
        courses = set()
        course_stats = {}
        
        for doc in all_documents:
            if doc.course_name and doc.course_name.strip() and doc.learning_goals:
                course_name = doc.course_name.strip()
                courses.add(course_name)
                
                # Track statistics per course
                if course_name not in course_stats:
                    course_stats[course_name] = {
                        'documents': 0,
                        'total_goals': 0,
                        'creators': set()
                    }
                
                course_stats[course_name]['documents'] += 1
                course_stats[course_name]['total_goals'] += len(doc.learning_goals)
                if doc.creator:
                    course_stats[course_name]['creators'].add(doc.creator)
        
        # Format course list with statistics
        formatted_courses = []
        for course in sorted(courses):
            formatted_courses.append({
                'name': course,
                'document_count': course_stats[course]['documents'],
                'goal_count': course_stats[course]['total_goals'],
                'creator_count': len(course_stats[course]['creators'])
            })
        
        return jsonify({
            'success': True,
            'courses': formatted_courses,
            'total_courses': len(formatted_courses)
        })
        
    except Exception as e:
        print(f"Error getting available courses: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@main.route('/api/cluster', methods=['POST'])
def api_cluster_goals():
    """API endpoint for clustering learning goals"""
    from app.clustering_service import LearningGoalsClusteringService
    
    # Get all learning goals from Firebase
    all_documents = search_documents(limit=1000)  # Get all documents
    
    # Extract all learning goals
    all_goals = []
    goal_sources = []  # Track which document each goal came from
    
    for doc in all_documents:
        for goal in doc.learning_goals:
            all_goals.append(goal)
            goal_sources.append({
                'document_name': doc.name,
                'document_id': doc.id,
                'creator': doc.creator,
                'course_name': doc.course_name
            })
    
    if len(all_goals) < 2:
        return jsonify({
            'success': False,
            'message': 'Need at least 2 learning goals to cluster'
        })
    
    # Get clustering parameters
    n_clusters = request.json.get('n_clusters', 5)
    
    # Initialize clustering service
    clustering_service = LearningGoalsClusteringService()
    
    try:
        # Generate embeddings
        print(f"Generating embeddings for {len(all_goals)} learning goals...")
        embeddings = clustering_service.generate_embeddings(all_goals)
        
        # Perform fast clustering
        n_clusters = min(n_clusters, len(all_goals))
        print(f"Performing optimized clustering with {n_clusters} clusters...")
        clustering_start = time.time()
        labels = clustering_service.cluster_fast(embeddings, n_clusters)
        
        # Calculate quality metrics
        metrics_start = time.time()
        silhouette_avg = clustering_service.calculate_cluster_quality_fast(embeddings, labels)
        inter_cluster_separation, intra_cluster_cohesion = clustering_service.calculate_cluster_separation_metrics_fast(embeddings, labels)
        metrics_elapsed = time.time() - metrics_start
        total_elapsed = time.time() - clustering_start
        
        print(f"⚡ Metrics calculation completed in {metrics_elapsed:.2f} seconds")
        print(f"🎯 Total clustering + metrics time: {total_elapsed:.2f} seconds")
        
        # Organize results
        clusters = {}
        for i, (goal, label, source) in enumerate(zip(all_goals, labels, goal_sources)):
            # Convert numpy int64 to regular Python int
            label = int(label) if label != -1 else f"outlier_{i}"
            
            if label not in clusters:
                clusters[label] = {
                    'goals': [],
                    'sources': [],
                    'size': 0
                }
            
            clusters[label]['goals'].append(goal)
            clusters[label]['sources'].append(source)
            clusters[label]['size'] += 1
        
        # Format response
        formatted_clusters = []
        for cluster_id, cluster_data in clusters.items():
            # Get representative goal (longest one)
            representative_goal = max(cluster_data['goals'], key=len)
            
            formatted_clusters.append({
                'id': int(cluster_id) if isinstance(cluster_id, (int, float)) else cluster_id,
                'size': cluster_data['size'],
                'goals': cluster_data['goals'],
                'sources': cluster_data['sources'],
                'representative_goal': representative_goal
            })
        
        # Sort clusters by size (largest first)
        formatted_clusters.sort(key=lambda x: x['size'], reverse=True)
        
        print(f"Clustering completed: {len(formatted_clusters)} clusters found")
        
        return jsonify({
            'success': True,
            'clusters': formatted_clusters,
            'total_goals': len(all_goals),
            'n_clusters': len(formatted_clusters),
            'silhouette_score': round(float(silhouette_avg), 3),
            'inter_cluster_separation': round(float(inter_cluster_separation), 3),
            'intra_cluster_cohesion': round(float(intra_cluster_cohesion), 3),
            'method_used': 'fast'
        })
        
    except Exception as e:
        print(f"Clustering error: {e}")
        return jsonify({
            'success': False,
            'message': f'Clustering failed: {str(e)}'
        })

@main.route('/api/cluster-stream', methods=['POST'])
def api_cluster_goals_stream():
    """API endpoint for clustering learning goals with real-time progress updates via SSE"""
    from flask import Response, stream_with_context
    from app.clustering_service import LearningGoalsClusteringService
    import json
    
    def generate_progress():
        try:
            # Get request data before entering generator
            request_data = request.get_json()
            n_clusters = request_data.get('n_clusters', 5)
            course_filter = request_data.get('course_filter', None)  # New course filter parameter
            
            # Send initial status
            yield f"data: {json.dumps({'status': 'starting', 'message': 'Initializing clustering process...'})}\n\n"
            
            # Get all learning goals from Firebase
            if course_filter:
                loading_message = f"Loading learning goals from course: {course_filter}..."
                yield f"data: {json.dumps({'status': 'loading', 'message': loading_message})}\n\n"
            else:
                yield f"data: {json.dumps({'status': 'loading', 'message': 'Loading learning goals from database...'})}\n\n"
            
            all_documents = search_documents(limit=1000)
            
            # Extract learning goals with course filtering
            all_goals = []
            goal_sources = []
            
            for doc in all_documents:
                # Apply course filter if specified
                if course_filter and doc.course_name != course_filter:
                    continue
                    
                for goal in doc.learning_goals:
                    all_goals.append(goal)
                    goal_sources.append({
                        'document_name': doc.name,
                        'document_id': doc.id,
                        'creator': doc.creator,
                        'course_name': doc.course_name
                    })
            
            if course_filter:
                loaded_message = f"Loaded {len(all_goals)} learning goals from course: {course_filter}"
                yield f"data: {json.dumps({'status': 'loaded', 'message': loaded_message, 'totalGoals': len(all_goals)})}\n\n"
            else:
                loaded_message = f"Loaded {len(all_goals)} learning goals from {len(all_documents)} documents"
                yield f"data: {json.dumps({'status': 'loaded', 'message': loaded_message, 'totalGoals': len(all_goals)})}\n\n"
            
            if len(all_goals) < 2:
                if course_filter:
                    error_message = f'Need at least 2 learning goals to cluster. Course "{course_filter}" only has {len(all_goals)} goals.'
                    yield f"data: {json.dumps({'status': 'error', 'message': error_message})}\n\n"
                else:
                    yield f"data: {json.dumps({'status': 'error', 'message': 'Need at least 2 learning goals to cluster'})}\n\n"
                return
            
            # Initialize clustering service
            clustering_service = LearningGoalsClusteringService()
            
            # Generate embeddings
            print(f"Generating embeddings for {len(all_goals)} learning goals...")
            embeddings = clustering_service.generate_embeddings(all_goals)
            
            # Store embeddings for download
            session_id = str(uuid.uuid4())
            embeddings_storage[session_id] = {
                'goals': all_goals,
                'embeddings': embeddings.tolist(),  # Convert numpy array to list for JSON serialization
                'timestamp': time.time()
            }
            
            yield f"data: {json.dumps({'status': 'embeddings_complete', 'message': 'Embeddings generated successfully', 'embeddings_session_id': session_id})}\n\n"
            
            # Perform clustering
            n_clusters = min(n_clusters, len(all_goals))
            clustering_message = f"Grouping goals into {n_clusters} clusters..."
            yield f"data: {json.dumps({'status': 'clustering', 'message': clustering_message, 'step': 'clustering', 'progress': 0})}\n\n"
            labels = clustering_service.cluster_fast(embeddings, n_clusters)
            yield f"data: {json.dumps({'status': 'clustering_complete', 'message': 'Clustering completed', 'step': 'clustering', 'progress': 100})}\n\n"
            
            # Calculate quality metrics
            yield f"data: {json.dumps({'status': 'metrics', 'message': 'Calculating quality metrics...', 'step': 'metrics', 'progress': 0})}\n\n"
            
            # Silhouette score
            yield f"data: {json.dumps({'status': 'metrics', 'message': 'Calculating silhouette score...', 'step': 'metrics', 'progress': 33})}\n\n"
            silhouette_avg = clustering_service.calculate_cluster_quality_fast(embeddings, labels)
            
            # Separation metrics
            yield f"data: {json.dumps({'status': 'metrics', 'message': 'Calculating cluster separation...', 'step': 'metrics', 'progress': 66})}\n\n"
            inter_cluster_separation, intra_cluster_cohesion = clustering_service.calculate_cluster_separation_metrics_fast(embeddings, labels)
            
            yield f"data: {json.dumps({'status': 'metrics_complete', 'message': 'Quality metrics calculated', 'step': 'metrics', 'progress': 100})}\n\n"
            
            # Organize results
            yield f"data: {json.dumps({'status': 'organizing', 'message': 'Organizing results...', 'step': 'organizing', 'progress': 0})}\n\n"
            
            clusters = {}
            for i, (goal, label, source) in enumerate(zip(all_goals, labels, goal_sources)):
                label = int(label) if label != -1 else f"outlier_{i}"
                
                if label not in clusters:
                    clusters[label] = {
                        'goals': [],
                        'sources': [],
                        'size': 0
                    }
                
                clusters[label]['goals'].append(goal)
                clusters[label]['sources'].append(source)
                clusters[label]['size'] += 1
                
                # Send progress updates periodically
                if i % 100 == 0:
                    progress = int((i / len(all_goals)) * 100)
                    organizing_message = f"Processing goal {i+1} of {len(all_goals)}..."
                    yield f"data: {json.dumps({'status': 'organizing', 'message': organizing_message, 'step': 'organizing', 'progress': progress})}\n\n"
            
            # Format final results
            formatted_clusters = []
            for cluster_id, cluster_data in clusters.items():
                representative_goal = max(cluster_data['goals'], key=len)
                formatted_clusters.append({
                    'id': int(cluster_id) if isinstance(cluster_id, (int, float)) else cluster_id,
                    'size': cluster_data['size'],
                    'goals': cluster_data['goals'],
                    'sources': cluster_data['sources'],
                    'representative_goal': representative_goal
                })
            
            formatted_clusters.sort(key=lambda x: x['size'], reverse=True)
            
            # Send final results
            final_results = {
                'status': 'complete',
                'message': 'Clustering completed successfully',
                'results': {
                    'success': True,
                    'clusters': formatted_clusters,
                    'total_goals': len(all_goals),
                    'n_clusters': len(formatted_clusters),
                    'silhouette_score': round(float(silhouette_avg), 3),
                    'inter_cluster_separation': round(float(inter_cluster_separation), 3),
                    'intra_cluster_cohesion': round(float(intra_cluster_cohesion), 3),
                    'method_used': 'fast'
                }
            }
            yield f"data: {json.dumps(final_results)}\n\n"
            
        except Exception as e:
            print(f"Clustering stream error: {e}")
            error_message = f"Clustering failed: {str(e)}"
            yield f"data: {json.dumps({'status': 'error', 'message': error_message})}\n\n"
    
    return Response(
        stream_with_context(generate_progress()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'  # Disable buffering for nginx
        }
    )

@main.route('/api/find-optimal-clusters', methods=['POST'])
def api_find_optimal_clusters():
    """API endpoint to find optimal cluster sizes"""
    from app.clustering_service import LearningGoalsClusteringService
    
    # Get all learning goals from Firebase
    all_documents = search_documents(limit=1000)
    
    # Extract all learning goals
    all_goals = []
    for doc in all_documents:
        all_goals.extend(doc.learning_goals)
    
    if len(all_goals) < 4:
        return jsonify({
            'success': False,
            'message': 'Need at least 4 learning goals for optimization analysis'
        })
    
    # Get optimization parameters
    request_data = request.get_json() or {}
    use_multires = request_data.get('use_multires', True)  # Default to multi-resolution for efficiency
    
    # Initialize clustering service
    clustering_service = LearningGoalsClusteringService()
    
    try:
        optimization_method = "multi-resolution" if use_multires else "exhaustive"
        print(f"Finding optimal cluster sizes for {len(all_goals)} learning goals using {optimization_method} search...")
        
        # Generate embeddings
        embeddings = clustering_service.generate_embeddings(all_goals)
        
        # Find optimal cluster sizes using true optimization
        results, best_composite_k, best_separation_k = clustering_service.find_optimal_cluster_sizes(
            embeddings, 
            max_clusters=min(len(all_goals) - 1, len(all_goals) // 2 + 50),  # Test broader range
            use_multires=use_multires,
            use_fast=True  # Use all performance optimizations
        )
        
        if results is None:
            return jsonify({
                'success': False,
                'message': 'Unable to perform optimization analysis'
            })
        
        print(f"True optimization completed: Best composite {best_composite_k}, Best separation {best_separation_k}")
        
        return jsonify({
            'success': True,
            'total_goals': len(all_goals),
            'best_composite_k': int(best_composite_k),
            'best_separation_k': int(best_separation_k),
            'best_cohesion_k': int(results['best_cohesion_k']),
            'best_silhouette_k': int(results['best_silhouette_k']),
            'max_composite_score': round(float(results['max_composite']), 3),
            'max_separation_score': round(float(results['max_separation']), 3),
            'max_cohesion_score': round(float(results['max_cohesion']), 3),
            'max_silhouette_score': round(float(results['max_silhouette']), 3),
            'analysis_data': {
                'cluster_sizes': [int(k) for k in results['cluster_sizes']],
                'silhouette_scores': [round(float(s), 3) for s in results['silhouette_scores']],
                'separation_scores': [round(float(s), 3) for s in results['separation_scores']],
                'cohesion_scores': [round(float(s), 3) for s in results['cohesion_scores']],
                'composite_scores': [round(float(s), 3) for s in results['composite_scores']]
            },
            'recommendation': {
                'primary': int(best_composite_k),
                'alternative': int(best_separation_k),
                'explanation': f"True optimization suggests {best_composite_k} clusters for best overall quality (composite score: {results['max_composite']:.3f}). Alternative: {best_separation_k} clusters for maximum cluster separation."
            }
        })
        
    except Exception as e:
        print(f"Optimization error: {e}")
        return jsonify({
            'success': False,
            'message': f'Optimization failed: {str(e)}'
        })

@main.route('/api/find-optimal-clusters-stream', methods=['POST'])
def api_find_optimal_clusters_stream():
    """API endpoint for finding optimal cluster sizes with real-time progress updates via SSE"""
    from flask import Response, stream_with_context
    from app.clustering_service import LearningGoalsClusteringService
    import json
    import threading
    import queue
    
    def generate_optimization_progress():
        try:
            # Get request data
            request_data = request.get_json() or {}
            course_filter = request_data.get('course_filter', None)  # New course filter parameter
            
            # Send initial status
            yield f"data: {json.dumps({'status': 'starting', 'message': 'Initializing optimization process...'})}\n\n"
            
            # Get all learning goals from Firebase
            if course_filter:
                loading_message = f"Loading learning goals from course: {course_filter}..."
                yield f"data: {json.dumps({'status': 'loading', 'message': loading_message})}\n\n"
            else:
                yield f"data: {json.dumps({'status': 'loading', 'message': 'Loading learning goals from database...'})}\n\n"
            
            all_documents = search_documents(limit=1000)
            
            # Extract learning goals with course filtering
            all_goals = []
            for doc in all_documents:
                # Apply course filter if specified
                if course_filter and doc.course_name != course_filter:
                    continue
                all_goals.extend(doc.learning_goals)
            
            if course_filter:
                loaded_message = f"Loaded {len(all_goals)} learning goals from course: {course_filter}"
                yield f"data: {json.dumps({'status': 'loaded', 'message': loaded_message, 'totalGoals': len(all_goals)})}\n\n"
            else:
                loaded_message = f"Loaded {len(all_goals)} learning goals from {len(all_documents)} documents"
                yield f"data: {json.dumps({'status': 'loaded', 'message': loaded_message, 'totalGoals': len(all_goals)})}\n\n"
            
            if len(all_goals) < 4:
                if course_filter:
                    error_message = f'Need at least 4 learning goals for optimization. Course "{course_filter}" only has {len(all_goals)} goals.'
                    yield f"data: {json.dumps({'status': 'error', 'message': error_message})}\n\n"
                else:
                    yield f"data: {json.dumps({'status': 'error', 'message': 'Need at least 4 learning goals for optimization'})}\n\n"
                return
            
            # Initialize clustering service
            clustering_service = LearningGoalsClusteringService()
            
            # Use a queue for progress updates from the optimization thread
            progress_queue = queue.Queue()
            optimization_result = {}
            
            def run_optimization():
                try:
                    # Stream progress callback
                    def stream_progress(message, completed, total, percentage):
                        progress_queue.put({
                            'type': 'progress',
                            'message': message,
                            'completed': completed,
                            'total': total,
                            'percentage': percentage
                        })
                    
                    # Check for stop signal (implement basic mechanism)
                    global optimization_should_stop
                    optimization_should_stop = False
                    
                    def check_stop_wrapper(original_func):
                        def wrapper(*args, **kwargs):
                            if optimization_should_stop:
                                raise InterruptedError("Optimization stopped by user")
                            return original_func(*args, **kwargs)
                        return wrapper
                    
                    # Generate embeddings first
                    progress_queue.put({'type': 'embeddings_start'})
                    embeddings = clustering_service.generate_embeddings(all_goals)
                    
                    # Store embeddings for download
                    session_id = str(uuid.uuid4())
                    embeddings_storage[session_id] = {
                        'goals': all_goals,
                        'embeddings': embeddings.tolist(),
                        'timestamp': time.time()
                    }
                    
                    progress_queue.put({'type': 'embeddings_complete', 'session_id': session_id})
                    
                    # Start optimization
                    progress_queue.put({'type': 'optimization_start'})
                    
                    results, best_composite_k, best_separation_k = clustering_service.find_optimal_cluster_sizes(
                        embeddings, 
                        progress_callback=stream_progress,
                        use_elbow_detection=True
                    )
                    
                    if results:
                        # Calculate additional metrics for the best results
                        embeddings_final = clustering_service.generate_embeddings(all_goals)
                        
                        # Get metrics for best composite result
                        labels_composite = clustering_service.cluster_fast(embeddings_final, best_composite_k)
                        silhouette_composite = clustering_service.calculate_cluster_quality_fast(embeddings_final, labels_composite)
                        separation_composite, cohesion_composite = clustering_service.calculate_cluster_separation_metrics_fast(embeddings_final, labels_composite)
                        
                        # Get metrics for best separation result  
                        labels_separation = clustering_service.cluster_fast(embeddings_final, best_separation_k)
                        silhouette_separation = clustering_service.calculate_cluster_quality_fast(embeddings_final, labels_separation)
                        separation_separation, cohesion_separation = clustering_service.calculate_cluster_separation_metrics_fast(embeddings_final, labels_separation)
                        
                        optimization_result['results'] = {
                            'success': True,
                            'total_goals': len(all_goals),
                            'best_composite_k': best_composite_k,
                            'best_separation_k': best_separation_k,
                            'best_cohesion_k': results.get('best_cohesion_k', best_composite_k),
                            'best_silhouette_k': results.get('best_silhouette_k', best_composite_k),
                            'max_composite_score': round(float(results['max_composite']), 3),
                            'max_separation_score': round(float(results['max_separation']), 3),
                            'max_cohesion_score': round(float(results['max_cohesion']), 3),
                            'max_silhouette_score': round(float(results['max_silhouette']), 3),
                            'recommendation': {
                                'explanation': f"For {len(all_goals)} learning goals, the algorithm recommends {best_composite_k} clusters for balanced quality, or {best_separation_k} clusters for maximum separation."
                            }
                        }
                    else:
                        optimization_result['results'] = {
                            'success': False,
                            'message': 'Optimization failed to find suitable cluster sizes'
                        }
                    
                    progress_queue.put({'type': 'complete'})
                    
                except Exception as e:
                    print(f"Optimization error: {e}")
                    progress_queue.put({'type': 'error', 'message': str(e)})
            
            # Start optimization in background thread
            opt_thread = threading.Thread(target=run_optimization)
            opt_thread.start()
            
            # Process progress updates
            while True:
                try:
                    # Check for progress updates
                    update = progress_queue.get(timeout=1)
                    
                    if update['type'] == 'embeddings_start':
                        yield f"data: {json.dumps({'status': 'embeddings', 'message': 'Generating embeddings...'})}\n\n"
                    elif update['type'] == 'embeddings_complete':
                        yield f"data: {json.dumps({'status': 'embeddings_complete', 'message': 'Embeddings generated', 'embeddings_session_id': update['session_id']})}\n\n"
                    elif update['type'] == 'optimization_start':
                        yield f"data: {json.dumps({'status': 'optimization_start', 'message': 'Starting elbow detection...'})}\n\n"
                    elif update['type'] == 'progress':
                        yield f"data: {json.dumps({'status': 'optimizing', 'message': update['message']})}\n\n"
                    elif update['type'] == 'complete':
                        yield f"data: {json.dumps({'status': 'optimization_complete', 'message': 'Optimization complete'})}\n\n"
                        yield f"data: {json.dumps({'status': 'results', 'results': optimization_result['results']})}\n\n"
                        break
                    elif update['type'] == 'error':
                        yield f"data: {json.dumps({'status': 'error', 'message': update['message']})}\n\n"
                        break
                    
                except queue.Empty:
                    # No update received, check if thread is still alive
                    if not opt_thread.is_alive():
                        break
                    continue
                    
        except Exception as e:
            print(f"Stream error: {e}")
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"
    
    return Response(stream_with_context(generate_optimization_progress()), 
                   mimetype='text/plain',
                   headers={'Cache-Control': 'no-cache'})

@main.route('/api/add-elbow-refinement', methods=['POST'])
def api_add_elbow_refinement():
    """API endpoint for adding refinement points around a target region"""
    from flask import Response, stream_with_context
    from app.clustering_service import LearningGoalsClusteringService
    import json
    import threading
    import queue
    
    def generate_refinement_progress():
        try:
            # Get request data
            request_data = request.get_json() or {}
            target_region_center = request_data.get('target_region_center')
            existing_results_data = request_data.get('existing_results', [])
            region_radius = request_data.get('region_radius', None)
            course_filter = request_data.get('course_filter', None)  # New course filter parameter
            
            if not target_region_center:
                yield f"data: {json.dumps({'status': 'error', 'message': 'Target region center is required'})}\n\n"
                return
            
            # Send initial status
            starting_message = f"Adding refinement points around k={target_region_center}..."
            yield f"data: {json.dumps({'status': 'starting', 'message': starting_message})}\n\n"
            
            # Get all learning goals from Firebase
            if course_filter:
                loading_message = f"Loading learning goals from course: {course_filter}..."
                yield f"data: {json.dumps({'status': 'loading', 'message': loading_message})}\n\n"
            else:
                yield f"data: {json.dumps({'status': 'loading', 'message': 'Loading learning goals...'})}\n\n"
            
            all_documents = search_documents(limit=1000)
            all_goals = []
            for doc in all_documents:
                # Apply course filter if specified
                if course_filter and doc.course_name != course_filter:
                    continue
                all_goals.extend(doc.learning_goals)
            
            if course_filter:
                loaded_message = f"Loaded {len(all_goals)} learning goals from course: {course_filter}"
                yield f"data: {json.dumps({'status': 'loaded', 'message': loaded_message, 'totalGoals': len(all_goals)})}\n\n"
            else:
                loaded_message = f"Loaded {len(all_goals)} learning goals"
                yield f"data: {json.dumps({'status': 'loaded', 'message': loaded_message, 'totalGoals': len(all_goals)})}\n\n"
            
            # Initialize clustering service and generate embeddings
            clustering_service = LearningGoalsClusteringService()
            
            yield f"data: {json.dumps({'status': 'embeddings', 'message': 'Generating embeddings...'})}\n\n"
            embeddings = clustering_service.generate_embeddings(all_goals)
            
            # Store/update embeddings for download
            session_id = str(uuid.uuid4())
            embeddings_storage[session_id] = {
                'goals': all_goals,
                'embeddings': embeddings.tolist(),
                'timestamp': time.time()
            }
            
            yield f"data: {json.dumps({'status': 'embeddings_complete', 'message': 'Embeddings ready', 'embeddings_session_id': session_id})}\n\n"
            
            # Use a queue for progress updates
            progress_queue = queue.Queue()
            refinement_result = {}
            
            def run_refinement():
                try:
                    def stream_progress(message, completed, total, percentage):
                        progress_queue.put({
                            'type': 'progress',
                            'message': message,
                            'completed': completed,
                            'total': total,
                            'percentage': percentage
                        })
                    
                    # Add refinement points
                    progress_queue.put({'type': 'refinement_start'})
                    
                    updated_results = clustering_service.add_elbow_refinement_points(
                        embeddings=embeddings,
                        existing_results=existing_results_data,
                        target_region_center=target_region_center,
                        region_radius=region_radius,
                        progress_callback=stream_progress
                    )
                    
                    refinement_result['results'] = {
                        'success': True,
                        'updated_results': updated_results,
                        'new_points_added': len(updated_results) - len(existing_results_data),
                        'total_points': len(updated_results)
                    }
                    
                    progress_queue.put({'type': 'complete'})
                    
                except Exception as e:
                    print(f"Refinement error: {e}")
                    progress_queue.put({'type': 'error', 'message': str(e)})
            
            # Start refinement in background thread
            ref_thread = threading.Thread(target=run_refinement)
            ref_thread.start()
            
            # Process progress updates
            while True:
                try:
                    update = progress_queue.get(timeout=1)
                    
                    if update['type'] == 'refinement_start':
                        yield f"data: {json.dumps({'status': 'refining', 'message': 'Adding refinement points...'})}\n\n"
                    elif update['type'] == 'progress':
                        yield f"data: {json.dumps({'status': 'refining', 'message': update['message']})}\n\n"
                    elif update['type'] == 'complete':
                        yield f"data: {json.dumps({'status': 'refinement_complete', 'message': 'Refinement complete'})}\n\n"
                        yield f"data: {json.dumps({'status': 'results', 'results': refinement_result['results']})}\n\n"
                        break
                    elif update['type'] == 'error':
                        yield f"data: {json.dumps({'status': 'error', 'message': update['message']})}\n\n"
                        break
                    
                except queue.Empty:
                    if not ref_thread.is_alive():
                        break
                    continue
                    
        except Exception as e:
            print(f"Refinement stream error: {e}")
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"
    
    return Response(stream_with_context(generate_refinement_progress()), 
                   mimetype='text/plain',
                   headers={'Cache-Control': 'no-cache'})

@main.route('/api/extract-learning-goals', methods=['POST'])
def extract_learning_goals_from_text():
    """Extract learning goals from provided text content"""
    try:
        data = request.get_json()
        text = data.get('text')
        custom_system_message = data.get('system_message')
        model = data.get('model', 'gpt-4o')
        
        if not text or not text.strip():
            return jsonify({'success': False, 'message': 'No text provided'}), 400
        
        # Extract learning goals using OpenAI
        api_key = current_app.config['OPENAI_API_KEY']
        extraction_result = extract_learning_goals(text, api_key, custom_system_message, model=model)
        
        return jsonify({
            'success': True,
            'learning_goals': extraction_result['learning_goals'],
            'system_message_used': extraction_result['system_message_used']
        })
        
    except Exception as e:
        print(f"Error extracting learning goals from text: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@main.route('/api/store-session-data', methods=['POST'])
def store_session_data():
    """Store processed file data in server session (called from client-side)"""
    try:
        data = request.get_json()
        processed_files = data.get('processed_files', [])
        
        if not processed_files:
            return jsonify({'success': False, 'message': 'No processed files provided'}), 400
        
        # Store in session
        session['processed_files'] = processed_files
        
        return jsonify({'success': True, 'message': f'Stored {len(processed_files)} files in session'})
        
    except Exception as e:
        print(f"Error storing session data: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@main.route('/api/download-embeddings/<session_id>')
def download_embeddings(session_id):
    """Download embeddings data as CSV"""
    try:
        # Get embeddings data from temporary storage
        if session_id not in embeddings_storage:
            return jsonify({'error': 'Embeddings data not found or expired'}), 404
        
        embeddings_data = embeddings_storage[session_id]
        goals = embeddings_data['goals']
        embeddings = embeddings_data['embeddings']
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['learning_goal', 'embedding_vector'])
        
        # Write data
        for goal, embedding in zip(goals, embeddings):
            # Convert embedding to a comma-separated string
            embedding_str = '[' + ','.join([f'{x:.6f}' for x in embedding]) + ']'
            writer.writerow([goal, embedding_str])
        
        # Create response
        output.seek(0)
        response = Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=learning_goals_embeddings.csv'}
        )
        
        # Clean up after download (optional - could keep for a while)
        # del embeddings_storage[session_id]
        
        return response
        
    except Exception as e:
        print(f"Error downloading embeddings: {e}")
        return jsonify({'error': str(e)}), 500

@main.route('/debug_storage_paths')
def debug_storage_paths():
    """Debug page to check for documents with storage path issues"""
    from app.firebase_service import search_documents, smart_resolve_storage_path
    import re
    
    # Get all documents
    all_documents = search_documents(limit=1000)
    
    results = {
        'total_documents': len(all_documents),
        'duplicated_path_count': 0,
        'working_paths': 0,
        'broken_paths': 0,
        'auto_correctable': 0,
        'documents': []
    }
    
    # Pattern to detect duplicated filenames
    duplicated_pattern = re.compile(r'pdfs/(.+?)_(.+?)_\2$')
    
    for doc in all_documents:
        if not doc.storage_path:
            continue
            
        doc_info = {
            'id': doc.id,
            'name': doc.name,
            'creator': doc.creator,
            'storage_path': doc.storage_path,
            'has_duplicated_pattern': False,
            'path_works': False,
            'auto_correctable': False,
            'corrected_path': None
        }
        
        # Check if it has duplicated pattern
        if duplicated_pattern.match(doc.storage_path):
            doc_info['has_duplicated_pattern'] = True
            results['duplicated_path_count'] += 1
        
        # Try smart resolution (without fixing in DB)
        try:
            path_result = smart_resolve_storage_path(doc, fix_in_db=False)
            
            if path_result['resolved_path']:
                doc_info['path_works'] = True
                results['working_paths'] += 1
                
                if path_result['corrected']:
                    doc_info['auto_correctable'] = True
                    doc_info['corrected_path'] = path_result['resolved_path']
                    results['auto_correctable'] += 1
            else:
                results['broken_paths'] += 1
        except Exception as e:
            results['broken_paths'] += 1
            doc_info['error'] = str(e)
        
        results['documents'].append(doc_info)
    
    return jsonify(results)

@main.route('/cluster-tree')
def cluster_tree():
    """Hierarchical clustering tree page"""
    return render_template('cluster_tree.html')

@main.route('/api/cluster-tree', methods=['POST'])
def api_cluster_tree():
    """API endpoint for building hierarchical clustering trees"""
    try:
        data = request.get_json()
        n_levels = data.get('n_levels', 8)
        linkage_method = data.get('linkage_method', 'ward')
        sampling_method = data.get('sampling_method', 'intelligent')
        course_filter = data.get('course')
        
        print(f"Building hierarchical tree: {n_levels} levels, {linkage_method} linkage, {sampling_method} sampling")
        
        # Validate parameters
        if n_levels < 3 or n_levels > 15:
            return jsonify({
                'success': False,
                'message': 'Number of levels must be between 3 and 15'
            })
        
        if linkage_method not in ['ward', 'complete', 'average', 'single']:
            return jsonify({
                'success': False,
                'message': 'Invalid linkage method'
            })
        
        if sampling_method not in ['natural', 'intelligent']:
            return jsonify({
                'success': False,
                'message': 'Invalid sampling method'
            })
        
        # Get all learning goals from Firebase
        all_documents = search_documents(limit=1000)
        
        # Extract all learning goals with course filtering
        all_goals = []
        all_sources = []
        
        for doc in all_documents:
            # Apply course filter if specified
            if course_filter and doc.course_name != course_filter:
                continue
                
            for goal in doc.learning_goals:
                all_goals.append(goal)
                all_sources.append({
                    'document_name': doc.name,
                    'creator': doc.creator,
                    'course_name': doc.course_name,
                    'institution': doc.institution
                })
        
        if len(all_goals) < 2:
            return jsonify({
                'success': False,
                'message': 'Need at least 2 learning goals to build a tree'
            })
        
        # Generate embeddings (same way as regular clustering)
        from app.clustering_service import LearningGoalsClusteringService
        clustering_service = LearningGoalsClusteringService()
        print(f"📊 Generating embeddings for {len(all_goals)} learning goals...")
        embeddings = clustering_service.generate_embeddings(all_goals)
        
        # Build hierarchical tree
        print(f"🌳 Building hierarchical tree...")
        tree_result = clustering_service.build_hierarchical_tree(
            embeddings, all_goals, all_sources, n_levels, linkage_method, sampling_method
        )
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            """Recursively convert numpy types to native Python types"""
            import numpy as np
            
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        tree_result = convert_numpy_types(tree_result)
        
        print(f"✅ Hierarchical tree built successfully!")
        
        return jsonify({
            'success': True,
            **tree_result
        })
        
    except Exception as e:
        print(f"Hierarchical tree building error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Tree building failed: {str(e)}'
        })

@main.route('/api/cluster-tree-stream', methods=['POST'])
def api_cluster_tree_stream():
    """API endpoint for building hierarchical clustering trees with progress streaming"""
    from flask import Response, stream_with_context
    from app.clustering_service import LearningGoalsClusteringService
    import json
    import time
    
    def generate_tree_progress():
        try:
            # Get request data
            request_data = request.get_json() or {}
            n_levels = request_data.get('n_levels', 8)
            linkage_method = request_data.get('linkage_method', 'ward')
            sampling_method = request_data.get('sampling_method', 'intelligent')
            course_filter = request_data.get('course')
            
            # Send initial status
            yield f"data: {json.dumps({'status': 'starting', 'message': 'Initializing hierarchical tree builder...'})}\n\n"
            
            # Validate parameters
            if n_levels < 3 or n_levels > 15:
                yield f"data: {json.dumps({'status': 'error', 'message': 'Number of levels must be between 3 and 15'})}\n\n"
                return
            
            if linkage_method not in ['ward', 'complete', 'average', 'single']:
                yield f"data: {json.dumps({'status': 'error', 'message': 'Invalid linkage method'})}\n\n"
                return
            
            if sampling_method not in ['natural', 'intelligent']:
                yield f"data: {json.dumps({'status': 'error', 'message': 'Invalid sampling method'})}\n\n"
                return
            
            # Step 1: Loading data
            if course_filter:
                loading_message = f"Loading learning goals from course: {course_filter}..."
                yield f"data: {json.dumps({'status': 'loading', 'message': loading_message, 'step': 'loading', 'progress': 0})}\n\n"
            else:
                yield f"data: {json.dumps({'status': 'loading', 'message': 'Loading learning goals from database...', 'step': 'loading', 'progress': 0})}\n\n"
            
            all_documents = search_documents(limit=1000)
            all_goals = []
            all_sources = []
            
            for doc in all_documents:
                # Apply course filter if specified
                if course_filter and doc.course_name != course_filter:
                    continue
                    
                for goal in doc.learning_goals:
                    all_goals.append(goal)
                    all_sources.append({
                        'document_name': doc.name,
                        'creator': doc.creator,
                        'course_name': doc.course_name,
                        'institution': doc.institution
                    })
            
            if len(all_goals) < 2:
                error_message = f'Need at least 2 learning goals to build a tree. Found {len(all_goals)} goals.'
                if course_filter:
                    error_message += f' Course "{course_filter}" has too few goals.'
                yield f"data: {json.dumps({'status': 'error', 'message': error_message})}\n\n"
                return
            
            loaded_message = f"Loaded {len(all_goals)} learning goals"
            if course_filter:
                loaded_message += f" from course: {course_filter}"
            yield f"data: {json.dumps({'status': 'loaded', 'message': loaded_message, 'step': 'loading', 'progress': 100, 'totalGoals': len(all_goals)})}\n\n"
            
            # Step 2: Generate embeddings
            yield f"data: {json.dumps({'status': 'embeddings', 'message': 'Generating STEM-optimized embeddings...', 'step': 'embeddings', 'progress': 0})}\n\n"
            
            clustering_service = LearningGoalsClusteringService()
            embeddings = clustering_service.generate_embeddings(all_goals)
            
            yield f"data: {json.dumps({'status': 'embeddings_complete', 'message': 'Embeddings generated successfully', 'step': 'embeddings', 'progress': 100})}\n\n"
            
            # Step 3: Build hierarchical tree
            yield f"data: {json.dumps({'status': 'tree_building', 'message': f'Building {n_levels}-level hierarchical tree...', 'step': 'tree-building', 'progress': 0})}\n\n"
            
            # Add some progress updates during tree building
            yield f"data: {json.dumps({'status': 'tree_building', 'message': 'Computing distance matrix...', 'step': 'tree-building', 'progress': 25})}\n\n"
            
            yield f"data: {json.dumps({'status': 'tree_building', 'message': f'Applying {linkage_method} linkage clustering...', 'step': 'tree-building', 'progress': 50})}\n\n"
            
            yield f"data: {json.dumps({'status': 'tree_building', 'message': f'Using {sampling_method} sampling strategy...', 'step': 'tree-building', 'progress': 75})}\n\n"
            
            tree_result = clustering_service.build_hierarchical_tree(
                embeddings, all_goals, all_sources, n_levels, linkage_method, sampling_method
            )
            
            yield f"data: {json.dumps({'status': 'tree_building_complete', 'message': 'Hierarchical tree structure built', 'step': 'tree-building', 'progress': 100})}\n\n"
            
            # Step 4: Calculate quality metrics
            yield f"data: {json.dumps({'status': 'metrics', 'message': 'Calculating quality metrics...', 'step': 'metrics', 'progress': 0})}\n\n"
            
            yield f"data: {json.dumps({'status': 'metrics', 'message': 'Computing silhouette scores...', 'step': 'metrics', 'progress': 33})}\n\n"
            
            yield f"data: {json.dumps({'status': 'metrics', 'message': 'Analyzing cluster separation...', 'step': 'metrics', 'progress': 66})}\n\n"
            
            yield f"data: {json.dumps({'status': 'metrics_complete', 'message': 'Quality metrics calculated', 'step': 'metrics', 'progress': 100})}\n\n"
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                """Recursively convert numpy types to native Python types"""
                import numpy as np
                
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            tree_result = convert_numpy_types(tree_result)
            
            # Send final results
            final_result = {
                'success': True,
                **tree_result
            }
            
            yield f"data: {json.dumps({'status': 'complete', 'message': 'Hierarchical tree built successfully!', 'results': final_result})}\n\n"
            
        except Exception as e:
            print(f"Hierarchical tree building error: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'status': 'error', 'message': f'Tree building failed: {str(e)}'})}\n\n"
    
    return Response(stream_with_context(generate_tree_progress()), 
                   mimetype='text/plain',
                   headers={'Cache-Control': 'no-cache'})

@main.route('/api/export-tree-level')
def api_export_tree_level():
    """Export a specific tree level as CSV or JSON"""
    try:
        node_id = request.args.get('node_id')
        format_type = request.args.get('format', 'csv').lower()
        
        if not node_id:
            return jsonify({'success': False, 'message': 'Node ID required'})
        
        # For this implementation, we'll need to rebuild the tree or store it in session
        # For now, return an error message suggesting the user use the view option
        return jsonify({
            'success': False,
            'message': 'Export functionality requires session storage. Please use "View Flattened" instead.'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Export failed: {str(e)}'
        })

@main.route('/cluster-tree-flattened')
def cluster_tree_flattened():
    """View flattened version of a tree level"""
    node_id = request.args.get('node_id')
    if not node_id:
        return "Node ID required", 400
    
    # For this implementation, show a placeholder page
    # In a full implementation, you'd store the tree in session or database
    return f"""
    <html>
    <head>
        <title>Flattened View - {node_id}</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>
        <div class="container mt-4">
            <div class="card">
                <div class="card-header">
                    <h3>Flattened View: {node_id}</h3>
                </div>
                <div class="card-body">
                    <div class="alert alert-info">
                        <strong>Note:</strong> This is a placeholder for the flattened view of cluster {node_id}.
                        In a full implementation, this would show all learning goals that belong to this cluster
                        and its sub-clusters in a traditional flat format similar to the regular clustering page.
                    </div>
                    <p>To implement this fully, you would:</p>
                    <ul>
                        <li>Store the tree structure in session or database</li>
                        <li>Retrieve the specific node by ID</li>
                        <li>Flatten all goals from the node and its children</li>
                        <li>Display them in a traditional clustering format</li>
                    </ul>
                    <a href="javascript:window.close()" class="btn btn-secondary">Close Window</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """ 

# HDBSCAN Clustering Routes
@main.route('/cluster-hdbscan')
def cluster_hdbscan_page():
    """HDBSCAN clustering page"""
    return render_template('cluster_hdbscan.html')

@main.route('/api/cluster-hdbscan', methods=['POST'])
def api_cluster_hdbscan():
    """API endpoint for HDBSCAN clustering with progress streaming"""
    from flask import Response, stream_with_context
    from app.clustering_service import LearningGoalsClusteringService
    import json
    import threading
    import queue
    import time
    import uuid
    
    def generate_hdbscan_progress():
        try:
            # Get request data
            request_data = request.get_json() or {}
            min_cluster_size = request_data.get('min_cluster_size', 5)
            min_samples = request_data.get('min_samples', None)
            alpha = request_data.get('alpha', 1.0)
            epsilon = request_data.get('epsilon', 0.0)
            course_filter = request_data.get('course_filter', None)
            
            # Validate basic parameters first
            if min_cluster_size < 2:
                min_cluster_size = 2
            
            if min_samples is not None and min_samples < 1:
                min_samples = 1
            
            # Validate alpha parameter - HDBSCAN requires alpha > 0
            if alpha is None or alpha <= 0:
                alpha = 1.0
                print(f"⚠️ Invalid alpha value, setting to default: {alpha}")
            
            # Convert alpha to float to ensure proper type
            try:
                alpha = float(alpha)
                if alpha <= 0:
                    alpha = 1.0
                    print(f"⚠️ Alpha must be > 0, setting to default: {alpha}")
            except (ValueError, TypeError):
                alpha = 1.0
                print(f"⚠️ Could not convert alpha to float, setting to default: {alpha}")
            
            # Send initial status
            yield f"data: {json.dumps({'status': 'starting', 'message': 'Initializing HDBSCAN clustering...'})}\n\n"
            
            # Get all learning goals from Firebase
            if course_filter:
                loading_message = f"Loading learning goals from course: {course_filter}..."
                yield f"data: {json.dumps({'status': 'loading', 'message': loading_message})}\n\n"
            else:
                yield f"data: {json.dumps({'status': 'loading', 'message': 'Loading learning goals...'})}\n\n"
            
            all_documents = search_documents(limit=1000)
            all_goals = []
            all_sources = []
            
            for doc in all_documents:
                # Apply course filter if specified
                if course_filter and doc.course_name != course_filter:
                    continue
                    
                for goal in doc.learning_goals:
                    all_goals.append(goal)
                    all_sources.append({
                        'document_name': doc.name,
                        'creator': doc.creator,
                        'course_name': doc.course_name
                    })
            
            # Validate and adjust parameters based on actual data
            if len(all_goals) < 2:
                error_message = f'Need at least 2 learning goals for HDBSCAN clustering. Found {len(all_goals)} goals.'
                if course_filter:
                    error_message += f' Course "{course_filter}" has too few goals.'
                yield f"data: {json.dumps({'status': 'error', 'message': error_message})}\n\n"
                return
            
            if len(all_goals) < min_cluster_size:
                # Auto-adjust min_cluster_size to be reasonable
                adjusted_min_cluster_size = max(2, min(min_cluster_size, len(all_goals) // 3))
                if adjusted_min_cluster_size < 2:
                    adjusted_min_cluster_size = 2
                    
                warning_message = f'Adjusted min_cluster_size from {min_cluster_size} to {adjusted_min_cluster_size} based on {len(all_goals)} available goals.'
                yield f"data: {json.dumps({'status': 'warning', 'message': warning_message})}\n\n"
                min_cluster_size = adjusted_min_cluster_size
                
                # Also adjust min_samples if needed
                if min_samples is not None and min_samples > min_cluster_size:
                    min_samples = min_cluster_size
                    yield f"data: {json.dumps({'status': 'warning', 'message': f'Adjusted min_samples to {min_samples} to match min_cluster_size'})}\n\n"
            
            loaded_message = f"Loaded {len(all_goals)} learning goals"
            if course_filter:
                loaded_message += f" from course: {course_filter}"
            yield f"data: {json.dumps({'status': 'loaded', 'message': loaded_message, 'totalGoals': len(all_goals)})}\n\n"
            
            # Initialize clustering service
            clustering_service = LearningGoalsClusteringService()
            
            # Generate embeddings
            yield f"data: {json.dumps({'status': 'embeddings', 'message': 'Generating STEM-optimized embeddings...', 'step': 'embeddings', 'progress': 0})}\n\n"
            embeddings = clustering_service.generate_embeddings(all_goals)
            
            # Store embeddings for download
            session_id = str(uuid.uuid4())
            embeddings_storage[session_id] = {
                'goals': all_goals,
                'embeddings': embeddings.tolist(),
                'timestamp': time.time()
            }
            
            yield f"data: {json.dumps({'status': 'embeddings_complete', 'message': 'Embeddings generated', 'step': 'embeddings', 'progress': 100, 'embeddings_session_id': session_id})}\n\n"
            
            # Perform HDBSCAN clustering
            yield f"data: {json.dumps({'status': 'clustering', 'message': 'Performing HDBSCAN clustering...', 'step': 'clustering', 'progress': 0})}\n\n"
            
            labels = clustering_service.cluster_hdbscan(
                embeddings, 
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                alpha=alpha,
                epsilon=epsilon
            )
            
            yield f"data: {json.dumps({'status': 'clustering_complete', 'message': 'HDBSCAN clustering completed', 'step': 'clustering', 'progress': 100})}\n\n"
            
            # Calculate quality metrics
            yield f"data: {json.dumps({'status': 'metrics', 'message': 'Calculating quality metrics...', 'step': 'metrics', 'progress': 0})}\n\n"
            
            quality_metrics = clustering_service.calculate_hdbscan_quality_metrics(embeddings, labels)
            
            yield f"data: {json.dumps({'status': 'metrics', 'message': 'Calculating cluster stability...', 'step': 'metrics', 'progress': 50})}\n\n"
            
            # Format clusters and outliers
            formatted_clusters, outliers = clustering_service.format_hdbscan_clusters(
                all_goals, all_sources, labels, include_outliers=True
            )
            
            yield f"data: {json.dumps({'status': 'metrics_complete', 'message': 'Quality metrics calculated', 'step': 'metrics', 'progress': 100})}\n\n"
            
            # Prepare results
            results = {
                'success': True,
                'clusters': formatted_clusters,
                'outliers': outliers,
                'metrics': quality_metrics,
                'parameters': {
                    'min_cluster_size': min_cluster_size,
                    'min_samples': min_samples if min_samples is not None else min_cluster_size,
                    'alpha': alpha,
                    'epsilon': epsilon
                },
                'total_goals': len(all_goals),
                'course_filter': course_filter
            }
            
            # Send final results
            yield f"data: {json.dumps({'status': 'complete', 'message': 'HDBSCAN clustering complete!', 'results': results})}\n\n"
            
        except Exception as e:
            print(f"HDBSCAN clustering error: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'status': 'error', 'message': f'HDBSCAN clustering failed: {str(e)}'})}\n\n"
    
    return Response(stream_with_context(generate_hdbscan_progress()), 
                   mimetype='text/plain',
                   headers={'Cache-Control': 'no-cache'})

@main.route('/api/optimize-hdbscan', methods=['POST'])
def api_optimize_hdbscan():
    """API endpoint for HDBSCAN parameter optimization with progress streaming"""
    from flask import Response, stream_with_context
    from app.clustering_service import LearningGoalsClusteringService
    import json
    import threading
    import queue
    import time
    import uuid
    
    def generate_optimization_progress():
        try:
            # Get request data
            request_data = request.get_json() or {}
            min_cluster_size_range = request_data.get('min_cluster_size_range', None)
            min_samples_range = request_data.get('min_samples_range', None)
            course_filter = request_data.get('course_filter', None)
            
            # Send initial status
            yield f"data: {json.dumps({'status': 'starting', 'message': 'Initializing HDBSCAN parameter optimization...'})}\n\n"
            
            # Get all learning goals from Firebase
            if course_filter:
                loading_message = f"Loading learning goals from course: {course_filter}..."
                yield f"data: {json.dumps({'status': 'loading', 'message': loading_message})}\n\n"
            else:
                yield f"data: {json.dumps({'status': 'loading', 'message': 'Loading learning goals...'})}\n\n"
            
            all_documents = search_documents(limit=1000)
            
            # Extract learning goals with course filtering
            all_goals = []
            for doc in all_documents:
                # Apply course filter if specified
                if course_filter and doc.course_name != course_filter:
                    continue
                all_goals.extend(doc.learning_goals)
            
            if course_filter:
                loaded_message = f"Loaded {len(all_goals)} learning goals from course: {course_filter}"
                yield f"data: {json.dumps({'status': 'loaded', 'message': loaded_message, 'totalGoals': len(all_goals)})}\n\n"
            else:
                loaded_message = f"Loaded {len(all_goals)} learning goals from {len(all_documents)} documents"
                yield f"data: {json.dumps({'status': 'loaded', 'message': loaded_message, 'totalGoals': len(all_goals)})}\n\n"
            
            if len(all_goals) < 4:
                if course_filter:
                    error_message = f'Need at least 4 learning goals for optimization. Course "{course_filter}" only has {len(all_goals)} goals.'
                    yield f"data: {json.dumps({'status': 'error', 'message': error_message})}\n\n"
                else:
                    yield f"data: {json.dumps({'status': 'error', 'message': 'Need at least 4 learning goals for optimization'})}\n\n"
                return
            
            # Initialize clustering service
            clustering_service = LearningGoalsClusteringService()
            
            # Use a queue for progress updates from the optimization thread
            progress_queue = queue.Queue()
            optimization_result = {}
            
            def run_optimization():
                try:
                    # Stream progress callback
                    def stream_progress(message, completed, total, percentage):
                        progress_queue.put({
                            'type': 'progress',
                            'message': message,
                            'completed': completed,
                            'total': total,
                            'percentage': percentage
                        })
                    
                    # Check for stop signal (implement basic mechanism)
                    global optimization_should_stop
                    optimization_should_stop = False
                    
                    def check_stop_wrapper(original_func):
                        def wrapper(*args, **kwargs):
                            if optimization_should_stop:
                                raise InterruptedError("Optimization stopped by user")
                            return original_func(*args, **kwargs)
                        return wrapper
                    
                    # Generate embeddings first
                    progress_queue.put({'type': 'embeddings_start'})
                    embeddings = clustering_service.generate_embeddings(all_goals)
                    
                    # Store embeddings for download
                    session_id = str(uuid.uuid4())
                    embeddings_storage[session_id] = {
                        'goals': all_goals,
                        'embeddings': embeddings.tolist(),
                        'timestamp': time.time()
                    }
                    
                    progress_queue.put({'type': 'embeddings_complete', 'session_id': session_id})
                    
                    # Start optimization
                    progress_queue.put({'type': 'optimization_start'})
                    
                    results, best_composite_k, best_separation_k = clustering_service.find_optimal_cluster_sizes(
                        embeddings, 
                        progress_callback=stream_progress,
                        use_elbow_detection=True
                    )
                    
                    if results:
                        # Calculate additional metrics for the best results
                        embeddings_final = clustering_service.generate_embeddings(all_goals)
                        
                        # Get metrics for best composite result
                        labels_composite = clustering_service.cluster_fast(embeddings_final, best_composite_k)
                        silhouette_composite = clustering_service.calculate_cluster_quality_fast(embeddings_final, labels_composite)
                        separation_composite, cohesion_composite = clustering_service.calculate_cluster_separation_metrics_fast(embeddings_final, labels_composite)
                        
                        # Get metrics for best separation result  
                        labels_separation = clustering_service.cluster_fast(embeddings_final, best_separation_k)
                        silhouette_separation = clustering_service.calculate_cluster_quality_fast(embeddings_final, labels_separation)
                        separation_separation, cohesion_separation = clustering_service.calculate_cluster_separation_metrics_fast(embeddings_final, labels_separation)
                        
                        optimization_result['results'] = {
                            'success': True,
                            'total_goals': len(all_goals),
                            'best_composite_k': best_composite_k,
                            'best_separation_k': best_separation_k,
                            'best_cohesion_k': results.get('best_cohesion_k', best_composite_k),
                            'best_silhouette_k': results.get('best_silhouette_k', best_composite_k),
                            'max_composite_score': round(float(results['max_composite']), 3),
                            'max_separation_score': round(float(results['max_separation']), 3),
                            'max_cohesion_score': round(float(results['max_cohesion']), 3),
                            'max_silhouette_score': round(float(results['max_silhouette']), 3),
                            'recommendation': {
                                'explanation': f"For {len(all_goals)} learning goals, the algorithm recommends {best_composite_k} clusters for balanced quality, or {best_separation_k} clusters for maximum separation."
                            }
                        }
                    else:
                        optimization_result['results'] = {
                            'success': False,
                            'message': 'Optimization failed to find suitable cluster sizes'
                        }
                    
                    progress_queue.put({'type': 'complete'})
                    
                except Exception as e:
                    print(f"Optimization error: {e}")
                    progress_queue.put({'type': 'error', 'message': str(e)})
            
            # Start optimization in background thread
            opt_thread = threading.Thread(target=run_optimization)
            opt_thread.start()
            
            # Process progress updates
            while True:
                try:
                    # Check for progress updates
                    update = progress_queue.get(timeout=1)
                    
                    if update['type'] == 'embeddings_start':
                        yield f"data: {json.dumps({'status': 'embeddings', 'message': 'Generating embeddings...'})}\n\n"
                    elif update['type'] == 'embeddings_complete':
                        yield f"data: {json.dumps({'status': 'embeddings_complete', 'message': 'Embeddings generated', 'embeddings_session_id': update['session_id']})}\n\n"
                    elif update['type'] == 'optimization_start':
                        yield f"data: {json.dumps({'status': 'optimization_start', 'message': 'Starting elbow detection...'})}\n\n"
                    elif update['type'] == 'progress':
                        yield f"data: {json.dumps({'status': 'optimizing', 'message': update['message']})}\n\n"
                    elif update['type'] == 'complete':
                        yield f"data: {json.dumps({'status': 'optimization_complete', 'message': 'Optimization complete'})}\n\n"
                        yield f"data: {json.dumps({'status': 'results', 'results': optimization_result['results']})}\n\n"
                        break
                    elif update['type'] == 'error':
                        yield f"data: {json.dumps({'status': 'error', 'message': update['message']})}\n\n"
                        break
                    
                except queue.Empty:
                    # No update received, check if thread is still alive
                    if not opt_thread.is_alive():
                        break
                    continue
                    
        except Exception as e:
            print(f"Stream error: {e}")
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"
    
    return Response(stream_with_context(generate_optimization_progress()), 
                   mimetype='text/plain',
                   headers={'Cache-Control': 'no-cache'})

@main.route('/api/export-hdbscan-clusters', methods=['POST'])
def api_export_hdbscan_clusters():
    """Export HDBSCAN clustering results as CSV or JSON"""
    try:
        data = request.get_json()
        clusters = data.get('clusters', [])
        outliers = data.get('outliers', [])
        export_format = data.get('format', 'csv').lower()
        include_outliers = data.get('include_outliers', True)
        
        if export_format == 'csv':
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(['cluster_id', 'cluster_size', 'learning_goal', 'document_name', 'creator', 'course_name', 'is_outlier'])
            
            # Write cluster data
            for cluster in clusters:
                cluster_id = cluster['id']
                cluster_size = cluster['size']
                
                for goal, source in zip(cluster['goals'], cluster['sources']):
                    writer.writerow([
                        cluster_id,
                        cluster_size,
                        goal,
                        source['document_name'],
                        source['creator'],
                        source['course_name'],
                        False
                    ])
            
            # Write outlier data if requested
            if include_outliers:
                for outlier in outliers:
                    writer.writerow([
                        -1,  # Outlier cluster ID
                        1,   # Outlier size is always 1
                        outlier['goal'],
                        outlier['source']['document_name'],
                        outlier['source']['creator'],
                        outlier['source']['course_name'],
                        True
                    ])
            
            output.seek(0)
            return Response(
                output.getvalue(),
                mimetype='text/csv',
                headers={'Content-Disposition': 'attachment; filename=hdbscan_clusters.csv'}
            )
            
        elif export_format == 'json':
            export_data = {
                'clusters': clusters,
                'outliers': outliers if include_outliers else [],
                'export_timestamp': time.time(),
                'total_clusters': len(clusters),
                'total_outliers': len(outliers),
                'include_outliers': include_outliers
            }
            
            return Response(
                json.dumps(export_data, indent=2),
                mimetype='application/json',
                headers={'Content-Disposition': 'attachment; filename=hdbscan_clusters.json'}
            )
        else:
            return jsonify({'success': False, 'message': 'Unsupported export format'}), 400
            
    except Exception as e:
        print(f"Export error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500 


# Learning Goals Hierarchy Management Routes
@main.route('/learning-goals-hierarchy')
def learning_goals_hierarchy_page():
    """Learning Goals Hierarchy management page"""
    return render_template('learning_goals_hierarchy.html')

@main.route('/api/upload-hierarchy', methods=['POST'])
def api_upload_hierarchy():
    """API endpoint for uploading hierarchical learning goals from CSV"""
    from app.models import LearningGoalsHierarchy
    from app.firebase_service import save_hierarchy
    
    try:
        data = request.get_json()
        hierarchy_data = data.get('hierarchy', [])
        metadata = data.get('metadata', {})
        
        if not hierarchy_data:
            return jsonify({
                'success': False,
                'message': 'No hierarchy data provided'
            })
        
        # Calculate data size estimate
        total_goals = sum(len(node.get('goals', [])) for node in hierarchy_data)
        estimated_size = estimate_hierarchy_size(hierarchy_data)
        optimization_applied = False
        
        print(f"📊 Hierarchy upload stats: {total_goals} goals, estimated size: {estimated_size/1024:.1f}KB")
        
        # Create hierarchy object
        hierarchy = LearningGoalsHierarchy(
            name=f"CSV Upload {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            description="Hierarchy uploaded from CSV file",
            creator="System",  # Could be enhanced to track actual user
            course_name="",
            institution="",
            root_nodes=hierarchy_data,
            metadata=metadata
        )
        
        # Save to database (optimization happens automatically if needed)
        hierarchy_id = save_hierarchy(hierarchy)
        
        # Check if optimization was likely applied based on size
        if estimated_size > 800000:  # 800KB threshold
            optimization_applied = True
            print(f"🔧 Large hierarchy detected, optimization likely applied")
        
        # Calculate stats
        total_groups = len(hierarchy_data)
        avg_goals_per_group = round(total_goals / total_groups, 1) if total_groups > 0 else 0
        
        stats = {
            'total_goals': total_goals,
            'total_groups': total_groups,
            'avg_goals_per_group': avg_goals_per_group,
            'uploaded_at': metadata.get('uploaded_at', datetime.now().isoformat()),
            'hierarchy_id': hierarchy_id,
            'estimated_size_kb': round(estimated_size / 1024, 1),
            'optimization_applied': optimization_applied
        }
        
        return jsonify({
            'success': True,
            'message': 'Hierarchy uploaded successfully',
            'hierarchy_id': hierarchy_id,
            'stats': stats,
            'optimization_applied': optimization_applied
        })
        
    except Exception as e:
        print(f"Hierarchy upload error: {e}")
        error_message = str(e)
        
        # Provide better error messages for common issues
        if "exceeds the maximum allowed size" in error_message:
            error_message = "Hierarchy data is too large. The system is optimizing storage structure automatically. Please try uploading again in a moment."
        elif "timeout" in error_message.lower():
            error_message = "Upload timed out due to large data size. The system will optimize storage and try again automatically."
        
        return jsonify({
            'success': False,
            'message': f'Upload failed: {error_message}'
        })

def estimate_hierarchy_size(hierarchy_data):
    """Estimate the size of hierarchy data in bytes"""
    import json
    
    # Convert to JSON string to estimate size
    try:
        json_str = json.dumps(hierarchy_data)
        return len(json_str.encode('utf-8'))
    except Exception:
        # Fallback estimation
        total_goals = sum(len(node.get('goals', [])) for node in hierarchy_data)
        avg_goal_length = 80  # Estimated average goal length
        overhead_per_node = 200  # Estimated metadata overhead per node
        
        estimated_size = (total_goals * avg_goal_length) + (len(hierarchy_data) * overhead_per_node)
        return estimated_size

@main.route('/api/save-hierarchy', methods=['POST'])
def api_save_hierarchy():
    """API endpoint for saving modified hierarchical learning goals"""
    from app.models import LearningGoalsHierarchy
    from app.firebase_service import save_hierarchy
    
    try:
        data = request.get_json()
        hierarchy_data = data.get('hierarchy', [])
        metadata = data.get('metadata', {})
        
        if not hierarchy_data:
            return jsonify({
                'success': False,
                'message': 'No hierarchy data provided'
            })
        
        # Create updated hierarchy object
        hierarchy = LearningGoalsHierarchy(
            name=f"Updated Hierarchy {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            description="Modified hierarchy with user edits",
            creator="System",  # Could be enhanced to track actual user
            course_name="",
            institution="",
            root_nodes=hierarchy_data,
            metadata=metadata
        )
        
        # Save to database
        hierarchy_id = save_hierarchy(hierarchy)
        
        return jsonify({
            'success': True,
            'message': 'Hierarchy saved successfully',
            'hierarchy_id': hierarchy_id
        })
        
    except Exception as e:
        print(f"Hierarchy save error: {e}")
        return jsonify({
            'success': False,
            'message': f'Save failed: {str(e)}'
        })

@main.route('/api/get-hierarchies', methods=['GET'])
def api_get_hierarchies():
    """API endpoint for retrieving saved hierarchies"""
    from app.firebase_service import get_hierarchies
    
    try:
        # Get query parameters
        limit = request.args.get('limit', 20, type=int)
        creator = request.args.get('creator', None)
        course_name = request.args.get('course_name', None)
        
        # Get hierarchies from database
        hierarchies = get_hierarchies(limit=limit, creator=creator, course_name=course_name)
        
        # Format response
        formatted_hierarchies = []
        for hierarchy in hierarchies:
            formatted_hierarchies.append({
                'id': hierarchy.id,
                'name': hierarchy.name,
                'description': hierarchy.description,
                'creator': hierarchy.creator,
                'course_name': hierarchy.course_name,
                'total_goals': hierarchy.get_total_goals(),
                'total_groups': hierarchy.get_total_groups(),
                'created_at': hierarchy.created_at.isoformat() if hierarchy.created_at else None,
                'modified_at': hierarchy.modified_at.isoformat() if hierarchy.modified_at else None,
                'is_active': hierarchy.is_active
            })
        
        return jsonify({
            'success': True,
            'hierarchies': formatted_hierarchies,
            'total': len(formatted_hierarchies)
        })
        
    except Exception as e:
        print(f"Get hierarchies error: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to retrieve hierarchies: {str(e)}'
        })

@main.route('/api/get-hierarchy/<hierarchy_id>', methods=['GET'])
def api_get_hierarchy(hierarchy_id):
    """API endpoint for retrieving a specific hierarchy"""
    from app.firebase_service import get_hierarchy
    
    try:
        hierarchy = get_hierarchy(hierarchy_id)
        
        if not hierarchy:
            return jsonify({
                'success': False,
                'message': 'Hierarchy not found'
            }), 404
        
        return jsonify({
            'success': True,
            'hierarchy': {
                'id': hierarchy.id,
                'name': hierarchy.name,
                'description': hierarchy.description,
                'creator': hierarchy.creator,
                'course_name': hierarchy.course_name,
                'institution': hierarchy.institution,
                'root_nodes': hierarchy.root_nodes,
                'metadata': hierarchy.metadata,
                'total_goals': hierarchy.get_total_goals(),
                'total_groups': hierarchy.get_total_groups(),
                'created_at': hierarchy.created_at.isoformat() if hierarchy.created_at else None,
                'modified_at': hierarchy.modified_at.isoformat() if hierarchy.modified_at else None,
                'is_active': hierarchy.is_active
            }
        })
        
    except Exception as e:
        print(f"Get hierarchy error: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to retrieve hierarchy: {str(e)}'
        })

@main.route('/api/delete-hierarchy/<hierarchy_id>', methods=['POST'])
def api_delete_hierarchy(hierarchy_id):
    """API endpoint for deleting a hierarchy"""
    from app.firebase_service import delete_hierarchy
    
    try:
        success = delete_hierarchy(hierarchy_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Hierarchy deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to delete hierarchy'
            }), 500
        
    except Exception as e:
        print(f"Delete hierarchy error: {e}")
        return jsonify({
            'success': False,
            'message': f'Delete failed: {str(e)}'
        })

# Tree Artifact Management Routes
@main.route('/artifacts')
def artifacts_page():
    """Artifacts management page"""
    return render_template('artifacts.html')

@main.route('/api/save-artifact', methods=['POST'])
def api_save_artifact():
    """API endpoint for saving tree artifacts from cluster_tree.html"""
    from app.models import Artifact
    from app.firebase_service import save_artifact
    
    try:
        data = request.get_json()
        
        name = data.get('name', '')
        tree_structure = data.get('tree_structure', [])
        parameters = data.get('parameters', {})
        metadata = data.get('metadata', {})
        
        if not name or not tree_structure:
            return jsonify({
                'success': False,
                'message': 'Name and tree structure are required'
            })
        
        # Create artifact object
        artifact = Artifact(
            name=name,
            tree_structure=tree_structure,
            parameters=parameters,
            metadata=metadata
        )
        
        # Save to database
        artifact_id = save_artifact(artifact)
        
        return jsonify({
            'success': True,
            'message': 'Artifact saved successfully',
            'artifact_id': artifact_id,
            'artifact_name': name
        })
        
    except Exception as e:
        print(f"Artifact save error: {e}")
        return jsonify({
            'success': False,
            'message': f'Save failed: {str(e)}'
        })

@main.route('/api/get-artifacts', methods=['GET'])
def api_get_artifacts():
    """API endpoint for retrieving saved artifacts"""
    from app.firebase_service import get_artifacts
    
    try:
        # Get query parameters
        limit = request.args.get('limit', 20, type=int)
        creator = request.args.get('creator', None)
        
        # Get artifacts from database
        artifacts = get_artifacts(limit=limit, creator=creator)
        
        # Format response
        formatted_artifacts = []
        for artifact in artifacts:
            formatted_artifacts.append({
                'id': artifact.id,
                'name': artifact.name,
                'total_goals': artifact.get_total_goals(),
                'parameter_summary': artifact.get_parameter_summary(),
                'created_at': artifact.created_at.isoformat() if artifact.created_at else None,
                'modified_at': artifact.modified_at.isoformat() if artifact.modified_at else None,
                'parameters': artifact.parameters,
                'metadata': artifact.metadata
            })
        
        return jsonify({
            'success': True,
            'artifacts': formatted_artifacts,
            'total': len(formatted_artifacts)
        })
        
    except Exception as e:
        print(f"Get artifacts error: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to retrieve artifacts: {str(e)}'
        })

@main.route('/api/get-artifact/<artifact_id>', methods=['GET'])
def api_get_artifact(artifact_id):
    """API endpoint for retrieving a specific artifact"""
    from app.firebase_service import get_artifact
    
    try:
        artifact = get_artifact(artifact_id)
        
        if not artifact:
            return jsonify({
                'success': False,
                'message': 'Artifact not found'
            }), 404
        
        return jsonify({
            'success': True,
            'artifact': {
                'id': artifact.id,
                'name': artifact.name,
                'tree_structure': artifact.tree_structure,
                'parameters': artifact.parameters,
                'metadata': artifact.metadata,
                'total_goals': artifact.get_total_goals(),
                'parameter_summary': artifact.get_parameter_summary(),
                'created_at': artifact.created_at.isoformat() if artifact.created_at else None,
                'modified_at': artifact.modified_at.isoformat() if artifact.modified_at else None
            }
        })
        
    except Exception as e:
        print(f"Get artifact error: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to retrieve artifact: {str(e)}'
        })

@main.route('/api/update-artifact-label', methods=['POST'])
def api_update_artifact_label():
    """API endpoint for updating artifact node labels"""
    from app.firebase_service import update_artifact_node_label
    
    try:
        data = request.get_json()
        artifact_id = data.get('artifact_id')
        node_id = data.get('node_id')
        new_label = data.get('new_label')
        
        if not all([artifact_id, node_id, new_label]):
            return jsonify({
                'success': False,
                'message': 'artifact_id, node_id, and new_label are required'
            })
        
        success = update_artifact_node_label(artifact_id, node_id, new_label)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Label updated successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to update label'
            })
        
    except Exception as e:
        print(f"Update artifact label error: {e}")
        return jsonify({
            'success': False,
            'message': f'Update failed: {str(e)}'
        })

@main.route('/api/delete-artifact/<artifact_id>', methods=['POST'])
def api_delete_artifact(artifact_id):
    """API endpoint for deleting artifacts"""
    from app.firebase_service import delete_artifact
    
    try:
        success = delete_artifact(artifact_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Artifact deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to delete artifact'
            })
        
    except Exception as e:
        print(f"Delete artifact error: {e}")
        return jsonify({
            'success': False,
            'message': f'Delete failed: {str(e)}'
        })

@main.route('/api/update-artifact-representative-text', methods=['POST'])
def api_update_artifact_representative_text():
    """API endpoint for updating artifact node representative text and state"""
    from app.firebase_service import update_artifact_node_representative_text
    
    try:
        data = request.get_json()
        artifact_id = data.get('artifact_id')
        node_id = data.get('node_id')
        representative_text = data.get('representative_text')
        text_state = data.get('text_state', 'manual')
        
        if not all([artifact_id, node_id, representative_text]):
            return jsonify({
                'success': False,
                'message': 'artifact_id, node_id, and representative_text are required'
            })
        
        success = update_artifact_node_representative_text(
            artifact_id, node_id, representative_text, text_state
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Representative text updated successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to update representative text'
            })
        
    except Exception as e:
        print(f"Update artifact representative text error: {e}")
        return jsonify({
            'success': False,
            'message': f'Update failed: {str(e)}'
        })


@main.route('/api/generate-representative-text', methods=['POST'])
def api_generate_representative_text():
    """API endpoint for generating representative text using AI"""
    from app.firebase_service import update_artifact_node_representative_text
    
    try:
        data = request.get_json()
        artifact_id = data.get('artifact_id')
        node_id = data.get('node_id')
        prompt = data.get('prompt')
        full_prompt = data.get('full_prompt')
        model = data.get('model', 'gpt-4o')
        skip_save = data.get('skip_save', False)  # NEW: Skip database save for batch operations
        
        if not all([artifact_id, node_id, prompt, full_prompt]):
            return jsonify({
                'success': False,
                'message': 'All required fields must be provided'
            })
        
        # Validate model
        if model not in ['gpt-4o', 'gpt-4o-mini', 'gpt-4o-2024-11-20']:
            return jsonify({
                'success': False,
                'message': 'Invalid model specified'
            })
        
        # Generate text using OpenAI
        api_key = current_app.config['OPENAI_API_KEY']
        generation_result = generate_representative_text_with_ai(
            full_prompt, api_key, model
        )
        
        if not generation_result['success']:
            return jsonify({
                'success': False,
                'message': generation_result['message']
            })
        
        generated_text = generation_result['text']
        
        # Only update database if skip_save is False (for individual generations)
        if not skip_save:
            # Update the artifact with the generated text and AI state
            success = update_artifact_node_representative_text(
                artifact_id, node_id, generated_text, 'ai', prompt
            )
            
            if success:
                return jsonify({
                    'success': True,
                    'generated_text': generated_text,
                    'message': 'Representative text generated and saved successfully'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Generated text but failed to save to database'
                })
        else:
            # For batch operations, just return the generated text without saving
            return jsonify({
                'success': True,
                'generated_text': generated_text,
                'message': 'Representative text generated (not saved - batch mode)'
            })
        
    except Exception as e:
        print(f"Generate representative text error: {e}")
        return jsonify({
            'success': False,
            'message': f'Generation failed: {str(e)}'
        })


def generate_representative_text_with_ai(prompt, api_key, model='gpt-4o'):
    """Generate representative text using OpenAI API"""
    try:
        from openai import OpenAI
        
        # Set up OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Create the AI prompt
        messages = [
            {
                "role": "system",
                "content": "You are an educational expert helping to create concise, overarching learning goals that capture the essence of multiple specific learning objectives. Your response should be a single, clear learning goal or skill statement without quotes or extra formatting."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=150,
            temperature=0.3,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        # Extract the generated text
        generated_text = response.choices[0].message.content.strip()
        
        # Clean up the text (remove quotes if present)
        if generated_text.startswith('"') and generated_text.endswith('"'):
            generated_text = generated_text[1:-1]
        
        return {
            'success': True,
            'text': generated_text
        }
        
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return {
            'success': False,
            'message': f'AI generation failed: {str(e)}'
        }

@main.route('/api/batch-update-artifact-representative-texts', methods=['POST'])
def api_batch_update_artifact_representative_texts():
    """API endpoint for batch updating multiple artifact node representative texts"""
    from app.firebase_service import batch_update_artifact_node_representative_texts
    
    try:
        data = request.get_json()
        artifact_id = data.get('artifact_id')
        updates = data.get('updates', [])
        
        if not artifact_id or not updates:
            return jsonify({
                'success': False,
                'message': 'artifact_id and updates array are required'
            })
        
        # Validate updates format
        for update in updates:
            if not all(key in update for key in ['node_id', 'representative_text', 'text_state']):
                return jsonify({
                    'success': False,
                    'message': 'Each update must have node_id, representative_text, and text_state'
                })
            # ai_prompt is optional
        
        success = batch_update_artifact_node_representative_texts(artifact_id, updates)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Successfully updated {len(updates)} representative texts'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to update representative texts'
            })
        
    except Exception as e:
        print(f"Batch update artifact representative texts error: {e}")
        return jsonify({
            'success': False,
            'message': f'Batch update failed: {str(e)}'
        })

# =============================================
# MOVE/DELETE/ARCHIVE API ENDPOINTS
# =============================================

@main.route('/api/move-learning-goal', methods=['POST'])
def api_move_learning_goal():
    """API endpoint for moving learning goals or nodes within the tree"""
    from app.firebase_service import get_artifact, update_artifact_tree_structure, perform_tree_move
    
    try:
        data = request.get_json()
        artifact_id = data.get('artifact_id')
        source_node_id = data.get('source_node_id')
        destination_node_id = data.get('destination_node_id')
        move_type = data.get('move_type')  # 'goal' or 'node'
        goal_index = data.get('goal_index')  # Only for goal moves
        
        if not all([artifact_id, source_node_id, destination_node_id, move_type]):
            return jsonify({
                'success': False,
                'message': 'Missing required parameters'
            })
        
        # Load the artifact
        artifact = get_artifact(artifact_id)
        if not artifact:
            return jsonify({
                'success': False,
                'message': 'Artifact not found'
            })
        
        # Perform the move operation
        updated_tree, success, message = perform_tree_move(
            artifact.tree_structure, 
            source_node_id, 
            destination_node_id, 
            move_type, 
            goal_index
        )
        
        if not success:
            return jsonify({
                'success': False,
                'message': message
            })
        
        # Update the artifact in the database
        update_success = update_artifact_tree_structure(artifact_id, updated_tree)
        
        if not update_success:
            return jsonify({
                'success': False,
                'message': 'Failed to save changes to database'
            })
        
        return jsonify({
            'success': True,
            'message': 'Move completed successfully',
            'updated_tree_structure': updated_tree
        })
        
    except Exception as e:
        print(f"Move learning goal error: {e}")
        return jsonify({
            'success': False,
            'message': f'Move failed: {str(e)}'
        })

# api_delete_node endpoint removed - we only delete individual learning goals now

@main.route('/api/create-group-and-move-goal', methods=['POST'])
def api_create_group_and_move_goal():
    """API endpoint for creating a new group and moving a goal to it"""
    from app.firebase_service import get_artifact, update_artifact_tree_structure, create_new_group_and_move_goal
    
    try:
        data = request.get_json()
        artifact_id = data.get('artifact_id')
        source_node_id = data.get('source_node_id')
        goal_index = data.get('goal_index')
        new_group_name = data.get('new_group_name')
        location = data.get('location')
        
        if not all([artifact_id, source_node_id, new_group_name, location]) or goal_index is None:
            return jsonify({
                'success': False,
                'message': 'Missing required parameters'
            })
        
        # Load the artifact
        artifact = get_artifact(artifact_id)
        if not artifact:
            return jsonify({
                'success': False,
                'message': 'Artifact not found'
            })
        
        # Create new group and move the goal
        updated_tree, success, message = create_new_group_and_move_goal(
            artifact.tree_structure,
            source_node_id,
            goal_index,
            new_group_name,
            location
        )
        
        if not success:
            return jsonify({
                'success': False,
                'message': message
            })
        
        # Update the artifact in the database
        update_success = update_artifact_tree_structure(artifact_id, updated_tree)
        
        if not update_success:
            return jsonify({
                'success': False,
                'message': 'Failed to save changes to database'
            })
        
        return jsonify({
            'success': True,
            'message': 'New group created and goal moved successfully',
            'updated_tree_structure': updated_tree
        })
        
    except Exception as e:
        print(f"Create group and move goal error: {e}")
        return jsonify({
            'success': False,
            'message': f'Operation failed: {str(e)}'
        })

@main.route('/api/delete-goal', methods=['POST'])
def api_delete_goal():
    """API endpoint for deleting individual learning goals"""
    from app.firebase_service import get_artifact, update_artifact_tree_structure, archive_learning_goals, find_node_by_id, get_node_path, remove_goal_from_tree
    
    try:
        data = request.get_json()
        artifact_id = data.get('artifact_id')
        node_id = data.get('node_id')
        goal_index = data.get('goal_index')
        
        if not artifact_id or not node_id or goal_index is None:
            return jsonify({
                'success': False,
                'message': 'Missing required parameters'
            })
        
        # Load the artifact
        artifact = get_artifact(artifact_id)
        if not artifact:
            return jsonify({
                'success': False,
                'message': 'Artifact not found'
            })
        
        # Find the node containing the goal
        node = find_node_by_id(artifact.tree_structure, node_id)
        if not node or not node.get('goals') or goal_index >= len(node['goals']):
            return jsonify({
                'success': False,
                'message': 'Learning goal not found'
            })
        
        # Prepare goal for archiving
        goal_text = node['goals'][goal_index]
        source = node.get('sources', [{}])[goal_index] if node.get('sources') else {}
        node_path = get_node_path(artifact.tree_structure, node_id)
        
        goal_to_archive = {
            'goal_text': goal_text,
            'original_path': node_path,
            'document_name': source.get('document_name', ''),
            'creator': source.get('creator', ''),
            'course_name': source.get('course_name', ''),
            'artifact_id': artifact_id
        }
        
        # Archive the goal
        archived_count = archive_learning_goals([goal_to_archive])
        
        # Remove the goal from the tree and clean up if needed
        updated_tree = remove_goal_from_tree(artifact.tree_structure, node_id, goal_index)
        
        # Update the artifact in the database
        update_success = update_artifact_tree_structure(artifact_id, updated_tree)
        
        if not update_success:
            return jsonify({
                'success': False,
                'message': 'Failed to save changes to database'
            })
        
        return jsonify({
            'success': True,
            'message': 'Learning goal deleted successfully',
            'updated_tree_structure': updated_tree
        })
        
    except Exception as e:
        print(f"Delete goal error: {e}")
        return jsonify({
            'success': False,
            'message': f'Delete failed: {str(e)}'
        })

@main.route('/api/get-archived-goals/<artifact_id>')
def api_get_archived_goals(artifact_id):
    """API endpoint for retrieving archived goals for an artifact"""
    from app.firebase_service import get_archived_goals
    
    try:
        archived_goals = get_archived_goals(artifact_id)
        
        return jsonify({
            'success': True,
            'archived_goals': archived_goals
        })
        
    except Exception as e:
        print(f"Get archived goals error: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to load archived goals: {str(e)}'
        })

@main.route('/api/restore-goal', methods=['POST'])
def api_restore_goal():
    """API endpoint for restoring archived learning goals"""
    from app.firebase_service import get_artifact, restore_archived_goal, update_artifact_tree_structure
    
    try:
        data = request.get_json()
        artifact_id = data.get('artifact_id')
        archived_goal_id = data.get('archived_goal_id')
        
        if not artifact_id or not archived_goal_id:
            return jsonify({
                'success': False,
                'message': 'Missing required parameters'
            })
        
        # Load the artifact
        artifact = get_artifact(artifact_id)
        if not artifact:
            return jsonify({
                'success': False,
                'message': 'Artifact not found'
            })
        
        # Restore the goal and get the updated tree
        updated_tree, success, message = restore_archived_goal(archived_goal_id, artifact.tree_structure)
        
        if not success:
            return jsonify({
                'success': False,
                'message': message
            })
        
        # Update the artifact in the database
        update_success = update_artifact_tree_structure(artifact_id, updated_tree)
        
        if not update_success:
            return jsonify({
                'success': False,
                'message': 'Failed to save changes to database'
            })
        
        return jsonify({
            'success': True,
            'message': 'Learning goal restored successfully',
            'updated_tree_structure': updated_tree
        })
        
    except Exception as e:
        print(f"Restore goal error: {e}")
        return jsonify({
            'success': False,
            'message': f'Restore failed: {str(e)}'
        })

@main.route('/document-url/<document_id>')
def get_document_url(document_id):
    """Generate a fresh Firebase Storage URL for a document"""
    from app.firebase_service import smart_resolve_storage_path, bucket, using_mock
    import datetime as dt
    
    doc = get_document(document_id)
    if not doc or not doc.storage_path:
        return jsonify({'error': 'Document not found or no storage path'}), 404
    
    try:
        # Use smart resolution to find the correct storage path
        path_result = smart_resolve_storage_path(doc, fix_in_db=True)
        
        if not path_result['resolved_path']:
            return jsonify({'error': 'Document file not found in storage'}), 404
        
        if using_mock:
            # For mock storage, return a mock URL
            url = f"https://mock-storage.example.com/{path_result['resolved_path']}"
        else:
            # For real Firebase Storage, generate a signed URL that includes authentication
            try:
                blob = bucket.blob(path_result['resolved_path'])
                # Generate signed URL valid for 24 hours
                url = blob.generate_signed_url(
                    version="v4",
                    expiration=dt.timedelta(hours=24),
                    method="GET"
                )
                print(f"Successfully generated signed URL")
            except Exception as e:
                print(f"Error generating signed URL: {e}")
                # If signed URL fails, the files require authentication and can't use direct URLs
                return jsonify({'error': 'Unable to generate access URL for document'}), 500
        
        return jsonify({
            'url': url,
            'corrected': path_result['corrected']
        })
        
    except Exception as e:
        print(f"Error generating document URL: {e}")
        return jsonify({'error': str(e)}), 500

@main.route('/fix-storage-paths')
def fix_storage_paths():
    """Debug and fix storage path issues for all documents"""
    from app.firebase_service import search_documents, smart_resolve_storage_path
    
    # Get all documents
    all_documents = search_documents(limit=1000)
    
    results = {
        'total_documents': len(all_documents),
        'fixed_count': 0,
        'broken_count': 0,
        'already_working': 0,
        'details': []
    }
    
    for doc in all_documents:
        if not doc.storage_path:
            continue
            
        doc_info = {
            'id': doc.id,
            'name': doc.name,
            'original_path': doc.storage_path,
            'status': 'unknown'
        }
        
        try:
            # Try smart resolution with fixing enabled
            path_result = smart_resolve_storage_path(doc, fix_in_db=True)
            
            if path_result['resolved_path']:
                if path_result['corrected']:
                    doc_info['status'] = 'fixed'
                    doc_info['new_path'] = path_result['resolved_path']
                    results['fixed_count'] += 1
                else:
                    doc_info['status'] = 'working'
                    results['already_working'] += 1
            else:
                doc_info['status'] = 'broken'
                results['broken_count'] += 1
                
        except Exception as e:
            doc_info['status'] = 'error'
            doc_info['error'] = str(e)
            results['broken_count'] += 1
        
        results['details'].append(doc_info)
    
    return jsonify(results)