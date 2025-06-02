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
from app.firebase_service import upload_pdf_to_storage, save_document, get_document, search_documents, get_learning_goal_suggestions, delete_document, move_storage_file

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
            final_storage_path = f"pdfs/{creator}_{name}_{original_filename}"
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
        # Get all learning goals from Firebase
        all_documents = search_documents(limit=1000)  # Get all documents
        
        # Extract all learning goals
        all_goals = []
        documents_count = 0
        
        for doc in all_documents:
            if doc.learning_goals:  # Only count documents with learning goals
                documents_count += 1
                all_goals.extend(doc.learning_goals)
        
        return jsonify({
            'success': True,
            'total_goals': len(all_goals),
            'total_documents': documents_count,
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
                        'total_goals': 0
                    }
                
                course_stats[course_name]['documents'] += 1
                course_stats[course_name]['total_goals'] += len(doc.learning_goals)
        
        # Format course list with statistics
        formatted_courses = []
        for course in sorted(courses):
            formatted_courses.append({
                'name': course,
                'documents': course_stats[course]['documents'],
                'total_goals': course_stats[course]['total_goals']
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