import os
import json
import time
from flask import Blueprint, render_template, request, redirect, url_for, jsonify, current_app, flash, session
from app.models import Document
from app.pdf_utils import allowed_file, save_pdf, extract_text_from_pdf
from app.openai_service import extract_learning_goals, DEFAULT_SYSTEM_MESSAGE
from app.firebase_service import upload_pdf_to_storage, save_document, get_document, search_documents, get_learning_goal_suggestions, delete_document

main = Blueprint('main', __name__)

@main.route('/', methods=['GET'])
def index():
    """Render the main upload page"""
    return render_template('index.html', default_instructions=DEFAULT_SYSTEM_MESSAGE)

@main.route('/upload', methods=['POST'])
def upload_file():
    """Handle multiple PDF uploads and processing"""
    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
        
    files = request.files.getlist('file')
    
    # If user does not select file, browser also submits an empty part without filename
    if not files or files[0].filename == '':
        flash('No selected files')
        return redirect(request.url)
        
    # Get custom system message if provided
    custom_system_message = request.form.get('system_message', None)
    
    # Get model selection
    model = request.form.get('model', 'gpt-4o')
    
    # Process up to 10 files
    processed_files = []
    
    for file in files[:10]:  # Limit to first 10 files
        if allowed_file(file.filename, current_app.config['ALLOWED_EXTENSIONS']):
            # Save the uploaded file temporarily
            upload_folder = current_app.config['UPLOAD_FOLDER']
            pdf_path = save_pdf(file, upload_folder)
            
            if not pdf_path:
                continue
                
            # Extract text from PDF
            pdf_text = extract_text_from_pdf(pdf_path)
            
            if not pdf_text:
                os.remove(pdf_path)  # Clean up
                continue
                
            # Extract learning goals using OpenAI
            api_key = current_app.config['OPENAI_API_KEY']
            extraction_result = extract_learning_goals(pdf_text, api_key, custom_system_message, model=model)
            
            # Store file data
            processed_files.append({
                'pdf_path': pdf_path,
                'original_filename': file.filename,
                'learning_goals': extraction_result['learning_goals'],
                'lo_extraction_prompt': extraction_result['system_message_used']
            })
    
    if not processed_files:
        flash('No valid files were processed')
        return redirect(request.url)
        
    # Store data in session for the edit page
    session['processed_files'] = processed_files
    
    return redirect(url_for('main.edit_learning_goals'))

@main.route('/edit', methods=['GET'])
def edit_learning_goals():
    """Render the page for editing learning goals for multiple documents"""
    # Get data from session
    processed_files = session.get('processed_files', [])
    
    if not processed_files:
        flash('No file data found')
        return redirect(url_for('main.index'))
        
    return render_template('edit.html', 
                           processed_files=processed_files)

@main.route('/save', methods=['POST'])
def save_document_data():
    """Save multiple document data to Firebase"""
    # Get form data for global metadata
    global_creator = request.form.get('global_creator', '')
    global_course_name = request.form.get('global_course_name', '')
    global_institution = request.form.get('global_institution', '')
    global_doc_type = request.form.get('global_doc_type', '')
    global_notes = request.form.get('global_notes', '')
    
    # Get processed files from session
    processed_files = session.get('processed_files', [])
    
    if not processed_files:
        return jsonify({'success': False, 'message': 'No file data found'})
    
    results = []
    success_count = 0
    
    for index, file_data in enumerate(processed_files):
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
            
            # Get file path
            pdf_path = file_data['pdf_path']
            original_filename = file_data['original_filename']
            
            # Upload PDF to Google Cloud Storage
            destination_blob_name = f"pdfs/{creator}_{name}_{os.path.basename(pdf_path)}"
            result = upload_pdf_to_storage(pdf_path, destination_blob_name)
            
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
                storage_path=result['storage_path'],
                public_url=result.get('public_url'),
                lo_extraction_prompt=file_data.get('lo_extraction_prompt', '')
            )
            
            # Save to Firestore
            doc_id = save_document(doc)
            
            # Clean up the temporary file
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            
            success_count += 1
            results.append({
                'success': True,
                'document_id': doc_id,
                'name': doc.name,
                'storage_path': doc.storage_path
            })
            
        except Exception as e:
            print(f"Error saving document to Firebase: {e}")
            # Ensure temporary file is cleaned up even on error
            if 'pdf_path' in file_data and os.path.exists(file_data['pdf_path']):
                try:
                    os.remove(file_data['pdf_path'])
                except Exception as del_error:
                    print(f"Could not delete temporary file: {del_error}")
                    
            results.append({
                'success': False,
                'message': f'Error: {str(e)}',
                'name': file_data.get('original_filename', 'Unknown')
            })
    
    # Clear session data
    session.pop('processed_files', None)
    
    # Return aggregated results
    return jsonify({
        'success': success_count > 0,
        'total': len(processed_files),
        'success_count': success_count,
        'results': results,
        'message': f'{success_count} of {len(processed_files)} documents saved successfully'
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