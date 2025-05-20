import os
import json
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
            learning_goals = extract_learning_goals(pdf_text, api_key, custom_system_message)
            
            # Store file data
            processed_files.append({
                'pdf_path': pdf_path,
                'original_filename': file.filename,
                'learning_goals': learning_goals
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
                public_url=result.get('public_url')
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
    if not query:
        print("Searching for all documents in Firestore...")
        results = search_documents(limit=20)
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