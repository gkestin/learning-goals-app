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
    """Handle PDF upload and processing"""
    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
        
    file = request.files['file']
    
    # If user does not select file, browser also submits an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
        
    # Get custom system message if provided
    custom_system_message = request.form.get('system_message', None)
    
    if file and allowed_file(file.filename, current_app.config['ALLOWED_EXTENSIONS']):
        # Save the uploaded file temporarily
        upload_folder = current_app.config['UPLOAD_FOLDER']
        pdf_path = save_pdf(file, upload_folder)
        
        if not pdf_path:
            flash('Error saving file')
            return redirect(request.url)
            
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(pdf_path)
        
        if not pdf_text:
            flash('Could not extract text from PDF')
            os.remove(pdf_path)  # Clean up
            return redirect(request.url)
            
        # Extract learning goals using OpenAI
        api_key = current_app.config['OPENAI_API_KEY']
        learning_goals = extract_learning_goals(pdf_text, api_key, custom_system_message)
        
        # Store data in session for the edit page
        session['pdf_path'] = pdf_path
        session['original_filename'] = file.filename
        session['learning_goals'] = learning_goals
        
        return redirect(url_for('main.edit_learning_goals'))
    
    flash('Invalid file type')
    return redirect(request.url)

@main.route('/edit', methods=['GET'])
def edit_learning_goals():
    """Render the page for editing learning goals"""
    # Get data from session
    pdf_path = session.get('pdf_path')
    original_filename = session.get('original_filename')
    learning_goals = session.get('learning_goals', [])
    
    if not pdf_path or not original_filename:
        flash('No file data found')
        return redirect(url_for('main.index'))
        
    return render_template('edit.html', 
                           filename=original_filename,
                           learning_goals=learning_goals)

@main.route('/save', methods=['POST'])
def save_document_data():
    """Save document data to Firebase"""
    # Get form data
    name = request.form.get('document_name')
    creator = request.form.get('creator')
    course_name = request.form.get('course_name')
    institution = request.form.get('institution')
    doc_type = request.form.get('doc_type')
    notes = request.form.get('notes', '')
    learning_goals = request.form.getlist('learning_goals[]')
    
    # Get file path from session
    pdf_path = session.get('pdf_path')
    original_filename = session.get('original_filename')
    
    if not pdf_path or not original_filename:
        return jsonify({'success': False, 'message': 'No file data found'})
    
    try:
        # Upload PDF to Google Cloud Storage
        destination_blob_name = f"pdfs/{creator}_{name}_{os.path.basename(pdf_path)}"
        print(f"Uploading PDF to Firebase Storage: {destination_blob_name}")
        result = upload_pdf_to_storage(pdf_path, destination_blob_name)
        
        print(f"PDF uploaded, storage path: {result['storage_path']}")
        print(f"Public URL: {result.get('public_url', 'Not available')}")
        
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
        print("Saving document metadata to Firestore...")
        doc_id = save_document(doc)
        print(f"Document saved with ID: {doc_id}")
        
        # Clean up the temporary file
        if os.path.exists(pdf_path):
            print(f"Deleting local temporary file: {pdf_path}")
            os.remove(pdf_path)
            print("Local file deleted successfully")
        else:
            print(f"Warning: Local file {pdf_path} not found for deletion")
        
        # Clear session data
        session.pop('pdf_path', None)
        session.pop('original_filename', None)
        session.pop('learning_goals', None)
        
        return jsonify({
            'success': True, 
            'document_id': doc_id,
            'message': 'Document saved to Firebase successfully',
            'storage_path': result['storage_path']
        })
    
    except Exception as e:
        print(f"Error saving document to Firebase: {e}")
        # Ensure temporary file is cleaned up even on error
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
                print(f"Deleted temporary file {pdf_path} after error")
            except Exception as del_error:
                print(f"Could not delete temporary file: {del_error}")
                
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

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