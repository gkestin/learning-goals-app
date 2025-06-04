import os
import datetime
import firebase_admin
from firebase_admin import credentials, firestore, storage
from app.models import Document
import datetime as datetime_module  # For timedelta
import uuid
import re

# Global variables
db = None
bucket = None
using_mock = False

def init_app(app):
    """Initialize Firebase and Firestore connections"""
    global db, bucket, using_mock
    
    try:
        # Check if we have the required configurations
        cred_path = app.config.get('GOOGLE_APPLICATION_CREDENTIALS')
        bucket_name = app.config.get('FIREBASE_STORAGE_BUCKET')
        
        print(f"Firebase configuration from app.config:")
        print(f"- Credentials path: {cred_path}")
        print(f"- Storage bucket: {bucket_name}")
        
        # Check if we're reading different values from environment directly
        env_bucket = os.environ.get('FIREBASE_STORAGE_BUCKET')
        if env_bucket != bucket_name:
            print(f"WARNING: Environment has different bucket ({env_bucket}) than app.config ({bucket_name})")
        
        if not cred_path or not os.path.exists(cred_path):
            print(f"Firebase credentials file not found at: {cred_path}")
            print("IMPORTANT: Initializing with mock services - NOT SAVING TO REAL FIREBASE!")
            # Initialize mock services for development
            init_mock_services()
            return
            
        if not bucket_name:
            print("FIREBASE_STORAGE_BUCKET not configured")
            print("IMPORTANT: Initializing with mock services - NOT SAVING TO REAL FIREBASE!")
            # Initialize mock services for development 
            init_mock_services()
            return
        
        # Validate and normalize bucket name
        # Remove any gs:// prefix if present
        if bucket_name.startswith('gs://'):
            bucket_name = bucket_name[5:]
            print(f"Removed gs:// prefix, bucket name now: {bucket_name}")
            
        # Ensure bucket name is in proper format (project-id.firebasestorage.app)  
        if bucket_name.endswith('.appspot.com'):
            project_id = bucket_name.split('.')[0]
            bucket_name = f"{project_id}.firebasestorage.app"
            print(f"Converted bucket name from appspot.com to correct Storage format: {bucket_name}")
        elif not bucket_name.endswith('.firebasestorage.app'):
            # If it doesn't have either domain extension, assume it's just the project ID
            if '.' not in bucket_name:
                bucket_name = f"{bucket_name}.firebasestorage.app"
                print(f"Added .firebasestorage.app domain to bucket name: {bucket_name}")
            
        print(f"Final bucket name to use: {bucket_name}")    
        
        # Initialize Firebase if not already initialized
        if not firebase_admin._apps:
            try:
                print(f"Initializing Firebase with bucket: '{bucket_name}'")
                cred = credentials.Certificate(cred_path)
                
                # No need to convert bucket name here since we already handled it above
                
                firebase_admin.initialize_app(cred, {
                    'storageBucket': bucket_name
                })
                
                db = firestore.client()
                bucket = storage.bucket()
                print(f"Bucket name after initialization: {bucket.name if bucket else 'None'}")
                using_mock = False
                print("✅ Firebase initialized successfully - USING REAL FIREBASE!")
            except Exception as e:
                print(f"Firebase initialization error: {e}")
                print("IMPORTANT: Falling back to mock services - NOT SAVING TO REAL FIREBASE!")
                # Initialize mock services as fallback
                init_mock_services()
        else:
            db = firestore.client()
            bucket = storage.bucket()
            print(f"Bucket name from existing connection: {bucket.name if bucket else 'None'}")
            using_mock = False
            print("✅ Firebase connection established - USING REAL FIREBASE!")
    except Exception as e:
        print(f"Error in Firebase init_app: {e}")
        print("IMPORTANT: Falling back to mock services - NOT SAVING TO REAL FIREBASE!")
        # Initialize mock services as fallback
        init_mock_services()

def init_mock_services():
    """Initialize mock services for development/testing"""
    global db, bucket, using_mock
    
    using_mock = True
    print("⚠️ Using MOCK Firebase services - data will NOT be saved to real Firebase! ⚠️")
    
    # Simple mock implementation for Firestore
    class MockFirestore:
        def __init__(self):
            self.collections = {'documents': MockCollection('documents')}
            print("  - Mock Firestore database initialized")
            
        def collection(self, name):
            print(f"  - Accessing mock collection: {name}")
            if name not in self.collections:
                self.collections[name] = MockCollection(name)
            return self.collections[name]
    
    class MockCollection:
        def __init__(self, name):
            self.name = name
            self.documents = {}
            print(f"  - Mock collection created: {name}")
            
        def document(self, doc_id=None):
            if not doc_id:
                doc_id = f"mock_{len(self.documents) + 1}"
                print(f"  - Creating new mock document with ID: {doc_id}")
            else:
                print(f"  - Accessing mock document: {doc_id}")
            return MockDocument(doc_id, self)
            
        def stream(self):
            print(f"  - Streaming all documents from mock collection: {self.name}")
            return [doc for doc in self.documents.values()]
            
        def limit(self, n):
            print(f"  - Limiting results to {n} in mock collection")
            return self
    
    class MockDocument:
        def __init__(self, id, collection):
            self.id = id
            self.collection = collection
            self.data = {}
            self.exists = False
            
        def set(self, data):
            print(f"  - Setting data for mock document: {self.id}")
            self.data = data
            self.collection.documents[self.id] = self
            self.exists = True
            return self.id
            
        def get(self):
            print(f"  - Getting mock document: {self.id}")
            return self
            
        def to_dict(self):
            return self.data
    
    class MockStorage:
        def __init__(self):
            self.files = {}
            print("  - Mock Storage initialized")
            
        def blob(self, path):
            print(f"  - Creating mock blob: {path}")
            return MockBlob(path, self)
    
    class MockBlob:
        def __init__(self, path, storage):
            self.path = path
            self.storage = storage
            # Mock public URL using Firebase Storage URL format
            self.public_url = f"https://firebasestorage.googleapis.com/v0/b/mock-project.firebasestorage.app/o/{path.replace('/', '%2F')}?alt=media"
            
        def upload_from_filename(self, file_path):
            print(f"  - Mock uploading file from {file_path} to {self.path}")
            self.storage.files[self.path] = file_path
            return True
            
        def make_public(self):
            print(f"  - Making mock blob public: {self.path}")
            return True
    
    # Initialize mock services
    db = MockFirestore()
    bucket = MockStorage()
    print("⚠️ Mock Firebase services initialized - your data is only stored in memory! ⚠️")

def upload_pdf_to_storage(file_path, destination_blob_name=None):
    """Upload a PDF file to Google Cloud Storage and return the public URL"""
    global bucket, using_mock
    
    if not bucket:
        print("Bucket not initialized, initializing mock services.")
        init_mock_services()
    
    if using_mock:
        print(f"⚠️ MOCK STORAGE: Pretending to upload {file_path} to Firebase Storage")
    else:
        print(f"✅ REAL STORAGE: Uploading {file_path} to Firebase Storage")
        print(f"   Bucket details: Name={bucket.name}, Path={bucket.path}")
        
    if not destination_blob_name:
        destination_blob_name = f"pdfs/{os.path.basename(file_path)}"
    
    print(f"Creating blob: {destination_blob_name}")
    blob = bucket.blob(destination_blob_name)
    
    print(f"Starting upload: {file_path} -> {destination_blob_name}")
    try:
        blob.upload_from_filename(file_path)
        print(f"Upload completed successfully")
        
        # Explicitly make blob public with better error handling
        print(f"Making blob public: {destination_blob_name}")
        try:
            # Make the blob publicly accessible
            blob.make_public()
            print(f"Successfully made blob public")
        except Exception as e:
            print(f"Error making blob public: {e}")
            print("Will attempt to generate signed URL instead")
    except Exception as e:
        print(f"Error during upload: {e}")
        print(f"Bucket: {bucket.name if hasattr(bucket, 'name') else 'Unknown'}")
        raise
    
    # Use the Firebase Storage standard URL format with token if needed
    storage_path = destination_blob_name
    if not using_mock:
        bucket_name = bucket.name
        encoded_path = storage_path.replace('/', '%2F')
        
        # First try to get a public URL
        try:
            public_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o/{encoded_path}?alt=media"
            # Try to create a signed URL with longer expiration (1 week)
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=datetime_module.timedelta(weeks=1),
                method="GET"
            )
            print(f"Generated signed URL: {signed_url}")
            # If successful, use the signed URL
            public_url = signed_url
        except Exception as e:
            print(f"Error generating signed URL: {e}")
            # Fall back to regular URL
            public_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o/{encoded_path}?alt=media"
            
        print(f"Generated Firebase Storage URL: {public_url}")
    else:
        public_url = blob.public_url
    
    result = {
        'storage_path': storage_path,
        'public_url': public_url
    }
    
    if using_mock:
        print(f"⚠️ MOCK STORAGE: PDF 'uploaded' to mock storage at {destination_blob_name}")
    else:
        print(f"✅ REAL STORAGE: PDF uploaded to Firebase Storage at {destination_blob_name}")
        print(f"   Public URL: {public_url}")
        
    return result

def save_document(document):
    """Save document metadata to Firestore"""
    global db, using_mock
    
    if not db:
        init_mock_services()
    
    if using_mock:
        print(f"⚠️ MOCK DATABASE: Saving document '{document.name}' to mock Firestore")
    else:
        print(f"✅ REAL DATABASE: Saving document '{document.name}' to Firestore")
        
    doc_ref = db.collection('documents').document()
    doc_data = document.to_dict()
    
    # Convert datetime to Firebase timestamp
    if isinstance(doc_data['created_at'], datetime.datetime):
        # For real Firestore, use SERVER_TIMESTAMP
        if hasattr(firestore, 'SERVER_TIMESTAMP') and not using_mock:
            doc_data['created_at'] = firestore.SERVER_TIMESTAMP
    
    doc_id = doc_ref.set(doc_data)
    
    if using_mock:
        print(f"⚠️ MOCK DATABASE: Document saved with mock ID: {doc_ref.id}")
    else:
        print(f"✅ REAL DATABASE: Document saved to Firestore with ID: {doc_ref.id}")
        
    return doc_ref.id

def get_document(doc_id):
    """Retrieve a document by its ID"""
    global db, using_mock
    
    if not db:
        init_mock_services()
    
    if using_mock:
        print(f"⚠️ MOCK DATABASE: Retrieving document with ID '{doc_id}' from mock Firestore")
    else:
        print(f"✅ REAL DATABASE: Retrieving document with ID '{doc_id}' from Firestore")
        
    doc_ref = db.collection('documents').document(doc_id)
    doc = doc_ref.get()
    
    if doc.exists:
        if using_mock:
            print(f"⚠️ MOCK DATABASE: Found mock document: {doc_id}")
        else:
            print(f"✅ REAL DATABASE: Found document in Firestore: {doc_id}")
        return Document.from_dict(doc.to_dict(), doc_id=doc.id)
    
    if using_mock:
        print(f"⚠️ MOCK DATABASE: Document not found in mock database: {doc_id}")
    else:
        print(f"✅ REAL DATABASE: Document not found in Firestore: {doc_id}")
    return None

def search_documents(search_terms=None, limit=1000):
    """Search for documents by learning goals"""
    global db, using_mock
    
    if not db:
        init_mock_services()
    
    if using_mock:
        print(f"⚠️ MOCK DATABASE: Searching for documents in mock Firestore")
        if search_terms:
            print(f"⚠️ MOCK DATABASE: Search terms: {search_terms}")
    else:
        print(f"✅ REAL DATABASE: Searching for documents in Firestore")
        if search_terms:
            print(f"✅ REAL DATABASE: Search terms: {search_terms}")
        
    query = db.collection('documents')
    
    # Basic search by iterating through documents
    docs = query.limit(limit).stream()
    results = []
    
    for doc in docs:
        doc_data = doc.to_dict()
        learning_goals = doc_data.get('learning_goals', [])
        
        # If no search terms, include all docs
        if not search_terms:
            results.append(Document.from_dict(doc_data, doc_id=doc.id))
            continue
            
        # Check if any learning goal contains any of the search terms
        for goal in learning_goals:
            if any(term.lower() in goal.lower() for term in search_terms):
                results.append(Document.from_dict(doc_data, doc_id=doc.id))
                break
    
    if using_mock:
        print(f"⚠️ MOCK DATABASE: Found {len(results)} documents in mock database")
    else:
        print(f"✅ REAL DATABASE: Found {len(results)} documents in Firestore")
        
    return results

def get_learning_goal_suggestions(prefix, limit=10):
    """Get autocomplete suggestions for learning goals based on prefix"""
    global db, using_mock
    
    if not db:
        init_mock_services()
    
    if using_mock:
        print(f"⚠️ MOCK DATABASE: Getting learning goal suggestions for '{prefix}' from mock Firestore")
    else:
        print(f"✅ REAL DATABASE: Getting learning goal suggestions for '{prefix}' from Firestore")
        
    if not prefix or len(prefix) < 3:
        return []
        
    # Get all documents
    docs = db.collection('documents').stream()
    all_goals = set()
    
    # Extract all learning goals
    for doc in docs:
        goals = doc.to_dict().get('learning_goals', [])
        all_goals.update(goals)
    
    # Filter goals that contain the prefix (case insensitive)
    prefix = prefix.lower()
    matching_goals = [goal for goal in all_goals if prefix in goal.lower()]
    
    # Sort by relevance (starting with prefix is more relevant)
    sorted_goals = sorted(
        matching_goals,
        key=lambda g: (0 if g.lower().startswith(prefix) else 1, len(g))
    )
    
    suggestions = sorted_goals[:limit]
    
    if using_mock:
        print(f"⚠️ MOCK DATABASE: Found {len(suggestions)} suggestions in mock database")
    else:
        print(f"✅ REAL DATABASE: Found {len(suggestions)} suggestions in Firestore")
        
    return suggestions

def delete_document(doc_id):
    """Delete a document and its associated file from Firebase with smart path correction"""
    global db, bucket, using_mock
    
    if not db:
        init_mock_services()
    
    # First, get the document to retrieve the storage path
    doc = get_document(doc_id)
    if not doc:
        print(f"Document not found for deletion: {doc_id}")
        return False
        
    try:
        # 1. Delete the file from Storage if it exists
        file_deleted = False
        
        if doc.storage_path:
            if using_mock:
                print(f"⚠️ MOCK STORAGE: Pretending to delete file: {doc.storage_path}")
                if doc.storage_path in bucket.files:
                    del bucket.files[doc.storage_path]
                    file_deleted = True
            else:
                # Use smart resolution to find the correct path
                path_result = smart_resolve_storage_path(doc, fix_in_db=True)
                
                if path_result['resolved_path']:
                    print(f"✅ REAL STORAGE: Deleting file: {path_result['resolved_path']}")
                    
                    try:
                        blob = bucket.blob(path_result['resolved_path'])
                        blob.delete()
                        print(f"✅ File deleted successfully: {path_result['resolved_path']}")
                        file_deleted = True
                        
                        if path_result['corrected']:
                            print(f"📝 Note: Storage path was auto-corrected from duplicated filename")
                            
                    except Exception as e:
                        print(f"❌ Failed to delete file: {e}")
                else:
                    print(f"❌ Could not resolve storage path for deletion")
                        
                if not file_deleted:
                    print(f"⚠️ Warning: Could not delete file from storage, but continuing with document deletion")
        
        # 2. Delete the document from Firestore
        if using_mock:
            print(f"⚠️ MOCK DATABASE: Deleting document from mock Firestore: {doc_id}")
            if doc_id in db.collections['documents'].documents:
                del db.collections['documents'].documents[doc_id]
        else:
            print(f"✅ REAL DATABASE: Deleting document from Firestore: {doc_id}")
            db.collection('documents').document(doc_id).delete()
        
        print(f"✅ Document deleted successfully: {doc_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error deleting document: {e}")
        return False

def generate_signed_upload_url(filename, content_type='application/pdf'):
    """Generate a signed URL for direct upload to Cloud Storage"""
    global bucket, using_mock
    
    if not bucket:
        print("Bucket not initialized, initializing mock services.")
        init_mock_services()
    
    # Create a unique storage path
    unique_id = str(uuid.uuid4())[:8]
    storage_path = f"temp_uploads/{unique_id}_{filename}"
    
    if using_mock:
        print(f"⚠️ MOCK STORAGE: Generating mock signed URL for {storage_path}")
        return {
            'upload_url': f"https://mock-storage.example.com/upload/{storage_path}",
            'download_url': f"https://mock-storage.example.com/download/{storage_path}",
            'storage_path': storage_path
        }
    
    print(f"✅ REAL STORAGE: Generating signed upload URL for {storage_path}")
    
    blob = bucket.blob(storage_path)
    
    # Generate signed URL for upload (valid for 1 hour)
    upload_url = blob.generate_signed_url(
        version="v4",
        expiration=datetime_module.timedelta(hours=1),
        method="PUT",
        content_type=content_type
    )
    
    # Generate signed URL for download (valid for 1 hour)
    download_url = blob.generate_signed_url(
        version="v4",
        expiration=datetime_module.timedelta(hours=1),
        method="GET"
    )
    
    print(f"Generated signed upload URL for: {storage_path}")
    
    return {
        'upload_url': upload_url,
        'download_url': download_url,
        'storage_path': storage_path
    }

def download_from_storage(storage_path, local_path):
    """Download a file from Cloud Storage to local path"""
    global bucket, using_mock
    
    if not bucket:
        print("Bucket not initialized, initializing mock services.")
        init_mock_services()
    
    if using_mock:
        print(f"⚠️ MOCK STORAGE: Pretending to download {storage_path} to {local_path}")
        # Create a mock file for testing
        with open(local_path, 'w') as f:
            f.write("Mock PDF content")
        return
    
    print(f"✅ REAL STORAGE: Downloading {storage_path} to {local_path}")
    
    blob = bucket.blob(storage_path)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Download the file
    blob.download_to_filename(local_path)
    
    print(f"Successfully downloaded {storage_path} to {local_path}")

def move_storage_file(source_path, destination_path):
    """Move a file from one location to another in Cloud Storage"""
    global bucket, using_mock
    
    if not bucket:
        print("Bucket not initialized, initializing mock services.")
        init_mock_services()
    
    if using_mock:
        print(f"⚠️ MOCK STORAGE: Pretending to move {source_path} to {destination_path}")
        return {
            'storage_path': destination_path,
            'public_url': f"https://mock-storage.example.com/{destination_path}"
        }
    
    print(f"✅ REAL STORAGE: Moving {source_path} to {destination_path}")
    
    source_blob = bucket.blob(source_path)
    destination_blob = bucket.blob(destination_path)
    
    # Copy the blob to the new location
    destination_blob.rewrite(source_blob)
    
    # Delete the source blob
    source_blob.delete()
    
    # Generate public URL for the destination
    try:
        # Try to create a signed URL with longer expiration (1 week)
        signed_url = destination_blob.generate_signed_url(
            version="v4",
            expiration=datetime_module.timedelta(weeks=1),
            method="GET"
        )
        public_url = signed_url
    except Exception as e:
        print(f"Error generating signed URL: {e}")
        # Fall back to regular URL
        bucket_name = bucket.name
        encoded_path = destination_path.replace('/', '%2F')
        public_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o/{encoded_path}?alt=media"
    
    print(f"Successfully moved {source_path} to {destination_path}")
    
    return {
        'storage_path': destination_path,
        'public_url': public_url
    }

def smart_resolve_storage_path(doc, fix_in_db=False):
    """
    Smart resolution for storage paths that may have issues with duplicated filenames.
    
    Returns:
        dict: {
            'resolved_path': corrected_path or None,
            'corrected': boolean indicating if correction was made,
            'exists': boolean indicating if file exists at resolved path
        }
    """
    global bucket, using_mock
    
    if not bucket:
        init_mock_services()
    
    original_path = doc.storage_path
    if not original_path:
        return {'resolved_path': None, 'corrected': False, 'exists': False}
    
    # First, try the original path
    if using_mock:
        exists = original_path in bucket.files
        if exists:
            return {'resolved_path': original_path, 'corrected': False, 'exists': True}
    else:
        try:
            blob = bucket.blob(original_path)
            if blob.exists():
                return {'resolved_path': original_path, 'corrected': False, 'exists': True}
        except Exception as e:
            print(f"Error checking original path {original_path}: {e}")
    
    # If original path doesn't work, try to resolve common issues
    
    # Issue 1: Duplicated filename in path (e.g., "pdfs/example.pdf/example.pdf")
    path_parts = original_path.split('/')
    if len(path_parts) >= 2:
        filename = path_parts[-1]
        if filename in path_parts[:-1]:
            # Remove the duplicated filename from the middle
            corrected_parts = []
            for part in path_parts[:-1]:
                if part != filename:
                    corrected_parts.append(part)
            corrected_parts.append(filename)
            corrected_path = '/'.join(corrected_parts)
            
            # Test this corrected path
            if using_mock:
                if corrected_path in bucket.files:
                    if fix_in_db:
                        # Update the document in the database
                        try:
                            doc_ref = db.collection('documents').document(doc.id)
                            doc_ref.update({'storage_path': corrected_path})
                            print(f"✅ Updated storage path in DB: {original_path} -> {corrected_path}")
                        except Exception as e:
                            print(f"Failed to update storage path in DB: {e}")
                    
                    return {'resolved_path': corrected_path, 'corrected': True, 'exists': True}
            else:
                try:
                    blob = bucket.blob(corrected_path)
                    if blob.exists():
                        if fix_in_db:
                            # Update the document in the database
                            try:
                                doc_ref = db.collection('documents').document(doc.id)
                                doc_ref.update({'storage_path': corrected_path})
                                print(f"✅ Updated storage path in DB: {original_path} -> {corrected_path}")
                            except Exception as e:
                                print(f"Failed to update storage path in DB: {e}")
                        
                        return {'resolved_path': corrected_path, 'corrected': True, 'exists': True}
                except Exception as e:
                    print(f"Error checking corrected path {corrected_path}: {e}")
    
    # If no resolution worked, return None
    return {'resolved_path': None, 'corrected': False, 'exists': False}


# Hierarchy Management Functions
def save_hierarchy(hierarchy):
    """Save a learning goals hierarchy to Firestore with proper serialization"""
    global db, using_mock
    
    if not db:
        init_mock_services()
    
    if using_mock:
        print(f"⚠️ MOCK DATABASE: Saving hierarchy '{hierarchy.name}' to mock Firestore")
    else:
        print(f"✅ REAL DATABASE: Saving hierarchy '{hierarchy.name}' to Firestore")
    
    # Create the main hierarchy document
    doc_ref = db.collection('learning_goals_hierarchies').document()
    
    # Serialize the hierarchy structure for Firestore
    serialized_hierarchy = serialize_hierarchy_for_firestore(hierarchy)
    doc_data = serialized_hierarchy.to_dict()
    
    # Convert datetime to Firebase timestamp
    if isinstance(doc_data['created_at'], datetime.datetime):
        if hasattr(firestore, 'SERVER_TIMESTAMP') and not using_mock:
            doc_data['created_at'] = firestore.SERVER_TIMESTAMP
            doc_data['modified_at'] = firestore.SERVER_TIMESTAMP
    
    try:
        # Save the hierarchy document
        doc_id = doc_ref.set(doc_data)
        
        if using_mock:
            print(f"⚠️ MOCK DATABASE: Hierarchy saved with mock ID: {doc_ref.id}")
        else:
            print(f"✅ REAL DATABASE: Hierarchy saved to Firestore with ID: {doc_ref.id}")
        
        return doc_ref.id
        
    except Exception as e:
        print(f"❌ Error saving hierarchy: {e}")
        # If still too large, try aggressive flattening
        if "exceeds the maximum allowed size" in str(e) or "invalid nested entity" in str(e):
            print("📝 Applying aggressive flattening for Firestore compatibility...")
            return save_hierarchy_with_aggressive_flattening(hierarchy, doc_ref)
        raise


def serialize_hierarchy_for_firestore(hierarchy):
    """Serialize hierarchy structure to be Firestore-compatible"""
    from app.models import LearningGoalsHierarchy
    
    # Flatten the nested structure to avoid Firestore nesting limits
    flattened_nodes = flatten_hierarchy_nodes(hierarchy.root_nodes)
    
    # Create a serializable hierarchy
    serialized_hierarchy = LearningGoalsHierarchy(
        id=hierarchy.id,
        name=hierarchy.name,
        description=hierarchy.description,
        creator=hierarchy.creator,
        course_name=hierarchy.course_name,
        institution=hierarchy.institution,
        root_nodes=flattened_nodes,
        metadata=hierarchy.metadata,
        created_at=hierarchy.created_at,
        modified_at=hierarchy.modified_at,
        is_active=hierarchy.is_active
    )
    
    return serialized_hierarchy


def flatten_hierarchy_nodes(nodes):
    """Flatten hierarchy nodes to avoid nested object issues in Firestore"""
    flattened = []
    
    for node in nodes:
        # Create a flattened version of the node
        flattened_node = {
            'id': node.get('id'),
            'label': node.get('label', ''),
            'originalLabel': node.get('originalLabel', ''),
            'goals': node.get('goals', []),
            'levels': node.get('levels', {}),
            'signature': node.get('signature', []),
            'isExpanded': node.get('isExpanded', False),
            'csvLevel': node.get('csvLevel', 0),
            'displayLevel': node.get('displayLevel', 0),
            'size': node.get('size', 0),
            'modified': node.get('modified', False),
            'childIds': []  # Store child IDs instead of nested objects
        }
        
        # Extract child IDs and flatten children separately
        children = node.get('children', [])
        if children:
            flattened_node['childIds'] = [child.get('id') for child in children]
            # Recursively flatten children and add them to the main list
            flattened_children = flatten_hierarchy_nodes(children)
            flattened.extend(flattened_children)
        
        flattened.append(flattened_node)
    
    return flattened


def save_hierarchy_with_aggressive_flattening(hierarchy, doc_ref):
    """Save hierarchy with maximum flattening for large datasets"""
    from app.models import LearningGoalsHierarchy
    
    print("🔧 Applying maximum flattening for large hierarchy...")
    
    # Calculate summary statistics
    total_goals = hierarchy.get_total_goals()
    total_groups = hierarchy.get_total_groups()
    
    # Create minimal metadata
    minimal_metadata = {
        'total_goals': total_goals,
        'total_groups': total_groups,
        'optimization_level': 'maximum_flattening',
        'original_upload_time': hierarchy.metadata.get('uploaded_at', datetime.datetime.now().isoformat())
    }
    
    # Create structure summary with minimal data
    structure_summary = create_minimal_hierarchy_summary(hierarchy.root_nodes)
    
    minimal_hierarchy = LearningGoalsHierarchy(
        name=hierarchy.name,
        description=f"{hierarchy.description} (Maximum Flattening)",
        creator=hierarchy.creator,
        course_name=hierarchy.course_name,
        institution=hierarchy.institution,
        root_nodes=structure_summary,
        metadata=minimal_metadata,
        created_at=hierarchy.created_at,
        modified_at=hierarchy.modified_at,
        is_active=hierarchy.is_active
    )
    
    # Save minimal version
    doc_data = minimal_hierarchy.to_dict()
    
    if isinstance(doc_data['created_at'], datetime.datetime):
        if hasattr(firestore, 'SERVER_TIMESTAMP') and not using_mock:
            doc_data['created_at'] = firestore.SERVER_TIMESTAMP
            doc_data['modified_at'] = firestore.SERVER_TIMESTAMP
    
    doc_id = doc_ref.set(doc_data)
    
    # Save complete data in subcollection with chunking
    save_complete_hierarchy_data_chunked(doc_ref.id, hierarchy.root_nodes)
    
    print(f"✅ Hierarchy saved with maximum flattening: {doc_ref.id}")
    return doc_ref.id


def create_minimal_hierarchy_summary(nodes):
    """Create an ultra-minimal summary of hierarchy structure"""
    if not nodes:
        return []
    
    summary = []
    
    for i, node in enumerate(nodes):
        if isinstance(node, dict):
            # Handle children safely - check for None explicitly
            children = node.get('children')
            if children is None:
                children = []
            
            # Handle goals safely - check for None explicitly  
            goals = node.get('goals')
            if goals is None:
                goals = []
            
            node_summary = {
                'id': node.get('id', f'node_{i}'),
                'label': node.get('label', f'Group_{i+1}'),
                'displayLevel': node.get('displayLevel', 0),
                'size': node.get('size', 0),
                'hasChildren': len(children) > 0,
                'childCount': len(children),
                'goalCount': len(goals)
            }
        else:
            # Handle case where node might not be a dict
            node_summary = {
                'id': f'node_{i}',
                'label': f'Group_{i+1}',
                'displayLevel': 0,
                'size': 0,
                'hasChildren': False,
                'childCount': 0,
                'goalCount': 0
            }
        
        summary.append(node_summary)
    
    return summary


def save_complete_hierarchy_data_chunked(hierarchy_id, nodes):
    """Save complete hierarchy data in chunked subcollections"""
    global db, using_mock
    
    if using_mock:
        return
    
    print(f"💾 Saving complete hierarchy data in chunks...")
    
    # Flatten all nodes first
    flattened_nodes = flatten_hierarchy_nodes(nodes)
    
    # Save in chunks to avoid size limits
    chunk_size = 1000  # Start with larger chunks, split if too big
    chunks = [flattened_nodes[i:i + chunk_size] for i in range(0, len(flattened_nodes), chunk_size)]
    
    def save_chunk_with_retry(chunk_data, chunk_ref, chunk_id, attempt=1):
        """Save a chunk with retry logic that splits if too big"""
        try:
            chunk_ref.set(chunk_data)
            print(f"  📦 Saved chunk {chunk_id} with {len(chunk_data['nodes'])} nodes (attempt {attempt})")
            return True
        except Exception as e:
            if ("exceeds the maximum allowed size" in str(e) or "invalid nested entity" in str(e)) and len(chunk_data['nodes']) > 1:
                # Split chunk in half and try again
                nodes = chunk_data['nodes']
                mid = len(nodes) // 2
                left_nodes = nodes[:mid]
                right_nodes = nodes[mid:]
                
                print(f"  🔄 Chunk {chunk_id} too big ({len(nodes)} nodes), splitting into {len(left_nodes)} + {len(right_nodes)} nodes")
                
                # Save left half
                left_chunk_data = {
                    'chunk_id': f"{chunk_id}a",
                    'nodes': left_nodes,
                    'chunk_size': len(left_nodes),
                    'created_at': firestore.SERVER_TIMESTAMP
                }
                left_chunk_ref = db.collection('learning_goals_hierarchies').document(hierarchy_id).collection('node_chunks').document(f'chunk_{chunk_id}a')
                save_chunk_with_retry(left_chunk_data, left_chunk_ref, f"{chunk_id}a", attempt + 1)
                
                # Save right half
                right_chunk_data = {
                    'chunk_id': f"{chunk_id}b",
                    'nodes': right_nodes,
                    'chunk_size': len(right_nodes),
                    'created_at': firestore.SERVER_TIMESTAMP
                }
                right_chunk_ref = db.collection('learning_goals_hierarchies').document(hierarchy_id).collection('node_chunks').document(f'chunk_{chunk_id}b')
                save_chunk_with_retry(right_chunk_data, right_chunk_ref, f"{chunk_id}b", attempt + 1)
                
                return True
            else:
                print(f"  ❌ Failed to save chunk {chunk_id}: {e}")
                return False
    
    total_saved = 0
    for i, chunk in enumerate(chunks):
        chunk_data = {
            'chunk_id': i,
            'nodes': chunk,
            'chunk_size': len(chunk),
            'created_at': firestore.SERVER_TIMESTAMP
        }
        
        chunk_ref = db.collection('learning_goals_hierarchies').document(hierarchy_id).collection('node_chunks').document(f'chunk_{i}')
        if save_chunk_with_retry(chunk_data, chunk_ref, i):
            total_saved += len(chunk)
    
    print(f"✅ Complete hierarchy data saved with {total_saved} total nodes")

def get_hierarchy(hierarchy_id):
    """Retrieve a specific hierarchy by its ID and reconstruct nested structure"""
    global db, using_mock
    
    if not db:
        init_mock_services()
    
    if using_mock:
        print(f"⚠️ MOCK DATABASE: Retrieving hierarchy with ID '{hierarchy_id}' from mock Firestore")
    else:
        print(f"✅ REAL DATABASE: Retrieving hierarchy with ID '{hierarchy_id}' from Firestore")
        
    doc_ref = db.collection('learning_goals_hierarchies').document(hierarchy_id)
    doc = doc_ref.get()
    
    if doc.exists:
        from app.models import LearningGoalsHierarchy
        doc_data = doc.to_dict()
        
        # Check if this is a flattened hierarchy that needs reconstruction
        if needs_hierarchy_reconstruction(doc_data):
            doc_data = reconstruct_hierarchy_structure(hierarchy_id, doc_data)
        
        if using_mock:
            print(f"⚠️ MOCK DATABASE: Found mock hierarchy: {hierarchy_id}")
        else:
            print(f"✅ REAL DATABASE: Found hierarchy in Firestore: {hierarchy_id}")
        return LearningGoalsHierarchy.from_dict(doc_data, hierarchy_id=doc.id)
    
    if using_mock:
        print(f"⚠️ MOCK DATABASE: Hierarchy not found in mock database: {hierarchy_id}")
    else:
        print(f"✅ REAL DATABASE: Hierarchy not found in Firestore: {hierarchy_id}")
    return None


def needs_hierarchy_reconstruction(doc_data):
    """Check if hierarchy document needs structure reconstruction"""
    root_nodes = doc_data.get('root_nodes', [])
    
    # Check if this is a flattened structure (nodes have childIds instead of children)
    for node in root_nodes:
        if isinstance(node, dict):
            if 'childIds' in node or ('hasChildren' in node and 'children' not in node):
                return True
            # Check if it's missing nested structure - handle None goals safely
            if 'children' not in node:
                goals = node.get('goals')
                if goals is None:
                    goals = []
                if node.get('size', 0) > len(goals):
                    return True
    
    return False


def reconstruct_hierarchy_structure(hierarchy_id, doc_data):
    """Reconstruct the nested hierarchy structure from flattened data"""
    global db, using_mock
    
    metadata = doc_data.get('metadata', {})
    optimization_level = metadata.get('optimization_level', '')
    
    if optimization_level == 'maximum_flattening':
        print(f"🔧 Reconstructing maximum flattened hierarchy...")
        return reconstruct_from_chunked_data(hierarchy_id, doc_data)
    else:
        print(f"🔧 Reconstructing flattened hierarchy structure...")
        return reconstruct_from_flattened_nodes(doc_data)


def reconstruct_from_flattened_nodes(doc_data):
    """Reconstruct hierarchy from flattened node structure"""
    root_nodes = doc_data.get('root_nodes', [])
    
    if not root_nodes:
        return doc_data
    
    # Check if nodes have childIds (flattened structure)
    if not any('childIds' in node for node in root_nodes if isinstance(node, dict)):
        # Already in correct format
        return doc_data
    
    # Create a map of all nodes by ID
    node_map = {}
    for node in root_nodes:
        if isinstance(node, dict) and 'id' in node:
            node_map[node['id']] = node.copy()
    
    # Reconstruct nested structure
    reconstructed_roots = []
    
    for node in root_nodes:
        if isinstance(node, dict):
            # Find root nodes (nodes that aren't children of other nodes)
            is_root = True
            for other_node in root_nodes:
                if isinstance(other_node, dict) and 'childIds' in other_node:
                    if node['id'] in other_node['childIds']:
                        is_root = False
                        break
            
            if is_root:
                reconstructed_node = reconstruct_node_tree(node['id'], node_map)
                if reconstructed_node:
                    reconstructed_roots.append(reconstructed_node)
    
    # Update the document data with reconstructed structure
    doc_data['root_nodes'] = reconstructed_roots
    
    print(f"✅ Hierarchy structure reconstructed with {len(reconstructed_roots)} root nodes")
    return doc_data


def reconstruct_node_tree(node_id, node_map):
    """Recursively reconstruct a node tree from flattened data"""
    if node_id not in node_map:
        return None
    
    node = node_map[node_id].copy()
    
    # Convert childIds back to nested children
    child_ids = node.pop('childIds', [])
    node['children'] = []
    
    for child_id in child_ids:
        child_node = reconstruct_node_tree(child_id, node_map)
        if child_node:
            node['children'].append(child_node)
    
    return node


def reconstruct_from_chunked_data(hierarchy_id, doc_data):
    """Reconstruct hierarchy from chunked subcollection data"""
    global db, using_mock
    
    if using_mock:
        print(f"⚠️ MOCK DATABASE: Cannot reconstruct from chunks in mock mode")
        return doc_data
    
    try:
        # Load all chunks from subcollection
        chunks_ref = db.collection('learning_goals_hierarchies').document(hierarchy_id).collection('node_chunks')
        chunk_docs = chunks_ref.stream()
        
        all_nodes = []
        for chunk_doc in chunk_docs:
            chunk_data = chunk_doc.to_dict()
            chunk_nodes = chunk_data.get('nodes', [])
            all_nodes.extend(chunk_nodes)
            print(f"  📦 Loaded chunk {chunk_data.get('chunk_id', 'unknown')} with {len(chunk_nodes)} nodes")
        
        if all_nodes:
            # Reconstruct nested structure from flattened chunks
            temp_doc_data = {'root_nodes': all_nodes}
            reconstructed_doc_data = reconstruct_from_flattened_nodes(temp_doc_data)
            
            # Update original document data with reconstructed structure
            doc_data['root_nodes'] = reconstructed_doc_data['root_nodes']
            
            # Update description to remove flattening note
            if doc_data.get('description', '').endswith(' (Maximum Flattening)'):
                doc_data['description'] = doc_data['description'][:-21]  # Remove " (Maximum Flattening)"
            
            print(f"✅ Chunked hierarchy reconstructed with {len(all_nodes)} total nodes")
        else:
            print(f"⚠️ No chunk data found, using summary structure")
        
    except Exception as e:
        print(f"⚠️ Error reconstructing from chunks: {e}")
        print(f"📋 Using minimal summary as fallback")
    
    return doc_data

def get_hierarchies(limit=20, creator=None, course_name=None):
    """Retrieve multiple hierarchies with optional filtering"""
    global db, using_mock
    
    if not db:
        init_mock_services()
    
    if using_mock:
        print(f"⚠️ MOCK DATABASE: Searching for hierarchies in mock Firestore")
    else:
        print(f"✅ REAL DATABASE: Searching for hierarchies in Firestore")
        
    query = db.collection('learning_goals_hierarchies')
    
    # Apply filters
    if creator:
        query = query.where('creator', '==', creator)
    if course_name:
        query = query.where('course_name', '==', course_name)
    
    # Order by modified date (most recent first) and limit
    query = query.order_by('modified_at', direction=firestore.Query.DESCENDING if not using_mock else None)
    query = query.limit(limit)
    
    docs = query.stream()
    results = []
    
    from app.models import LearningGoalsHierarchy
    for doc in docs:
        doc_data = doc.to_dict()
        hierarchy = LearningGoalsHierarchy.from_dict(doc_data, hierarchy_id=doc.id)
        results.append(hierarchy)
    
    if using_mock:
        print(f"⚠️ MOCK DATABASE: Found {len(results)} hierarchies in mock database")
    else:
        print(f"✅ REAL DATABASE: Found {len(results)} hierarchies in Firestore")
        
    return results


def delete_hierarchy(hierarchy_id):
    """Delete a hierarchy from Firestore"""
    global db, using_mock
    
    if not db:
        init_mock_services()
    
    try:
        if using_mock:
            print(f"⚠️ MOCK DATABASE: Deleting hierarchy from mock Firestore: {hierarchy_id}")
            if hierarchy_id in db.collections.get('learning_goals_hierarchies', {}).documents:
                del db.collections['learning_goals_hierarchies'].documents[hierarchy_id]
                return True
            return False
        else:
            print(f"✅ REAL DATABASE: Deleting hierarchy from Firestore: {hierarchy_id}")
            db.collection('learning_goals_hierarchies').document(hierarchy_id).delete()
            return True
        
    except Exception as e:
        print(f"❌ Error deleting hierarchy: {e}")
        return False


def update_hierarchy_node_label(hierarchy_id, node_id, new_label):
    """Update a specific node's label within a hierarchy"""
    global db, using_mock
    
    if not db:
        init_mock_services()
    
    try:
        # Get the hierarchy
        hierarchy = get_hierarchy(hierarchy_id)
        if not hierarchy:
            return False
        
        # Find and update the node
        def update_node_in_tree(nodes, target_node_id, new_label):
            for node in nodes:
                if isinstance(node, dict):
                    if node.get('id') == target_node_id:
                        node['label'] = new_label
                        node['modified_at'] = datetime.datetime.now().isoformat()
                        return True
                    if 'children' in node:
                        if update_node_in_tree(node['children'], target_node_id, new_label):
                            return True
                else:
                    if node.id == target_node_id:
                        node.label = new_label
                        node.modified_at = datetime.datetime.now()
                        return True
                    if hasattr(node, 'children') and node.children:
                        if update_node_in_tree(node.children, target_node_id, new_label):
                            return True
            return False
        
        # Update the node
        node_updated = update_node_in_tree(hierarchy.root_nodes, node_id, new_label)
        
        if node_updated:
            # Update the hierarchy's modified timestamp
            hierarchy.modified_at = datetime.datetime.now()
            
            # Save back to database
            if using_mock:
                print(f"⚠️ MOCK DATABASE: Updating hierarchy node in mock Firestore: {hierarchy_id}/{node_id}")
                db.collections['learning_goals_hierarchies'].documents[hierarchy_id].data = hierarchy.to_dict()
            else:
                print(f"✅ REAL DATABASE: Updating hierarchy node in Firestore: {hierarchy_id}/{node_id}")
                doc_ref = db.collection('learning_goals_hierarchies').document(hierarchy_id)
                doc_ref.set(hierarchy.to_dict())
            
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ Error updating hierarchy node: {e}")
        return False


# Artifact Management Functions
def save_artifact(artifact):
    """Save a tree artifact to Firestore with proper serialization"""
    global db, using_mock
    
    if not db:
        init_mock_services()
    
    if using_mock:
        print(f"⚠️ MOCK DATABASE: Saving artifact '{artifact.name}' to mock Firestore")
    else:
        print(f"✅ REAL DATABASE: Saving artifact '{artifact.name}' to Firestore")
    
    # Create the artifact document
    doc_ref = db.collection('tree_artifacts').document()
    
    # Serialize the artifact structure for Firestore
    serialized_artifact = serialize_artifact_for_firestore(artifact)
    doc_data = serialized_artifact.to_dict()
    
    # Convert datetime to Firebase timestamp
    if isinstance(doc_data['created_at'], datetime.datetime):
        if hasattr(firestore, 'SERVER_TIMESTAMP') and not using_mock:
            doc_data['created_at'] = firestore.SERVER_TIMESTAMP
            doc_data['modified_at'] = firestore.SERVER_TIMESTAMP
    
    try:
        # Save the artifact document
        doc_id = doc_ref.set(doc_data)
        
        if using_mock:
            print(f"⚠️ MOCK DATABASE: Artifact saved with mock ID: {doc_ref.id}")
        else:
            print(f"✅ REAL DATABASE: Artifact saved to Firestore with ID: {doc_ref.id}")
        
        return doc_ref.id
        
    except Exception as e:
        print(f"❌ Error saving artifact: {e}")
        # If still too large, try aggressive flattening
        if "exceeds the maximum allowed size" in str(e) or "invalid nested entity" in str(e):
            print("📝 Applying aggressive flattening for artifact Firestore compatibility...")
            return save_artifact_with_aggressive_flattening(artifact, doc_ref)
        raise


def serialize_artifact_for_firestore(artifact):
    """Serialize artifact structure to be Firestore-compatible"""
    from app.models import Artifact
    
    # Flatten the nested tree structure to avoid Firestore nesting limits
    flattened_structure = flatten_tree_nodes(artifact.tree_structure)
    
    # Create a serializable artifact
    serialized_artifact = Artifact(
        id=artifact.id,
        name=artifact.name,
        tree_structure=flattened_structure,
        parameters=artifact.parameters,
        metadata=artifact.metadata,
        created_at=artifact.created_at,
        modified_at=artifact.modified_at,
        is_active=artifact.is_active
    )
    
    return serialized_artifact


def flatten_tree_nodes(nodes):
    """Flatten tree nodes to avoid nested object issues in Firestore"""
    if not nodes:
        return []
    
    flattened = []
    
    for node in nodes:
        if not isinstance(node, dict):
            continue  # Skip non-dict nodes
        
        # Create a flattened version of the node with safe defaults
        flattened_node = {
            'id': node.get('id', str(uuid.uuid4())),  # Generate ID if missing
            'label': node.get('label', ''),
            'goals': node.get('goals', []),
            'sources': node.get('sources', []),
            'size': node.get('size', 0),
            'representative_goal': node.get('representative_goal', ''),
            'childIds': []  # Store child IDs instead of nested objects
        }
        
        # Extract child IDs and flatten children separately
        children = node.get('children', [])
        if children and isinstance(children, list):
            # Get child IDs safely
            child_ids = []
            for child in children:
                if isinstance(child, dict) and child.get('id'):
                    child_ids.append(child.get('id'))
                elif isinstance(child, dict):
                    # Generate ID for child if missing
                    child_id = str(uuid.uuid4())
                    child['id'] = child_id
                    child_ids.append(child_id)
            
            flattened_node['childIds'] = child_ids
            
            # Recursively flatten children and add them to the main list
            flattened_children = flatten_tree_nodes(children)
            flattened.extend(flattened_children)
        
        flattened.append(flattened_node)
    
    return flattened


def save_artifact_with_aggressive_flattening(artifact, doc_ref):
    """Save artifact with maximum flattening for large datasets"""
    from app.models import Artifact
    
    print("🔧 Applying maximum flattening for large artifact...")
    
    # Calculate summary statistics safely
    tree_structure = artifact.tree_structure or []
    total_goals = calculate_total_goals_in_tree(tree_structure)
    
    # Create minimal metadata
    minimal_metadata = {
        'total_goals': total_goals,
        'optimization_level': 'maximum_flattening',
        'original_save_time': datetime.datetime.now().isoformat()
    }
    
    # Create structure summary with minimal data
    structure_summary = create_minimal_tree_summary(tree_structure)
    
    minimal_artifact = Artifact(
        name=f"{artifact.name} (Flattened)",
        tree_structure=structure_summary,
        parameters=artifact.parameters or {},
        metadata=minimal_metadata,
        created_at=artifact.created_at,
        modified_at=artifact.modified_at,
        is_active=artifact.is_active
    )
    
    # Save minimal version
    doc_data = minimal_artifact.to_dict()
    
    if isinstance(doc_data['created_at'], datetime.datetime):
        if hasattr(firestore, 'SERVER_TIMESTAMP') and not using_mock:
            doc_data['created_at'] = firestore.SERVER_TIMESTAMP
            doc_data['modified_at'] = firestore.SERVER_TIMESTAMP
    
    doc_id = doc_ref.set(doc_data)
    
    # Save complete data in subcollection with chunking
    save_complete_artifact_data_chunked(doc_ref.id, tree_structure)
    
    print(f"✅ Artifact saved with maximum flattening: {doc_ref.id}")
    return doc_ref.id


def calculate_total_goals_in_tree(nodes):
    """Calculate total number of goals in a tree structure"""
    if not nodes:
        return 0
    
    total = 0
    for node in nodes:
        if isinstance(node, dict):
            # Count goals in this node - handle None explicitly
            goals = node.get('goals')
            if goals is None:
                goals = []
            if goals:
                total += len(goals)
            
            # Count goals in children - handle None explicitly
            children = node.get('children')
            if children is None:
                children = []
            if children:
                total += calculate_total_goals_in_tree(children)
            
            # If no goals but has size, use size
            if not goals and node.get('size', 0) > 0:
                total += node.get('size', 0)
    
    return total


def create_minimal_tree_summary(nodes):
    """Create an ultra-minimal summary of tree structure"""
    if not nodes:
        return []
    
    summary = []
    
    for i, node in enumerate(nodes):
        if isinstance(node, dict):
            # Handle children safely - check for None explicitly
            children = node.get('children')
            if children is None:
                children = []
            
            # Handle goals safely - check for None explicitly  
            goals = node.get('goals')
            if goals is None:
                goals = []
            
            node_summary = {
                'id': node.get('id', f'node_{i}'),
                'label': node.get('label', f'Group_{i+1}'),
                'displayLevel': node.get('displayLevel', 0),
                'size': node.get('size', 0),
                'hasChildren': len(children) > 0,
                'childCount': len(children),
                'goalCount': len(goals)
            }
        else:
            # Handle case where node might not be a dict
            node_summary = {
                'id': f'node_{i}',
                'label': f'Group_{i+1}',
                'displayLevel': 0,
                'size': 0,
                'hasChildren': False,
                'childCount': 0,
                'goalCount': 0
            }
        
        summary.append(node_summary)
    
    return summary


def save_complete_artifact_data_chunked(artifact_id, nodes):
    """Save complete artifact data in chunked subcollections"""
    global db, using_mock
    
    if using_mock:
        return
    
    if not nodes:
        print("⚠️ No nodes to save in chunks")
        return
    
    print(f"💾 Saving complete artifact data in chunks...")
    
    # Flatten all nodes first
    flattened_nodes = flatten_tree_nodes(nodes)
    
    if not flattened_nodes:
        print("⚠️ No flattened nodes to save")
        return
    
    # Save in chunks to avoid size limits
    chunk_size = 1000  # Start with larger chunks, split if too big
    chunks = [flattened_nodes[i:i + chunk_size] for i in range(0, len(flattened_nodes), chunk_size)]
    
    def save_chunk_with_retry(chunk_data, chunk_ref, chunk_id, attempt=1):
        """Save a chunk with retry logic that splits if too big"""
        try:
            chunk_ref.set(chunk_data)
            print(f"  📦 Saved artifact chunk {chunk_id} with {len(chunk_data['nodes'])} nodes (attempt {attempt})")
            return True
        except Exception as e:
            if ("exceeds the maximum allowed size" in str(e) or "invalid nested entity" in str(e)) and len(chunk_data['nodes']) > 1:
                # Split chunk in half and try again
                nodes = chunk_data['nodes']
                mid = len(nodes) // 2
                left_nodes = nodes[:mid]
                right_nodes = nodes[mid:]
                
                print(f"  🔄 Artifact chunk {chunk_id} too big ({len(nodes)} nodes), splitting into {len(left_nodes)} + {len(right_nodes)} nodes")
                
                # Save left half
                left_chunk_data = {
                    'chunk_id': f"{chunk_id}a",
                    'nodes': left_nodes,
                    'chunk_size': len(left_nodes),
                    'created_at': firestore.SERVER_TIMESTAMP
                }
                left_chunk_ref = db.collection('tree_artifacts').document(artifact_id).collection('node_chunks').document(f'chunk_{chunk_id}a')
                save_chunk_with_retry(left_chunk_data, left_chunk_ref, f"{chunk_id}a", attempt + 1)
                
                # Save right half
                right_chunk_data = {
                    'chunk_id': f"{chunk_id}b",
                    'nodes': right_nodes,
                    'chunk_size': len(right_nodes),
                    'created_at': firestore.SERVER_TIMESTAMP
                }
                right_chunk_ref = db.collection('tree_artifacts').document(artifact_id).collection('node_chunks').document(f'chunk_{chunk_id}b')
                save_chunk_with_retry(right_chunk_data, right_chunk_ref, f"{chunk_id}b", attempt + 1)
                
                return True
            else:
                print(f"  ❌ Failed to save artifact chunk {chunk_id}: {e}")
                return False
    
    total_saved = 0
    for i, chunk in enumerate(chunks):
        chunk_data = {
            'chunk_id': i,
            'nodes': chunk,
            'chunk_size': len(chunk),
            'created_at': firestore.SERVER_TIMESTAMP
        }
        
        chunk_ref = db.collection('tree_artifacts').document(artifact_id).collection('node_chunks').document(f'chunk_{i}')
        if save_chunk_with_retry(chunk_data, chunk_ref, i):
            total_saved += len(chunk)
    
    print(f"✅ Complete artifact data saved with {total_saved} total nodes")

def get_artifact(artifact_id):
    """Retrieve a specific artifact by its ID and reconstruct tree structure"""
    global db, using_mock
    
    if not db:
        init_mock_services()
    
    if using_mock:
        print(f"⚠️ MOCK DATABASE: Retrieving artifact with ID '{artifact_id}' from mock Firestore")
    else:
        print(f"✅ REAL DATABASE: Retrieving artifact with ID '{artifact_id}' from Firestore")
        
    doc_ref = db.collection('tree_artifacts').document(artifact_id)
    doc = doc_ref.get()
    
    if doc.exists:
        from app.models import Artifact
        doc_data = doc.to_dict()
        
        # Check if this is a flattened artifact that needs reconstruction
        if needs_artifact_reconstruction(doc_data):
            doc_data = reconstruct_artifact_structure(artifact_id, doc_data)
        
        if using_mock:
            print(f"⚠️ MOCK DATABASE: Found mock artifact: {artifact_id}")
        else:
            print(f"✅ REAL DATABASE: Found artifact in Firestore: {artifact_id}")
        return Artifact.from_dict(doc_data, artifact_id=doc.id)
    
    if using_mock:
        print(f"⚠️ MOCK DATABASE: Artifact not found in mock database: {artifact_id}")
    else:
        print(f"✅ REAL DATABASE: Artifact not found in Firestore: {artifact_id}")
    return None


def needs_artifact_reconstruction(doc_data):
    """Check if artifact document needs structure reconstruction"""
    tree_structure = doc_data.get('tree_structure', [])
    
    # Check if this is a flattened structure (nodes have childIds instead of children)
    for node in tree_structure:
        if isinstance(node, dict):
            if 'childIds' in node or ('hasChildren' in node and 'children' not in node):
                return True
            # Check if it's missing nested structure - handle None goals safely
            if 'children' not in node:
                goals = node.get('goals')
                if goals is None:
                    goals = []
                if node.get('size', 0) > len(goals):
                    return True
    
    return False


def reconstruct_artifact_structure(artifact_id, doc_data):
    """Reconstruct the nested artifact structure from flattened data"""
    global db, using_mock
    
    metadata = doc_data.get('metadata', {})
    optimization_level = metadata.get('optimization_level', '')
    
    if optimization_level == 'maximum_flattening':
        print(f"🔧 Reconstructing maximum flattened artifact...")
        return reconstruct_artifact_from_chunked_data(artifact_id, doc_data)
    else:
        print(f"🔧 Reconstructing flattened artifact structure...")
        return reconstruct_artifact_from_flattened_nodes(doc_data)


def reconstruct_artifact_from_flattened_nodes(doc_data):
    """Reconstruct artifact from flattened node structure"""
    tree_structure = doc_data.get('tree_structure', [])
    
    if not tree_structure:
        return doc_data
    
    # Check if nodes have childIds (flattened structure)
    if not any('childIds' in node for node in tree_structure if isinstance(node, dict)):
        # Already in correct format
        return doc_data
    
    # Create a map of all nodes by ID
    node_map = {}
    for node in tree_structure:
        if isinstance(node, dict) and 'id' in node:
            node_map[node['id']] = node.copy()
    
    # Reconstruct nested structure
    reconstructed_roots = []
    
    for node in tree_structure:
        if isinstance(node, dict):
            # Find root nodes (nodes that aren't children of other nodes)
            is_root = True
            for other_node in tree_structure:
                if isinstance(other_node, dict) and 'childIds' in other_node:
                    if node['id'] in other_node['childIds']:
                        is_root = False
                        break
            
            if is_root:
                reconstructed_node = reconstruct_artifact_node_tree(node['id'], node_map)
                if reconstructed_node:
                    reconstructed_roots.append(reconstructed_node)
    
    # Update the document data with reconstructed structure
    doc_data['tree_structure'] = reconstructed_roots
    
    print(f"✅ Artifact structure reconstructed with {len(reconstructed_roots)} root nodes")
    return doc_data


def reconstruct_artifact_node_tree(node_id, node_map):
    """Recursively reconstruct an artifact node tree from flattened data"""
    if node_id not in node_map:
        return None
    
    node = node_map[node_id].copy()
    
    # Convert childIds back to nested children
    child_ids = node.pop('childIds', [])
    node['children'] = []
    
    for child_id in child_ids:
        child_node = reconstruct_artifact_node_tree(child_id, node_map)
        if child_node:
            node['children'].append(child_node)
    
    return node


def reconstruct_artifact_from_chunked_data(artifact_id, doc_data):
    """Reconstruct artifact from chunked subcollection data"""
    global db, using_mock
    
    if using_mock:
        print(f"⚠️ MOCK DATABASE: Cannot reconstruct from chunks in mock mode")
        return doc_data
    
    try:
        # Load all chunks from subcollection
        chunks_ref = db.collection('tree_artifacts').document(artifact_id).collection('node_chunks')
        chunk_docs = chunks_ref.stream()
        
        all_nodes = []
        for chunk_doc in chunk_docs:
            chunk_data = chunk_doc.to_dict()
            chunk_nodes = chunk_data.get('nodes', [])
            all_nodes.extend(chunk_nodes)
            print(f"  📦 Loaded artifact chunk {chunk_data.get('chunk_id', 'unknown')} with {len(chunk_nodes)} nodes")
        
        if all_nodes:
            # Reconstruct nested structure from flattened chunks
            temp_doc_data = {'tree_structure': all_nodes}
            reconstructed_doc_data = reconstruct_artifact_from_flattened_nodes(temp_doc_data)
            
            # Update original document data with reconstructed structure
            doc_data['tree_structure'] = reconstructed_doc_data['tree_structure']
            
            # Update name to remove flattening note
            if doc_data.get('name', '').endswith(' (Flattened)'):
                doc_data['name'] = doc_data['name'][:-12]  # Remove " (Flattened)"
            
            print(f"✅ Chunked artifact reconstructed with {len(all_nodes)} total nodes")
        else:
            print(f"⚠️ No chunk data found, using summary structure")
        
    except Exception as e:
        print(f"⚠️ Error reconstructing from chunks: {e}")
        print(f"📋 Using minimal summary as fallback")
    
    return doc_data

def get_artifacts(limit=20, creator=None):
    """Retrieve multiple artifacts with optional filtering"""
    global db, using_mock
    
    if not db:
        init_mock_services()
    
    if using_mock:
        print(f"⚠️ MOCK DATABASE: Searching for artifacts in mock Firestore")
    else:
        print(f"✅ REAL DATABASE: Searching for artifacts in Firestore")
        
    query = db.collection('tree_artifacts')
    
    # Apply filters
    if creator:
        query = query.where('creator', '==', creator)
    
    # Order by modified date (most recent first) and limit
    if not using_mock:
        query = query.order_by('modified_at', direction=firestore.Query.DESCENDING)
    query = query.limit(limit)
    
    docs = query.stream()
    results = []
    
    from app.models import Artifact
    for doc in docs:
        doc_data = doc.to_dict()
        artifact = Artifact.from_dict(doc_data, artifact_id=doc.id)
        results.append(artifact)
    
    if using_mock:
        print(f"⚠️ MOCK DATABASE: Found {len(results)} artifacts in mock database")
    else:
        print(f"✅ REAL DATABASE: Found {len(results)} artifacts in Firestore")
        
    return results


def delete_artifact(artifact_id):
    """Delete an artifact from Firestore"""
    global db, using_mock
    
    if not db:
        init_mock_services()
    
    try:
        if using_mock:
            print(f"⚠️ MOCK DATABASE: Deleting artifact from mock Firestore: {artifact_id}")
            if artifact_id in db.collections.get('tree_artifacts', {}).documents:
                del db.collections['tree_artifacts'].documents[artifact_id]
                return True
            return False
        else:
            print(f"✅ REAL DATABASE: Deleting artifact from Firestore: {artifact_id}")
            db.collection('tree_artifacts').document(artifact_id).delete()
            return True
        
    except Exception as e:
        print(f"❌ Error deleting artifact: {e}")
        return False


def update_artifact_node_label(artifact_id, node_id, new_label):
    """Update a specific node's label within an artifact"""
    global db, using_mock
    
    if not db:
        init_mock_services()
    
    try:
        # Get the artifact (this will automatically reconstruct if needed)
        artifact = get_artifact(artifact_id)
        if not artifact:
            return False
        
        # Check if this artifact was saved with chunking
        metadata = artifact.metadata or {}
        is_chunked = metadata.get('optimization_level') == 'maximum_flattening'
        
        if is_chunked:
            # For chunked artifacts, only update the chunks and minimal summary
            print(f"🔧 Updating chunked artifact {artifact_id} node {node_id}")
            
            # Update the chunked data
            success = update_chunked_artifact_node_label(artifact_id, node_id, new_label)
            if not success:
                return False
            
            # Update the minimal summary in the main document if the node exists there
            if using_mock:
                print(f"⚠️ MOCK DATABASE: Chunked update for artifact: {artifact_id}/{node_id}")
                return True
            else:
                # Get the current main document
                doc_ref = db.collection('tree_artifacts').document(artifact_id)
                doc = doc_ref.get()
                if doc.exists:
                    doc_data = doc.to_dict()
                    tree_structure = doc_data.get('tree_structure', [])
                    
                    # Update the node in the minimal summary if it exists
                    updated_summary = False
                    for node in tree_structure:
                        if isinstance(node, dict) and node.get('id') == node_id:
                            node['label'] = new_label
                            updated_summary = True
                            break
                    
                    # Only update the main document if we found the node in the summary
                    if updated_summary:
                        doc_data['modified_at'] = firestore.SERVER_TIMESTAMP
                        doc_ref.set(doc_data)
                        print(f"✅ Updated minimal summary for chunked artifact")
                    
                    return True
                return False
        else:
            # For regular artifacts, use the normal update process
            return update_regular_artifact_node_label(artifact, artifact_id, node_id, new_label)
        
    except Exception as e:
        print(f"❌ Error updating artifact node: {e}")
        return False


def update_regular_artifact_node_label(artifact, artifact_id, node_id, new_label):
    """Update node label for regular (non-chunked) artifacts"""
    global db, using_mock
    
    # Find and update the node in tree structure
    def update_node_in_tree(nodes, target_node_id, new_label):
        for node in nodes:
            if isinstance(node, dict):
                if node.get('id') == target_node_id:
                    node['label'] = new_label
                    return True
                if 'children' in node:
                    if update_node_in_tree(node['children'], target_node_id, new_label):
                        return True
        return False
    
    # Update the node
    node_updated = update_node_in_tree(artifact.tree_structure, node_id, new_label)
    
    if node_updated:
        # Update the artifact's modified timestamp
        artifact.modified_at = datetime.datetime.now()
        
        # Re-serialize the artifact for Firestore (handle flattening if needed)
        serialized_artifact = serialize_artifact_for_firestore(artifact)
        doc_data = serialized_artifact.to_dict()
        
        # Convert datetime to Firebase timestamp
        if isinstance(doc_data['modified_at'], datetime.datetime):
            if hasattr(firestore, 'SERVER_TIMESTAMP') and not using_mock:
                doc_data['modified_at'] = firestore.SERVER_TIMESTAMP
        
        # Save back to database
        if using_mock:
            print(f"⚠️ MOCK DATABASE: Updating artifact representative text in mock Firestore: {artifact_id}/{node_id}")
            if 'tree_artifacts' not in db.collections:
                db.collections['tree_artifacts'] = type(db.collections['documents'])('tree_artifacts')
            db.collections['tree_artifacts'].documents[artifact_id].data = doc_data
        else:
            print(f"✅ REAL DATABASE: Updating artifact representative text in Firestore: {artifact_id}/{node_id}")
            doc_ref = db.collection('tree_artifacts').document(artifact_id)
            doc_ref.set(doc_data)
        
        return True
    
    return False


def update_chunked_artifact_node_label(artifact_id, node_id, new_label):
    """Update node label in chunked artifact data"""
    global db, using_mock
    
    if using_mock:
        return True
    
    print(f"🔧 Updating chunked artifact data for node {node_id}")
    
    # Get all chunks
    chunks_ref = db.collection('tree_artifacts').document(artifact_id).collection('node_chunks')
    chunk_docs = chunks_ref.stream()
    
    for chunk_doc in chunk_docs:
        chunk_data = chunk_doc.to_dict()
        nodes = chunk_data.get('nodes', [])
        updated = False
        
        # Update the node in this chunk if found
        for node in nodes:
            if isinstance(node, dict) and node.get('id') == node_id:
                node['label'] = new_label
                updated = True
                break
        
        # Save the chunk back if it was updated
        if updated:
            chunk_data['nodes'] = nodes
            chunk_doc.reference.set(chunk_data)
            print(f"  📦 Updated node {node_id} in chunk {chunk_data.get('chunk_id', 'unknown')}")
            return True
    
    # If we didn't find the node in any chunk, return False
    print(f"  ⚠️ Node {node_id} not found in any chunk")
    return False

def update_artifact_node_representative_text(artifact_id, node_id, representative_text, text_state='manual', ai_prompt=None):
    """Update a specific node's representative text, state, and AI prompt within an artifact"""
    global db, using_mock
    
    if not db:
        init_mock_services()
    
    try:
        # Get the artifact (this will automatically reconstruct if needed)
        artifact = get_artifact(artifact_id)
        if not artifact:
            return False
        
        # Check if this artifact was saved with chunking
        metadata = artifact.metadata or {}
        is_chunked = metadata.get('optimization_level') == 'maximum_flattening'
        
        if is_chunked:
            # For chunked artifacts, only update the chunks and minimal summary
            print(f"🔧 Updating chunked artifact {artifact_id} node {node_id} representative text")
            
            # Update the chunked data
            success = update_chunked_artifact_node_representative_text(
                artifact_id, node_id, representative_text, text_state, ai_prompt
            )
            if not success:
                return False
            
            # Update the minimal summary in the main document if the node exists there
            if using_mock:
                print(f"⚠️ MOCK DATABASE: Chunked representative text update for artifact: {artifact_id}/{node_id}")
                return True
            else:
                # Get the current main document
                doc_ref = db.collection('tree_artifacts').document(artifact_id)
                doc = doc_ref.get()
                if doc.exists:
                    doc_data = doc.to_dict()
                    tree_structure = doc_data.get('tree_structure', [])
                    
                    # Update the node in the minimal summary if it exists
                    updated_summary = False
                    for node in tree_structure:
                        if isinstance(node, dict) and node.get('id') == node_id:
                            # Update relevant fields in the summary
                            node['representative_goal'] = representative_text
                            node['text_state'] = text_state
                            if ai_prompt:
                                node['ai_prompt'] = ai_prompt
                            updated_summary = True
                            break
                    
                    # Only update the main document if we found the node in the summary
                    if updated_summary:
                        doc_data['modified_at'] = firestore.SERVER_TIMESTAMP
                        doc_ref.set(doc_data)
                        print(f"✅ Updated minimal summary for chunked artifact representative text")
                    
                    return True
                return False
        else:
            # For regular artifacts, use the normal update process
            return update_regular_artifact_node_representative_text(
                artifact, artifact_id, node_id, representative_text, text_state, ai_prompt
            )
        
    except Exception as e:
        print(f"❌ Error updating artifact node representative text: {e}")
        return False


def update_regular_artifact_node_representative_text(artifact, artifact_id, node_id, representative_text, text_state, ai_prompt):
    """Update representative text for regular (non-chunked) artifacts"""
    global db, using_mock
    
    # Find and update the node in tree structure
    def update_node_in_tree(nodes, target_node_id, representative_text, text_state, ai_prompt):
        for node in nodes:
            if isinstance(node, dict):
                if node.get('id') == target_node_id:
                    node['representative_goal'] = representative_text
                    node['text_state'] = text_state
                    if ai_prompt:
                        node['ai_prompt'] = ai_prompt
                    return True
                if 'children' in node:
                    if update_node_in_tree(node['children'], target_node_id, representative_text, text_state, ai_prompt):
                        return True
        return False
    
    # Update the node
    node_updated = update_node_in_tree(artifact.tree_structure, node_id, representative_text, text_state, ai_prompt)
    
    if node_updated:
        # Update the artifact's modified timestamp
        artifact.modified_at = datetime.datetime.now()
        
        # Re-serialize the artifact for Firestore (handle flattening if needed)
        serialized_artifact = serialize_artifact_for_firestore(artifact)
        doc_data = serialized_artifact.to_dict()
        
        # Convert datetime to Firebase timestamp
        if isinstance(doc_data['modified_at'], datetime.datetime):
            if hasattr(firestore, 'SERVER_TIMESTAMP') and not using_mock:
                doc_data['modified_at'] = firestore.SERVER_TIMESTAMP
        
        # Save back to database
        if using_mock:
            print(f"⚠️ MOCK DATABASE: Updating artifact representative text in mock Firestore: {artifact_id}/{node_id}")
            if 'tree_artifacts' not in db.collections:
                db.collections['tree_artifacts'] = type(db.collections['documents'])('tree_artifacts')
            db.collections['tree_artifacts'].documents[artifact_id].data = doc_data
        else:
            print(f"✅ REAL DATABASE: Updating artifact representative text in Firestore: {artifact_id}/{node_id}")
            doc_ref = db.collection('tree_artifacts').document(artifact_id)
            doc_ref.set(doc_data)
        
        return True
    
    return False


def update_chunked_artifact_node_representative_text(artifact_id, node_id, representative_text, text_state, ai_prompt):
    """Update representative text in chunked artifact data"""
    global db, using_mock
    
    if using_mock:
        return True
    
    print(f"🔧 Updating chunked artifact representative text for node {node_id}")
    
    # Get all chunks
    chunks_ref = db.collection('tree_artifacts').document(artifact_id).collection('node_chunks')
    chunk_docs = chunks_ref.stream()
    
    for chunk_doc in chunk_docs:
        chunk_data = chunk_doc.to_dict()
        nodes = chunk_data.get('nodes', [])
        updated = False
        
        # Update the node in this chunk if found
        for node in nodes:
            if isinstance(node, dict) and node.get('id') == node_id:
                node['representative_goal'] = representative_text
                node['text_state'] = text_state
                if ai_prompt:
                    node['ai_prompt'] = ai_prompt
                updated = True
                break
        
        # Save the chunk back if it was updated
        if updated:
            chunk_data['nodes'] = nodes
            chunk_doc.reference.set(chunk_data)
            print(f"  📦 Updated representative text for node {node_id} in chunk {chunk_data.get('chunk_id', 'unknown')}")
            return True
    
    # If we didn't find the node in any chunk, return False
    print(f"  ⚠️ Node {node_id} not found in any chunk")
    return False


def batch_update_artifact_node_representative_texts(artifact_id, updates):
    """Efficiently batch update multiple node representative texts in an artifact"""
    global db, using_mock
    
    if not db:
        init_mock_services()
    
    try:
        print(f"🔧 Batch updating {len(updates)} nodes in artifact {artifact_id}")
        
        # Get the artifact (this will automatically reconstruct if needed)
        artifact = get_artifact(artifact_id)
        if not artifact:
            print(f"❌ Artifact not found: {artifact_id}")
            return False
        
        # Check if this artifact was saved with chunking
        metadata = artifact.metadata or {}
        is_chunked = metadata.get('optimization_level') == 'maximum_flattening'
        
        if is_chunked:
            return batch_update_chunked_artifact(artifact_id, updates)
        else:
            return batch_update_regular_artifact(artifact, artifact_id, updates)
        
    except Exception as e:
        print(f"❌ Error batch updating artifact nodes: {e}")
        return False


def batch_update_chunked_artifact(artifact_id, updates):
    """Efficiently batch update chunked artifact by loading chunks once and updating all nodes"""
    global db, using_mock
    
    if using_mock:
        return True
    
    print(f"🔧 Batch updating chunked artifact {artifact_id}")
    
    # Create a mapping of updates by node_id for quick lookup
    updates_by_node_id = {update['node_id']: update for update in updates}
    
    # Get all chunks
    chunks_ref = db.collection('tree_artifacts').document(artifact_id).collection('node_chunks')
    chunk_docs = list(chunks_ref.stream())
    
    updated_chunks = []
    nodes_found = set()
    
    # Process each chunk
    for chunk_doc in chunk_docs:
        chunk_data = chunk_doc.to_dict()
        nodes = chunk_data.get('nodes', [])
        chunk_updated = False
        
        # Update all nodes in this chunk that need updating
        for node in nodes:
            if isinstance(node, dict):
                node_id = node.get('id')
                if node_id in updates_by_node_id:
                    update = updates_by_node_id[node_id]
                    node['representative_goal'] = update['representative_text']
                    node['text_state'] = update['text_state']
                    if update.get('ai_prompt'):
                        node['ai_prompt'] = update['ai_prompt']
                    
                    nodes_found.add(node_id)
                    chunk_updated = True
        
        # Mark chunk for saving if it was updated
        if chunk_updated:
            chunk_data['nodes'] = nodes
            updated_chunks.append((chunk_doc.reference, chunk_data))
    
    # Save all updated chunks in batch
    for chunk_ref, chunk_data in updated_chunks:
        chunk_ref.set(chunk_data)
        print(f"  📦 Updated chunk {chunk_data.get('chunk_id', 'unknown')} with {len([n for n in chunk_data['nodes'] if isinstance(n, dict) and n.get('id') in updates_by_node_id])} node updates")
    
    # Update the minimal summary in the main document if needed
    if nodes_found:
        try:
            doc_ref = db.collection('tree_artifacts').document(artifact_id)
            doc = doc_ref.get()
            if doc.exists:
                doc_data = doc.to_dict()
                tree_structure = doc_data.get('tree_structure', [])
                
                # Update nodes in the minimal summary if they exist
                summary_updated = False
                for node in tree_structure:
                    if isinstance(node, dict):
                        node_id = node.get('id')
                        if node_id in updates_by_node_id:
                            update = updates_by_node_id[node_id]
                            node['representative_goal'] = update['representative_text']
                            node['text_state'] = update['text_state']
                            if update.get('ai_prompt'):
                                node['ai_prompt'] = update['ai_prompt']
                            summary_updated = True
                
                # Save the main document if summary was updated
                if summary_updated:
                    doc_data['modified_at'] = firestore.SERVER_TIMESTAMP
                    doc_ref.set(doc_data)
                    print(f"  📄 Updated minimal summary for {len(nodes_found)} nodes")
        except Exception as e:
            print(f"⚠️ Error updating minimal summary: {e}")
    
    # Check if all nodes were found and updated
    not_found = set(updates_by_node_id.keys()) - nodes_found
    if not_found:
        print(f"  ⚠️ Nodes not found: {list(not_found)}")
    
    print(f"✅ Batch update completed: {len(nodes_found)}/{len(updates)} nodes updated")
    return len(nodes_found) > 0


def batch_update_regular_artifact(artifact, artifact_id, updates):
    """Batch update regular (non-chunked) artifact by updating tree structure once"""
    global db, using_mock
    
    print(f"🔧 Batch updating regular artifact {artifact_id}")
    
    # Create a mapping of updates by node_id for quick lookup
    updates_by_node_id = {update['node_id']: update for update in updates}
    
    # Find and update all nodes in tree structure
    def update_nodes_in_tree(nodes, updates_map):
        nodes_found = set()
        for node in nodes:
            if isinstance(node, dict):
                node_id = node.get('id')
                if node_id in updates_map:
                    update = updates_map[node_id]
                    node['representative_goal'] = update['representative_text']
                    node['text_state'] = update['text_state']
                    if update.get('ai_prompt'):
                        node['ai_prompt'] = update['ai_prompt']
                    nodes_found.add(node_id)
                
                # Recursively update children
                if 'children' in node:
                    child_nodes_found = update_nodes_in_tree(node['children'], updates_map)
                    nodes_found.update(child_nodes_found)
        
        return nodes_found
    
    # Update all nodes in the tree structure
    nodes_found = update_nodes_in_tree(artifact.tree_structure, updates_by_node_id)
    
    if nodes_found:
        # Update the artifact's modified timestamp
        artifact.modified_at = datetime.datetime.now()
        
        # Re-serialize the artifact for Firestore (handle flattening if needed)
        serialized_artifact = serialize_artifact_for_firestore(artifact)
        doc_data = serialized_artifact.to_dict()
        
        # Convert datetime to Firebase timestamp
        if isinstance(doc_data['modified_at'], datetime.datetime):
            if hasattr(firestore, 'SERVER_TIMESTAMP') and not using_mock:
                doc_data['modified_at'] = firestore.SERVER_TIMESTAMP
        
        # Save back to database
        if using_mock:
            print(f"⚠️ MOCK DATABASE: Batch updating artifact in mock Firestore: {artifact_id}")
            if 'tree_artifacts' not in db.collections:
                db.collections['tree_artifacts'] = type(db.collections['documents'])('tree_artifacts')
            db.collections['tree_artifacts'].documents[artifact_id].data = doc_data
        else:
            print(f"✅ REAL DATABASE: Batch updating artifact in Firestore: {artifact_id}")
            doc_ref = db.collection('tree_artifacts').document(artifact_id)
            doc_ref.set(doc_data)
    
    # Check if all nodes were found and updated
    not_found = set(updates_by_node_id.keys()) - nodes_found
    if not_found:
        print(f"  ⚠️ Nodes not found: {list(not_found)}")
    
    print(f"✅ Batch update completed: {len(nodes_found)}/{len(updates)} nodes updated")
    return len(nodes_found) > 0