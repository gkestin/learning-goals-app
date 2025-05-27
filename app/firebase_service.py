import os
import datetime
import firebase_admin
from firebase_admin import credentials, firestore, storage
from app.models import Document
import datetime as datetime_module  # For timedelta

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
    """Delete a document and its associated file from Firebase"""
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
        if doc.storage_path:
            if using_mock:
                print(f"⚠️ MOCK STORAGE: Pretending to delete file: {doc.storage_path}")
                if doc.storage_path in bucket.files:
                    del bucket.files[doc.storage_path]
            else:
                print(f"✅ REAL STORAGE: Deleting file from Firebase Storage: {doc.storage_path}")
                blob = bucket.blob(doc.storage_path)
                blob.delete()
                print(f"File deleted successfully: {doc.storage_path}")
        
        # 2. Delete the document from Firestore
        if using_mock:
            print(f"⚠️ MOCK DATABASE: Deleting document from mock Firestore: {doc_id}")
            if doc_id in db.collections['documents'].documents:
                del db.collections['documents'].documents[doc_id]
        else:
            print(f"✅ REAL DATABASE: Deleting document from Firestore: {doc_id}")
            db.collection('documents').document(doc_id).delete()
        
        print(f"Document deleted successfully: {doc_id}")
        return True
    except Exception as e:
        print(f"Error deleting document: {e}")
        return False 