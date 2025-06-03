from datetime import datetime

class Document:
    def __init__(self, id=None, name="", original_filename="", creator="", course_name="", 
                 learning_goals=None, storage_path="", created_at=None, public_url=None, 
                 institution="", doc_type="", notes="", lo_extraction_prompt=""):
        self.id = id
        self.name = name 
        self.original_filename = original_filename
        self.creator = creator
        self.course_name = course_name
        self.institution = institution
        self.doc_type = doc_type
        self.notes = notes
        self.lo_extraction_prompt = lo_extraction_prompt
        self.learning_goals = learning_goals or []
        self.storage_path = storage_path
        self.created_at = created_at or datetime.now()
        self.public_url = public_url
    
    @staticmethod
    def from_dict(data, doc_id=None):
        """Create a Document instance from Firestore document data"""
        doc = Document(
            id=doc_id,
            name=data.get('name', ''),
            original_filename=data.get('original_filename', ''),
            creator=data.get('creator', ''),
            course_name=data.get('course_name', ''),
            institution=data.get('institution', ''),
            doc_type=data.get('doc_type', ''),
            notes=data.get('notes', ''),
            learning_goals=data.get('learning_goals', []),
            storage_path=data.get('storage_path', ''),
            created_at=data.get('created_at', datetime.now()),
            public_url=data.get('public_url', None),
            lo_extraction_prompt=data.get('lo_extraction_prompt', '')
        )
        return doc
    
    def to_dict(self):
        """Convert Document instance to a dictionary for Firestore"""
        return {
            'name': self.name,
            'original_filename': self.original_filename,
            'creator': self.creator,
            'course_name': self.course_name,
            'institution': self.institution,
            'doc_type': self.doc_type,
            'notes': self.notes,
            'learning_goals': self.learning_goals,
            'storage_path': self.storage_path,
            'created_at': self.created_at,
            'public_url': self.public_url,
            'lo_extraction_prompt': self.lo_extraction_prompt
        }
        
    @property
    def document_url(self):
        """Generate a Firebase Storage URL for the document with smart path resolution"""
        if not self.storage_path:
            return None
        
        # Try to smart-resolve the path if needed
        try:
            from app.firebase_service import smart_resolve_storage_path
            result = smart_resolve_storage_path(self, fix_in_db=True)
            
            if result['resolved_path']:
                # Use the resolved path for URL generation
                encoded_path = result['resolved_path'].replace('/', '%2F')
                return f"__FIREBASE_STORAGE_URL_BASE__{encoded_path}?alt=media"
            else:
                # Fallback to original path even if it might not work
                encoded_path = self.storage_path.replace('/', '%2F')
                return f"__FIREBASE_STORAGE_URL_BASE__{encoded_path}?alt=media"
        except Exception:
            # Fallback to original behavior if smart resolution fails
            encoded_path = self.storage_path.replace('/', '%2F')
            return f"__FIREBASE_STORAGE_URL_BASE__{encoded_path}?alt=media" 