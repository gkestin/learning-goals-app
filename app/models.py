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

class HierarchyNode:
    """Model for storing hierarchical learning goals structure"""
    
    def __init__(self, id=None, label="", original_label="", goals=None, levels=None, 
                 signature=None, children=None, is_expanded=False, parent_id=None, 
                 hierarchy_id=None, created_at=None, modified_at=None):
        self.id = id
        self.label = label
        self.original_label = original_label or label
        self.goals = goals or []
        self.levels = levels or {}  # Dictionary of level_number: value
        self.signature = signature or []  # Pattern signature for grouping
        self.children = children or []
        self.is_expanded = is_expanded
        self.parent_id = parent_id
        self.hierarchy_id = hierarchy_id
        self.created_at = created_at or datetime.now()
        self.modified_at = modified_at or datetime.now()
    
    @staticmethod
    def from_dict(data, node_id=None):
        """Create a HierarchyNode instance from Firestore document data"""
        node = HierarchyNode(
            id=node_id,
            label=data.get('label', ''),
            original_label=data.get('original_label', ''),
            goals=data.get('goals', []),
            levels=data.get('levels', {}),
            signature=data.get('signature', []),
            children=data.get('children', []),
            is_expanded=data.get('is_expanded', False),
            parent_id=data.get('parent_id', None),
            hierarchy_id=data.get('hierarchy_id', None),
            created_at=data.get('created_at', datetime.now()),
            modified_at=data.get('modified_at', datetime.now())
        )
        return node
    
    def to_dict(self):
        """Convert HierarchyNode instance to a dictionary for Firestore"""
        return {
            'label': self.label,
            'original_label': self.original_label,
            'goals': self.goals,
            'levels': self.levels,
            'signature': self.signature,
            'children': self.children,
            'is_expanded': self.is_expanded,
            'parent_id': self.parent_id,
            'hierarchy_id': self.hierarchy_id,
            'created_at': self.created_at,
            'modified_at': self.modified_at
        }


class LearningGoalsHierarchy:
    """Model for storing the complete hierarchical learning goals structure"""
    
    def __init__(self, id=None, name="", description="", creator="", course_name="", 
                 institution="", root_nodes=None, metadata=None, created_at=None, 
                 modified_at=None, is_active=True):
        self.id = id
        self.name = name
        self.description = description
        self.creator = creator
        self.course_name = course_name
        self.institution = institution
        self.root_nodes = root_nodes or []
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.now()
        self.modified_at = modified_at or datetime.now()
        self.is_active = is_active
    
    def to_dict(self):
        return {
            'name': self.name,
            'description': self.description,
            'creator': self.creator,
            'course_name': self.course_name,
            'institution': self.institution,
            'root_nodes': self.root_nodes,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'modified_at': self.modified_at,
            'is_active': self.is_active
        }
    
    @classmethod
    def from_dict(cls, data, hierarchy_id=None):
        hierarchy = cls(
            id=hierarchy_id,
            name=data.get('name', ''),
            description=data.get('description', ''),
            creator=data.get('creator', ''),
            course_name=data.get('course_name', ''),
            institution=data.get('institution', ''),
            root_nodes=data.get('root_nodes', []),
            metadata=data.get('metadata', {}),
            created_at=data.get('created_at'),
            modified_at=data.get('modified_at'),
            is_active=data.get('is_active', True)
        )
        return hierarchy
    
    def get_total_goals(self):
        """Calculate total number of goals in the hierarchy"""
        def count_goals(nodes):
            total = 0
            for node in nodes:
                if isinstance(node, dict):
                    total += len(node.get('goals', []))
                    total += count_goals(node.get('children', []))
                else:
                    total += len(getattr(node, 'goals', []))
                    total += count_goals(getattr(node, 'children', []))
            return total
        
        return count_goals(self.root_nodes)
    
    def get_total_groups(self):
        """Calculate total number of groups in the hierarchy"""
        def count_groups(nodes):
            total = len(nodes)
            for node in nodes:
                if isinstance(node, dict):
                    total += count_groups(node.get('children', []))
                else:
                    total += count_groups(getattr(node, 'children', []))
            return total
        
        return count_groups(self.root_nodes)


class Artifact:
    def __init__(self, id=None, name="", tree_structure=None, parameters=None, 
                 metadata=None, created_at=None, modified_at=None, is_active=True):
        self.id = id
        self.name = name
        self.tree_structure = tree_structure or []
        self.parameters = parameters or {}
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.now()
        self.modified_at = modified_at or datetime.now()
        self.is_active = is_active
    
    def to_dict(self):
        return {
            'name': self.name,
            'tree_structure': self.tree_structure,
            'parameters': self.parameters,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'modified_at': self.modified_at,
            'is_active': self.is_active
        }
    
    @classmethod
    def from_dict(cls, data, artifact_id=None):
        artifact = cls(
            id=artifact_id,
            name=data.get('name', ''),
            tree_structure=data.get('tree_structure', []),
            parameters=data.get('parameters', {}),
            metadata=data.get('metadata', {}),
            created_at=data.get('created_at'),
            modified_at=data.get('modified_at'),
            is_active=data.get('is_active', True)
        )
        return artifact
    
    def get_total_goals(self):
        """Calculate total number of goals in the artifact"""
        # First try to get from metadata
        metadata_goals = self.metadata.get('total_goals', 0) if self.metadata else 0
        if metadata_goals and metadata_goals > 0:
            return metadata_goals
        
        # If not in metadata, calculate from tree structure
        if not self.tree_structure:
            return 0
        
        def count_goals_in_nodes(nodes):
            if not nodes:
                return 0
            
            total = 0
            for node in nodes:
                if isinstance(node, dict):
                    # Count goals in this node
                    goals = node.get('goals', [])
                    if goals:
                        total += len(goals)
                    
                    # Count goals in children
                    children = node.get('children', [])
                    if children:
                        total += count_goals_in_nodes(children)
                    
                    # If no goals but has size, use size
                    if not goals and node.get('size', 0) > 0:
                        total += node.get('size', 0)
            
            return total
        
        return count_goals_in_nodes(self.tree_structure)
    
    def get_parameter_summary(self):
        """Get a summary string of the parameters used to create this artifact"""
        params = self.parameters
        return f"{params.get('n_levels', 'N/A')} levels, {params.get('linkage_method', 'N/A')} linkage, {params.get('sampling_method', 'N/A')} sampling" 