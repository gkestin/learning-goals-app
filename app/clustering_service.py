from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np

# STEM context prefix for better embeddings
STEM_CONTEXT = "In STEM education, students should be able to: "

class LearningGoalsClusteringService:
    def __init__(self):
        # Load local sentence-transformer model
        self.model = SentenceTransformer('all-mpnet-base-v2')
        
    def prepare_goals_for_embedding(self, learning_goals):
        """Prepend STEM context to learning goals for better embeddings"""
        return [f"{STEM_CONTEXT}{goal}" for goal in learning_goals]
    
    def generate_embeddings(self, learning_goals):
        """Generate embeddings for learning goals"""
        prepared_goals = self.prepare_goals_for_embedding(learning_goals)
        embeddings = self.model.encode(prepared_goals)
        return embeddings
    
    def cluster_hierarchical(self, embeddings, n_clusters):
        """Agglomerative clustering"""
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        labels = clustering.fit_predict(embeddings)
        return labels
    

    
    def calculate_cluster_quality(self, embeddings, labels):
        """Calculate clustering quality metrics"""
        unique_labels = set(labels)
        if len(unique_labels) > 1 and len(unique_labels) < len(labels):
            try:
                return silhouette_score(embeddings, labels)
            except:
                return 0.0
        return 0.0
    
    def get_cluster_summary(self, goals, labels):
        """Organize goals into clusters and generate summaries"""
        clusters = {}
        
        for i, (goal, label) in enumerate(zip(goals, labels)):
            if label == -1:  # HDBSCAN outliers
                label = f"outlier_{i}"
            
            if label not in clusters:
                clusters[label] = {
                    'goals': [],
                    'size': 0
                }
            
            clusters[label]['goals'].append(goal)
            clusters[label]['size'] += 1
        
        # Generate representative goal for each cluster (longest goal)
        for cluster_id, cluster_data in clusters.items():
            cluster_data['representative_goal'] = max(
                cluster_data['goals'], 
                key=len
            )
        
        return clusters 