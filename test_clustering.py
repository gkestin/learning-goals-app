#!/usr/bin/env python3
"""
Quick test script for the clustering service
"""

from app.clustering_service import LearningGoalsClusteringService

def test_clustering():
    print("Testing STEM Learning Goals Clustering Service...")
    
    # Sample learning goals for testing
    test_goals = [
        "Understand linear algebra concepts and matrix operations",
        "Analyze statistical data and interpret results",
        "Apply calculus to solve real-world problems",
        "Understand basic statistics and probability",
        "Implement machine learning algorithms",
        "Design and analyze algorithms for efficiency",
        "Understand the fundamentals of linear algebra",
        "Apply statistical methods to data analysis",
        "Solve differential equations using calculus",
        "Implement data structures and algorithms"
    ]
    
    print(f"Test goals: {len(test_goals)} learning objectives")
    
    # Initialize service
    service = LearningGoalsClusteringService()
    
    try:
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = service.generate_embeddings(test_goals)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Test hierarchical clustering
        print("\nTesting hierarchical clustering...")
        labels_hierarchical = service.cluster_hierarchical(embeddings, n_clusters=3)
        print(f"Hierarchical labels: {labels_hierarchical}")
        
        # Test density-based clustering  
        print("\nTesting density-based clustering...")
        labels_density = service.cluster_density_based(embeddings, min_cluster_size=2)
        print(f"Density-based labels: {labels_density}")
        
        # Calculate quality scores
        quality_hier = service.calculate_cluster_quality(embeddings, labels_hierarchical)
        quality_density = service.calculate_cluster_quality(embeddings, labels_density)
        
        print(f"\nQuality scores:")
        print(f"Hierarchical: {quality_hier:.3f}")
        print(f"Density-based: {quality_density:.3f}")
        
        # Show cluster summaries
        print("\nHierarchical clustering results:")
        clusters_hier = service.get_cluster_summary(test_goals, labels_hierarchical)
        for cluster_id, cluster_data in clusters_hier.items():
            print(f"  Cluster {cluster_id}: {cluster_data['size']} goals")
            for goal in cluster_data['goals']:
                print(f"    - {goal}")
        
        print("\nDensity-based clustering results:")
        clusters_density = service.get_cluster_summary(test_goals, labels_density)
        for cluster_id, cluster_data in clusters_density.items():
            print(f"  Cluster {cluster_id}: {cluster_data['size']} goals")
            for goal in cluster_data['goals']:
                print(f"    - {goal}")
        
        print("\n✅ Clustering service test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_clustering() 