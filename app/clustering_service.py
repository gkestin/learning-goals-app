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
    
    def calculate_cluster_separation_metrics(self, embeddings, labels):
        """Calculate inter-cluster separation and intra-cluster cohesion"""
        from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
        import numpy as np
        
        if len(set(labels)) < 2:
            return 0.0, 1.0  # No separation possible with single cluster
            
        # Calculate cluster centroids
        cluster_centroids = {}
        unique_labels = set(labels)
        
        for label in unique_labels:
            cluster_mask = np.array(labels) == label
            cluster_embeddings = embeddings[cluster_mask]
            # Centroid is the mean of all embeddings in the cluster
            cluster_centroids[label] = np.mean(cluster_embeddings, axis=0)
        
        # Calculate inter-cluster separation (distance between centroids)
        centroid_list = list(cluster_centroids.values())
        if len(centroid_list) > 1:
            centroid_distances = cosine_distances(centroid_list)
            # Get average distance between all pairs of centroids
            n_centroids = len(centroid_list)
            total_distance = 0
            pair_count = 0
            for i in range(n_centroids):
                for j in range(i + 1, n_centroids):
                    total_distance += centroid_distances[i][j]
                    pair_count += 1
            inter_cluster_separation = total_distance / pair_count if pair_count > 0 else 0
        else:
            inter_cluster_separation = 0
        
        # Calculate intra-cluster cohesion (average similarity within clusters)
        total_cohesion = 0
        cluster_count = 0
        
        for label in unique_labels:
            cluster_mask = np.array(labels) == label
            cluster_embeddings = embeddings[cluster_mask]
            
            if len(cluster_embeddings) > 1:
                cluster_similarities = cosine_similarity(cluster_embeddings)
                # Get average similarity within cluster (excluding diagonal)
                n = len(cluster_embeddings)
                total_sim = 0
                pair_count = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        total_sim += cluster_similarities[i][j]
                        pair_count += 1
                
                if pair_count > 0:
                    cluster_cohesion = total_sim / pair_count
                    total_cohesion += cluster_cohesion
                    cluster_count += 1
        
        avg_intra_cluster_cohesion = total_cohesion / cluster_count if cluster_count > 0 else 0
        
        return float(inter_cluster_separation), float(avg_intra_cluster_cohesion)
    
    def find_optimal_cluster_sizes(self, embeddings, max_clusters=None, min_clusters=2, use_multires=True):
        """Find truly optimal cluster sizes by maximizing our actual quality metrics"""
        n_goals = len(embeddings)
        
        # Set reasonable bounds
        if max_clusters is None:
            max_clusters = min(n_goals - 1, n_goals // 2 + 50)  # Test up to half + buffer
        max_clusters = min(max_clusters, n_goals - 1)
        
        if max_clusters < min_clusters:
            return None, None, None
        
        # Use multi-resolution optimization for larger datasets (>100 potential clusters to test)
        search_range = max_clusters - min_clusters + 1
        if use_multires and search_range > 100:
            print(f"Using multi-resolution optimization for large search space ({search_range} clusters to test)")
            return self._find_optimal_multires(embeddings, max_clusters, min_clusters)
        else:
            print(f"Using exhaustive search for moderate search space ({search_range} clusters to test)")
            return self._find_optimal_exhaustive(embeddings, max_clusters, min_clusters)

    def _find_optimal_exhaustive(self, embeddings, max_clusters, min_clusters):
        """Exhaustive search - test every cluster size (original brute force method)"""
        # Test different cluster sizes
        cluster_sizes = range(min_clusters, max_clusters + 1)
        
        # Store all metrics
        results_data = []
        
        print(f"Testing cluster sizes from {min_clusters} to {max_clusters}...")
        
        for k in cluster_sizes:
            # Perform clustering
            labels = self.cluster_hierarchical(embeddings, k)
            
            # Calculate all our metrics
            silhouette_score = self.calculate_cluster_quality(embeddings, labels)
            inter_sep, intra_cohesion = self.calculate_cluster_separation_metrics(embeddings, labels)
            
            # Calculate composite quality score
            composite_score = self._calculate_composite_score(silhouette_score, inter_sep, intra_cohesion)
            
            results_data.append({
                'k': k,
                'silhouette': silhouette_score,
                'separation': inter_sep,
                'cohesion': intra_cohesion,
                'composite': composite_score
            })
        
        return self._process_optimization_results(results_data)

    def _find_optimal_multires(self, embeddings, max_clusters, min_clusters):
        """Multi-resolution search: coarse grid first, then fine search around best regions"""
        
        # Phase 1: Coarse search (test every 8th-12th value depending on range)
        search_range = max_clusters - min_clusters + 1
        coarse_step = max(8, search_range // 15)  # Adaptive step size
        
        coarse_range = list(range(min_clusters, max_clusters + 1, coarse_step))
        # Ensure we include the max value
        if coarse_range[-1] != max_clusters:
            coarse_range.append(max_clusters)
        
        print(f"Phase 1: Coarse search with step size {coarse_step} ({len(coarse_range)} points)")
        
        coarse_results = []
        for k in coarse_range:
            labels = self.cluster_hierarchical(embeddings, k)
            silhouette_score = self.calculate_cluster_quality(embeddings, labels)
            inter_sep, intra_cohesion = self.calculate_cluster_separation_metrics(embeddings, labels)
            composite_score = self._calculate_composite_score(silhouette_score, inter_sep, intra_cohesion)
            
            coarse_results.append({
                'k': k,
                'silhouette': silhouette_score,
                'separation': inter_sep,
                'cohesion': intra_cohesion,
                'composite': composite_score
            })
        
        # Find the best coarse results for each metric
        best_coarse_composite = max(coarse_results, key=lambda x: x['composite'])
        best_coarse_separation = max(coarse_results, key=lambda x: x['separation'])
        best_coarse_cohesion = max(coarse_results, key=lambda x: x['cohesion'])
        
        # Phase 2: Fine search around promising regions
        # Create a set of regions to search finely
        fine_search_regions = set()
        
        # Add region around best composite score
        fine_search_regions.update(self._get_fine_search_region(
            best_coarse_composite['k'], min_clusters, max_clusters, coarse_step
        ))
        
        # Add region around best separation (if different)
        if best_coarse_separation['k'] != best_coarse_composite['k']:
            fine_search_regions.update(self._get_fine_search_region(
                best_coarse_separation['k'], min_clusters, max_clusters, coarse_step
            ))
        
        # Add region around best cohesion (if different)
        if best_coarse_cohesion['k'] != best_coarse_composite['k'] and best_coarse_cohesion['k'] != best_coarse_separation['k']:
            fine_search_regions.update(self._get_fine_search_region(
                best_coarse_cohesion['k'], min_clusters, max_clusters, coarse_step
            ))
        
        # Remove values we already tested in coarse search
        tested_coarse = {r['k'] for r in coarse_results}
        fine_search_regions = fine_search_regions - tested_coarse
        
        print(f"Phase 2: Fine search around promising regions ({len(fine_search_regions)} additional points)")
        
        # Perform fine search
        fine_results = []
        for k in sorted(fine_search_regions):
            labels = self.cluster_hierarchical(embeddings, k)
            silhouette_score = self.calculate_cluster_quality(embeddings, labels)
            inter_sep, intra_cohesion = self.calculate_cluster_separation_metrics(embeddings, labels)
            composite_score = self._calculate_composite_score(silhouette_score, inter_sep, intra_cohesion)
            
            fine_results.append({
                'k': k,
                'silhouette': silhouette_score,
                'separation': inter_sep,
                'cohesion': intra_cohesion,
                'composite': composite_score
            })
        
        # Combine all results
        all_results = coarse_results + fine_results
        all_results.sort(key=lambda x: x['k'])  # Sort by cluster size
        
        print(f"Multi-resolution optimization complete: tested {len(all_results)} points vs {search_range} exhaustive")
        
        return self._process_optimization_results(all_results)

    def _get_fine_search_region(self, center_k, min_clusters, max_clusters, coarse_step):
        """Get the fine search region around a promising cluster size"""
        # Search +/- half the coarse step around the center, but at least +/- 3
        search_radius = max(3, coarse_step // 2)
        
        region_start = max(min_clusters, center_k - search_radius)
        region_end = min(max_clusters, center_k + search_radius)
        
        return set(range(region_start, region_end + 1))

    def _process_optimization_results(self, results_data):
        """Process optimization results and return formatted output"""
        # Find truly optimal solutions
        best_silhouette = max(results_data, key=lambda x: x['silhouette'])
        best_separation = max(results_data, key=lambda x: x['separation'])
        best_cohesion = max(results_data, key=lambda x: x['cohesion'])
        best_composite = max(results_data, key=lambda x: x['composite'])
        
        # Create comprehensive results
        results = {
            'cluster_sizes': [r['k'] for r in results_data],
            'silhouette_scores': [r['silhouette'] for r in results_data],
            'separation_scores': [r['separation'] for r in results_data],
            'cohesion_scores': [r['cohesion'] for r in results_data],
            'composite_scores': [r['composite'] for r in results_data],
            'best_composite_k': best_composite['k'],
            'best_separation_k': best_separation['k'],
            'best_cohesion_k': best_cohesion['k'],
            'best_silhouette_k': best_silhouette['k'],
            'max_composite': best_composite['composite'],
            'max_separation': best_separation['separation'],
            'max_cohesion': best_cohesion['cohesion'],
            'max_silhouette': best_silhouette['silhouette']
        }
        
        print(f"Optimization complete:")
        print(f"  Best composite score: {best_composite['k']} clusters (score: {best_composite['composite']:.3f})")
        print(f"  Best separation: {best_separation['k']} clusters (score: {best_separation['separation']:.3f})")
        print(f"  Best cohesion: {best_cohesion['k']} clusters (score: {best_cohesion['cohesion']:.3f})")
        
        return results, best_composite['k'], best_separation['k']
    
    def _calculate_composite_score(self, silhouette, separation, cohesion):
        """Calculate a composite quality score that balances all metrics"""
        # Normalize scores to 0-1 range and weight them
        # Separation is most important for avoiding redundancy
        # Cohesion is important for meaningful clusters
        # Silhouette provides overall structure validation
        
        # Handle edge cases
        if silhouette < 0:
            silhouette = 0
            
        # Weight the metrics (can be tuned based on your priorities)
        separation_weight = 0.4  # Most important - avoid redundant clusters
        cohesion_weight = 0.4    # Important - meaningful clusters
        silhouette_weight = 0.2  # Validation - overall structure
        
        # Normalize separation (0-2 range) to 0-1
        norm_separation = min(separation / 2.0, 1.0)
        
        # Cohesion and silhouette are already 0-1 range
        norm_cohesion = cohesion
        norm_silhouette = silhouette
        
        composite = (separation_weight * norm_separation + 
                    cohesion_weight * norm_cohesion + 
                    silhouette_weight * norm_silhouette)
        
        return composite
    
    def _calculate_inertia(self, embeddings, labels):
        """Calculate within-cluster sum of squares (inertia)"""
        from sklearn.metrics.pairwise import euclidean_distances
        import numpy as np
        
        total_inertia = 0
        unique_labels = set(labels)
        
        for label in unique_labels:
            cluster_mask = np.array(labels) == label
            cluster_embeddings = embeddings[cluster_mask]
            
            if len(cluster_embeddings) > 1:
                # Calculate centroid
                centroid = np.mean(cluster_embeddings, axis=0)
                # Calculate sum of squared distances to centroid
                distances = euclidean_distances(cluster_embeddings, [centroid])
                total_inertia += np.sum(distances ** 2)
        
        return total_inertia
    
    def _find_elbow_point(self, x_values, y_values):
        """Find elbow point using the knee/elbow detection method"""
        import numpy as np
        
        if len(x_values) < 3:
            return x_values[0] if x_values else 2
            
        # Calculate second derivatives to find the point of maximum curvature
        x_array = np.array(x_values)
        y_array = np.array(y_values)
        
        # Normalize values to 0-1 range for better comparison
        y_norm = (y_array - y_array.min()) / (y_array.max() - y_array.min()) if y_array.max() != y_array.min() else y_array
        
        # Calculate differences
        diffs = np.diff(y_norm)
        
        # Find the point where the rate of decrease slows down the most
        # (biggest change in slope)
        if len(diffs) > 1:
            second_diffs = np.diff(diffs)
            # Find the point where second derivative is maximum (biggest change in slope)
            elbow_idx = np.argmax(second_diffs) + 1  # +1 because we lost an element in diff
            return x_values[elbow_idx] if elbow_idx < len(x_values) else x_values[-1]
        
        return x_values[0]
    
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