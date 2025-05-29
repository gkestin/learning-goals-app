from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# STEM context prefix for better embeddings
STEM_CONTEXT = "In STEM education, students should be able to: "

class LearningGoalsClusteringService:
    def __init__(self):
        # Load local sentence-transformer model
        self.model = SentenceTransformer('all-mpnet-base-v2')
        
        # Performance optimizations - caching and precomputation
        self._embedding_cache = {}
        self._last_embeddings = None
        self._last_similarity_matrix = None
        self._last_distance_matrix = None
        self._cache_lock = threading.Lock()
        
    def prepare_goals_for_embedding(self, learning_goals):
        """Prepend STEM context to learning goals for better embeddings"""
        return [f"{STEM_CONTEXT}{goal}" for goal in learning_goals]
    
    def generate_embeddings(self, learning_goals):
        """Generate embeddings for learning goals with caching and batch processing"""
        # Create cache key from goals
        cache_key = hash(tuple(learning_goals))
        
        with self._cache_lock:
            if cache_key in self._embedding_cache:
                print("✅ Using cached embeddings")
                return self._embedding_cache[cache_key]
        
        print(f"🔄 Generating embeddings for {len(learning_goals)} goals...")
        start_time = time.time()
        
        prepared_goals = self.prepare_goals_for_embedding(learning_goals)
        
        # Use optimized batch processing for better performance
        embeddings = self.model.encode(
            prepared_goals,
            batch_size=32,  # Process in batches for efficiency
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for better cosine similarity
        )
        
        # Cache the result (keep cache size reasonable)
        with self._cache_lock:
            self._embedding_cache[cache_key] = embeddings
            # Limit cache size to prevent memory issues
            if len(self._embedding_cache) > 3:
                oldest_key = next(iter(self._embedding_cache))
                del self._embedding_cache[oldest_key]
        
        elapsed = time.time() - start_time
        print(f"⚡ Embedding generation completed in {elapsed:.2f} seconds")
        return embeddings
    
    def _precompute_similarity_matrices(self, embeddings):
        """Precompute similarity and distance matrices for reuse across multiple clustering operations"""
        embeddings_id = id(embeddings)
        
        # Check if we already have matrices for these embeddings
        if (self._last_embeddings is not None and 
            id(self._last_embeddings) == embeddings_id):
            return self._last_similarity_matrix, self._last_distance_matrix
        
        print("🔄 Precomputing similarity matrices...")
        start_time = time.time()
        
        # Compute similarity matrix (normalized embeddings make this more efficient)
        similarity_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - similarity_matrix  # Convert to distance
        
        # Cache for reuse
        self._last_embeddings = embeddings
        self._last_similarity_matrix = similarity_matrix
        self._last_distance_matrix = distance_matrix
        
        elapsed = time.time() - start_time
        print(f"⚡ Matrix precomputation completed in {elapsed:.2f} seconds")
        return similarity_matrix, distance_matrix
    
    def cluster_fast(self, embeddings, n_clusters):
        """Adaptive clustering: KMeans for large datasets, hierarchical for small datasets"""
        n_samples = len(embeddings)
        start_time = time.time()
        
        if n_samples > 500:
            # Use KMeans for large datasets (much faster, good quality)
            print(f"🚀 Using KMeans clustering for {n_samples} samples")
            
            # Optimize parameters based on cluster count for better performance
            if n_clusters > 500:
                # For very high cluster counts, use fewer iterations
                n_init = 5
                max_iter = 200
                print(f"🔧 High cluster count ({n_clusters}), using optimized parameters: n_init={n_init}, max_iter={max_iter}")
            else:
                n_init = 10
                max_iter = 300
            
            clustering = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=n_init,
                max_iter=max_iter,
                algorithm='lloyd'  # Fastest algorithm
            )
            labels = clustering.fit_predict(embeddings)
        else:
            # Use hierarchical clustering for smaller datasets (better quality)
            print(f"🔗 Using hierarchical clustering for {n_samples} samples")
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
            labels = clustering.fit_predict(embeddings)
        
        elapsed = time.time() - start_time
        print(f"⚡ Clustering completed in {elapsed:.2f} seconds")
        return labels
    
    def cluster_hierarchical(self, embeddings, n_clusters):
        """Original hierarchical clustering method (kept for compatibility)"""
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        labels = clustering.fit_predict(embeddings)
        return labels
    
    def calculate_cluster_quality_fast(self, embeddings, labels):
        """Fast silhouette score calculation with intelligent sampling for large datasets"""
        unique_labels = set(labels)
        if len(unique_labels) <= 1 or len(unique_labels) >= len(labels):
            return 0.0
        
        try:
            n_samples = len(embeddings)
            n_clusters = len(unique_labels)
            
            # For large datasets, use stratified sampling to speed up silhouette calculation
            if n_samples > 1000:
                print(f"📊 Using sampling for silhouette calculation ({n_samples} samples, {n_clusters} clusters)")
                
                # Adaptive sample size based on cluster count
                # Ensure we have enough samples to represent all clusters meaningfully
                min_samples_needed = n_clusters * 2  # At least 2 samples per cluster
                max_sample_size = min(1000, n_samples)  # Cap at 1000 for performance
                sample_size = max(min_samples_needed, min(500, max_sample_size))
                
                # If we need more samples than available, use all samples
                if sample_size >= n_samples * 0.8:
                    print(f"📊 Using full dataset for silhouette ({n_samples} samples)")
                    return silhouette_score(embeddings, labels)
                
                # Stratified sampling to ensure all clusters are represented
                unique_labels_list = list(unique_labels)
                samples_per_cluster = max(2, sample_size // len(unique_labels_list))  # At least 2 per cluster
                
                sampled_indices = []
                for label in unique_labels_list:
                    label_indices = [i for i, l in enumerate(labels) if l == label]
                    if len(label_indices) >= samples_per_cluster:
                        # Sample from this cluster
                        sampled_indices.extend(
                            np.random.choice(label_indices, samples_per_cluster, replace=False)
                        )
                    else:
                        # Use all samples from small clusters
                        sampled_indices.extend(label_indices)
                
                # Ensure we don't exceed our target sample size
                if len(sampled_indices) > sample_size:
                    sampled_indices = np.random.choice(sampled_indices, sample_size, replace=False)
                
                sample_embeddings = embeddings[sampled_indices]
                sample_labels = [labels[i] for i in sampled_indices]
                
                # Verify we still have multiple clusters in sample
                unique_sample_labels = len(set(sample_labels))
                if unique_sample_labels > 1 and len(sample_labels) > unique_sample_labels:
                    print(f"📊 Silhouette calculated on {len(sample_labels)} samples from {unique_sample_labels} clusters")
                    return silhouette_score(sample_embeddings, sample_labels)
                else:
                    print(f"⚠️ Insufficient diversity in sample ({unique_sample_labels} clusters), using full dataset")
                    return silhouette_score(embeddings, labels)
            else:
                return silhouette_score(embeddings, labels)
        except Exception as e:
            print(f"⚠️ Silhouette calculation failed: {e}")
            return 0.0
    
    def calculate_cluster_quality(self, embeddings, labels):
        """Original quality calculation (kept for compatibility)"""
        return self.calculate_cluster_quality_fast(embeddings, labels)
    
    def calculate_cluster_separation_metrics_fast(self, embeddings, labels):
        """Optimized separation metrics using precomputed matrices and vectorized operations"""
        if len(set(labels)) < 2:
            return 0.0, 1.0
        
        # Use precomputed matrices for efficiency
        similarity_matrix, distance_matrix = self._precompute_similarity_matrices(embeddings)
        
        unique_labels = list(set(labels))
        n_clusters = len(unique_labels)
        
        # Vectorized centroid calculation
        centroids = np.zeros((n_clusters, embeddings.shape[1]))
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            centroids[i] = np.mean(embeddings[mask], axis=0)
        
        # Calculate inter-cluster separation efficiently
        if n_clusters > 1:
            centroid_distances = cosine_distances(centroids)
            # Use upper triangle to avoid double counting
            upper_triangle_mask = np.triu(np.ones_like(centroid_distances, dtype=bool), k=1)
            inter_cluster_separation = np.mean(centroid_distances[upper_triangle_mask])
        else:
            inter_cluster_separation = 0.0
        
        # Calculate intra-cluster cohesion efficiently using precomputed similarity matrix
        total_cohesion = 0
        cluster_count = 0
        
        for label in unique_labels:
            cluster_indices = [i for i, l in enumerate(labels) if l == label]
            
            if len(cluster_indices) > 1:
                # Extract similarities for this cluster using precomputed matrix
                cluster_similarities = similarity_matrix[np.ix_(cluster_indices, cluster_indices)]
                
                # Get upper triangle (excluding diagonal) for efficiency
                n = len(cluster_indices)
                upper_triangle_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                
                if np.any(upper_triangle_mask):
                    cluster_cohesion = np.mean(cluster_similarities[upper_triangle_mask])
                    total_cohesion += cluster_cohesion
                    cluster_count += 1
        
        avg_intra_cluster_cohesion = total_cohesion / cluster_count if cluster_count > 0 else 0
        
        return float(inter_cluster_separation), float(avg_intra_cluster_cohesion)
    
    def calculate_cluster_separation_metrics(self, embeddings, labels):
        """Original separation metrics (kept for compatibility)"""
        return self.calculate_cluster_separation_metrics_fast(embeddings, labels)
    
    def _evaluate_cluster_size_fast(self, embeddings, k):
        """Fast evaluation of a single cluster size with all optimizations"""
        start_time = time.time()
        
        # Use fast clustering
        labels = self.cluster_fast(embeddings, k)
        
        # Calculate metrics efficiently
        silhouette_score = self.calculate_cluster_quality_fast(embeddings, labels)
        inter_sep, intra_cohesion = self.calculate_cluster_separation_metrics_fast(embeddings, labels)
        composite_score = self._calculate_composite_score(silhouette_score, inter_sep, intra_cohesion)
        
        elapsed = time.time() - start_time
        print(f"  ✅ k={k}: composite={composite_score:.3f}, silhouette={silhouette_score:.3f} ({elapsed:.1f}s)")
        
        return {
            'k': k,
            'silhouette': silhouette_score,
            'separation': inter_sep,
            'cohesion': intra_cohesion,
            'composite': composite_score
        }
    
    def _evaluate_cluster_size_original(self, embeddings, k):
        """Original evaluation method (kept for compatibility/comparison)"""
        labels = self.cluster_hierarchical(embeddings, k)
        silhouette_score = self.calculate_cluster_quality(embeddings, labels)
        inter_sep, intra_cohesion = self.calculate_cluster_separation_metrics(embeddings, labels)
        composite_score = self._calculate_composite_score(silhouette_score, inter_sep, intra_cohesion)
        
        return {
            'k': k,
            'silhouette': silhouette_score,
            'separation': inter_sep,
            'cohesion': intra_cohesion,
            'composite': composite_score
        }
    
    def find_optimal_cluster_sizes(self, embeddings, max_clusters=None, min_clusters=2, use_multires=True, use_fast=True):
        """Find optimal cluster sizes with full performance optimizations"""
        n_goals = len(embeddings)
        
        # Set reasonable bounds
        if max_clusters is None:
            max_clusters = min(n_goals - 1, n_goals // 2 + 50)
        max_clusters = min(max_clusters, n_goals - 1)
        
        if max_clusters < min_clusters:
            return None, None, None
        
        # Precompute similarity matrices once for all operations
        self._precompute_similarity_matrices(embeddings)
        
        search_range = max_clusters - min_clusters + 1
        optimization_method = "fast multi-resolution" if use_fast else "standard multi-resolution"
        
        if use_multires and search_range > 100:
            print(f"🚀 Using {optimization_method} optimization for {search_range} clusters")
            return self._find_optimal_multires_parallel(embeddings, max_clusters, min_clusters, use_fast)
        else:
            print(f"🚀 Using {optimization_method} exhaustive search for {search_range} clusters")
            return self._find_optimal_exhaustive_parallel(embeddings, max_clusters, min_clusters, use_fast)
    
    def _find_optimal_exhaustive_parallel(self, embeddings, max_clusters, min_clusters, use_fast=True):
        """Optimized exhaustive search with parallel processing"""
        cluster_sizes = list(range(min_clusters, max_clusters + 1))
        
        print(f"🔄 Testing {len(cluster_sizes)} cluster sizes with parallel processing...")
        start_time = time.time()
        
        # Use parallel processing with optimal thread count
        max_workers = min(4, len(cluster_sizes), 8)  # Limit threads to prevent overhead
        
        results_data = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            if use_fast:
                future_to_k = {
                    executor.submit(self._evaluate_cluster_size_fast, embeddings, k): k 
                    for k in cluster_sizes
                }
            else:
                future_to_k = {
                    executor.submit(self._evaluate_cluster_size_original, embeddings, k): k 
                    for k in cluster_sizes
                }
            
            # Collect results as they complete
            for future in as_completed(future_to_k):
                try:
                    result = future.result()
                    results_data.append(result)
                except Exception as e:
                    k = future_to_k[future]
                    print(f"❌ Error evaluating k={k}: {e}")
        
        # Sort results by k for consistency
        results_data.sort(key=lambda x: x['k'])
        
        elapsed = time.time() - start_time
        print(f"⚡ Exhaustive search completed in {elapsed:.2f} seconds")
        return self._process_optimization_results(results_data)
    
    def _find_optimal_multires_parallel(self, embeddings, max_clusters, min_clusters, use_fast=True):
        """Optimized multi-resolution search with parallel processing"""
        search_range = max_clusters - min_clusters + 1
        coarse_step = max(8, search_range // 15)
        
        coarse_range = list(range(min_clusters, max_clusters + 1, coarse_step))
        if coarse_range[-1] != max_clusters:
            coarse_range.append(max_clusters)
        
        print(f"🔄 Phase 1: Parallel coarse search ({len(coarse_range)} points)")
        start_time = time.time()
        
        # Parallel coarse search
        coarse_results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            if use_fast:
                future_to_k = {
                    executor.submit(self._evaluate_cluster_size_fast, embeddings, k): k 
                    for k in coarse_range
                }
            else:
                future_to_k = {
                    executor.submit(self._evaluate_cluster_size_original, embeddings, k): k 
                    for k in coarse_range
                }
            
            for future in as_completed(future_to_k):
                try:
                    result = future.result()
                    coarse_results.append(result)
                except Exception as e:
                    k = future_to_k[future]
                    print(f"❌ Error in coarse search k={k}: {e}")
        
        # Find promising regions
        best_coarse_composite = max(coarse_results, key=lambda x: x['composite'])
        best_coarse_separation = max(coarse_results, key=lambda x: x['separation'])
        best_coarse_cohesion = max(coarse_results, key=lambda x: x['cohesion'])
        
        # Phase 2: Fine search around promising regions
        fine_search_regions = set()
        fine_search_regions.update(self._get_fine_search_region(
            best_coarse_composite['k'], min_clusters, max_clusters, coarse_step))
        
        if best_coarse_separation['k'] != best_coarse_composite['k']:
            fine_search_regions.update(self._get_fine_search_region(
                best_coarse_separation['k'], min_clusters, max_clusters, coarse_step))
        
        if (best_coarse_cohesion['k'] != best_coarse_composite['k'] and 
            best_coarse_cohesion['k'] != best_coarse_separation['k']):
            fine_search_regions.update(self._get_fine_search_region(
                best_coarse_cohesion['k'], min_clusters, max_clusters, coarse_step))
        
        # Remove already tested values
        tested_coarse = {r['k'] for r in coarse_results}
        fine_search_regions = fine_search_regions - tested_coarse
        
        print(f"🔄 Phase 2: Parallel fine search ({len(fine_search_regions)} points)")
        
        # Parallel fine search
        fine_results = []
        if fine_search_regions:
            with ThreadPoolExecutor(max_workers=4) as executor:
                if use_fast:
                    future_to_k = {
                        executor.submit(self._evaluate_cluster_size_fast, embeddings, k): k 
                        for k in sorted(fine_search_regions)
                    }
                else:
                    future_to_k = {
                        executor.submit(self._evaluate_cluster_size_original, embeddings, k): k 
                        for k in sorted(fine_search_regions)
                    }
                
                for future in as_completed(future_to_k):
                    try:
                        result = future.result()
                        fine_results.append(result)
                    except Exception as e:
                        k = future_to_k[future]
                        print(f"❌ Error in fine search k={k}: {e}")
        
        # Combine and sort results
        all_results = coarse_results + fine_results
        all_results.sort(key=lambda x: x['k'])
        
        elapsed = time.time() - start_time
        print(f"⚡ Multi-resolution search completed in {elapsed:.2f} seconds")
        print(f"📊 Tested {len(all_results)} points vs {search_range} exhaustive ({100*(1-len(all_results)/search_range):.1f}% reduction)")
        
        return self._process_optimization_results(all_results)

    def _get_fine_search_region(self, center_k, min_clusters, max_clusters, coarse_step):
        """Get the fine search region around a promising cluster size"""
        search_radius = max(3, coarse_step // 2)
        region_start = max(min_clusters, center_k - search_radius)
        region_end = min(max_clusters, center_k + search_radius)
        return set(range(region_start, region_end + 1))

    def _process_optimization_results(self, results_data):
        """Process optimization results and return formatted output"""
        if not results_data:
            return None, None, None
            
        best_silhouette = max(results_data, key=lambda x: x['silhouette'])
        best_separation = max(results_data, key=lambda x: x['separation'])
        best_cohesion = max(results_data, key=lambda x: x['cohesion'])
        best_composite = max(results_data, key=lambda x: x['composite'])
        
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
        
        print(f"🎯 Optimization complete:")
        print(f"  🏆 Best composite: {best_composite['k']} clusters (score: {best_composite['composite']:.3f})")
        print(f"  🎯 Best separation: {best_separation['k']} clusters (score: {best_separation['separation']:.3f})")
        print(f"  🤝 Best cohesion: {best_cohesion['k']} clusters (score: {best_cohesion['cohesion']:.3f})")
        
        return results, best_composite['k'], best_separation['k']
    
    def _calculate_composite_score(self, silhouette, separation, cohesion):
        """Calculate a composite quality score optimized for STEM content clustering"""
        if silhouette < 0:
            silhouette = 0
            
        # STEM-optimized weights: cohesion is more important than separation for domain-specific content
        separation_weight = 0.3  # Reduced importance - STEM topics naturally closer
        cohesion_weight = 0.5    # Increased importance - meaningful groupings within domain
        silhouette_weight = 0.2  # Validation - overall structure
        
        # STEM-optimized normalization: separation scores of 0.2-0.4 are actually good
        # Normalize separation using STEM-appropriate range (0-0.6 instead of 0-2.0)
        norm_separation = min(separation / 0.6, 1.0)
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