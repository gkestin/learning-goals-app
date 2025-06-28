from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist

# STEM context prefix for better embeddings
STEM_CONTEXT = "In STEM education, students should be able to: "

def filter_none_goals(goals):
    """Filter out 'NONE' goals from a list of learning goals"""
    if not goals:
        return []
    return [goal for goal in goals if goal.strip().upper() != "NONE"]

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
        """Prepend STEM context to learning goals for better embeddings and filter NONE goals"""
        # Filter out "NONE" goals first
        filtered_goals = filter_none_goals(learning_goals)
        return [f"{STEM_CONTEXT}{goal}" for goal in filtered_goals]
    
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
    
    def generate_embeddings_with_progress(self, learning_goals, progress_callback=None):
        """Generate embeddings with progress callback for very large datasets"""
        # Check cache first
        cache_key = hash(tuple(learning_goals))
        
        with self._cache_lock:
            if cache_key in self._embedding_cache:
                print("✅ Using cached embeddings")
                if progress_callback:
                    progress_callback(100, "Using cached embeddings")
                return self._embedding_cache[cache_key]
        
        print(f"🔄 Generating embeddings for {len(learning_goals)} goals with progress tracking...")
        start_time = time.time()
        
        prepared_goals = self.prepare_goals_for_embedding(learning_goals)
        n_goals = len(prepared_goals)
        
        # For very large datasets, process in chunks with progress updates
        if n_goals > 1000 and progress_callback:
            chunk_size = 100
            embeddings_list = []
            
            for i in range(0, n_goals, chunk_size):
                chunk = prepared_goals[i:i+chunk_size]
                chunk_embeddings = self.model.encode(
                    chunk,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                embeddings_list.append(chunk_embeddings)
                
                # Calculate and report progress
                progress = min((i + chunk_size) / n_goals * 100, 100)
                if progress_callback:
                    progress_callback(progress, f"Processing embeddings: {min(i+chunk_size, n_goals)}/{n_goals} goals")
            
            embeddings = np.vstack(embeddings_list)
        else:
            # Regular processing for smaller datasets
            embeddings = self.model.encode(
                prepared_goals,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        
        # Cache the result
        with self._cache_lock:
            self._embedding_cache[cache_key] = embeddings
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
        
        # Calculate inertia (WCSS) for elbow plots
        inertia = self._calculate_inertia(embeddings, labels)
        
        elapsed = time.time() - start_time
        print(f"  ✅ k={k}: composite={composite_score:.3f}, silhouette={silhouette_score:.3f}, inertia={inertia:.0f} ({elapsed:.1f}s)")
        
        return {
            'k': k,
            'silhouette': silhouette_score,
            'separation': inter_sep,
            'cohesion': intra_cohesion,
            'composite': composite_score,
            'inertia': inertia
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
    
    def find_optimal_cluster_sizes(self, embeddings, max_clusters=None, min_clusters=2, use_multires=True, use_fast=True, progress_callback=None, use_elbow_detection=True, use_binary_search=False):
        """Find optimal cluster sizes with full performance optimizations and progress tracking"""
        n_goals = len(embeddings)
        
        # Set reasonable bounds
        if max_clusters is None:
            max_clusters = min(n_goals - 1, n_goals // 2 + 50)
        max_clusters = min(max_clusters, n_goals - 1)
        
        if max_clusters < min_clusters:
            return None, None, None
        
        # Use elbow detection by default for better curve mapping
        if use_elbow_detection:
            return self.find_optimal_cluster_sizes_elbow(embeddings, max_clusters, min_clusters, progress_callback)
        
        # Fallback to binary search (for testing/comparison)
        if use_binary_search:
            return self.find_optimal_cluster_sizes_binary(embeddings, max_clusters, min_clusters, progress_callback)
        
        # Fallback to original multi-resolution method
        # Precompute similarity matrices once for all operations
        self._precompute_similarity_matrices(embeddings)
        
        search_range = max_clusters - min_clusters + 1
        optimization_method = "fast multi-resolution" if use_fast else "standard multi-resolution"
        
        if use_multires and search_range > 100:
            print(f"🚀 Using {optimization_method} optimization for {search_range} clusters")
            return self._find_optimal_multires_parallel(embeddings, max_clusters, min_clusters, use_fast, progress_callback)
        else:
            print(f"🚀 Using {optimization_method} exhaustive search for {search_range} clusters")
            return self._find_optimal_exhaustive_parallel(embeddings, max_clusters, min_clusters, use_fast, progress_callback)
    
    def _find_optimal_exhaustive_parallel(self, embeddings, max_clusters, min_clusters, use_fast=True, progress_callback=None):
        """Optimized exhaustive search with parallel processing and progress tracking"""
        cluster_sizes = list(range(min_clusters, max_clusters + 1))
        total_tests = len(cluster_sizes)
        
        print(f"🔄 Testing {total_tests} cluster sizes with parallel processing...")
        start_time = time.time()
        
        if progress_callback:
            progress_callback(f"Starting exhaustive search of {total_tests} cluster sizes...", 0, total_tests, 0)
        
        # Use parallel processing with optimal thread count
        max_workers = min(4, len(cluster_sizes), 8)  # Limit threads to prevent overhead
        
        results_data = []
        completed_count = 0
        
        # Send list of planned tests
        if progress_callback:
            progress_callback(f"planned_tests:{','.join(map(str, cluster_sizes))}", 0, total_tests, 0)
        
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
                    completed_count += 1
                    
                    # Send progress update with completed test result
                    if progress_callback:
                        progress_callback(
                            f"completed:k={result['k']},silhouette={result['silhouette']:.3f},composite={result['composite']:.3f},inertia={result['inertia']:.0f}",
                            completed_count, total_tests, int((completed_count / total_tests) * 100)
                        )
                    
                except Exception as e:
                    k = future_to_k[future]
                    print(f"❌ Error evaluating k={k}: {e}")
                    completed_count += 1
                    if progress_callback:
                        progress_callback(f"error:k={k},error={str(e)}", completed_count, total_tests, int((completed_count / total_tests) * 100))
        
        # Sort results by k for consistency
        results_data.sort(key=lambda x: x['k'])
        
        elapsed = time.time() - start_time
        print(f"⚡ Exhaustive search completed in {elapsed:.2f} seconds")
        
        if progress_callback:
            progress_callback(f"Exhaustive search completed! Tested {total_tests} cluster sizes in {elapsed:.1f}s", total_tests, total_tests, 100)
        
        return self._process_optimization_results(results_data)
    
    def _find_optimal_multires_parallel(self, embeddings, max_clusters, min_clusters, use_fast=True, progress_callback=None):
        """Optimized multi-resolution search with parallel processing and progress tracking"""
        search_range = max_clusters - min_clusters + 1
        coarse_step = max(8, search_range // 15)
        
        coarse_range = list(range(min_clusters, max_clusters + 1, coarse_step))
        if coarse_range[-1] != max_clusters:
            coarse_range.append(max_clusters)
        
        # Phase 1: coarse search
        coarse_tests = len(coarse_range)
        
        print(f"🔄 Phase 1: Parallel coarse search ({coarse_tests} points)")
        start_time = time.time()
        
        completed_count = 0
        
        # Send planned coarse tests
        if progress_callback:
            progress_callback(f"phase:coarse,planned_tests:{','.join(map(str, coarse_range))}", 0, 0, 0)
        
        # Run coarse search first
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
                    completed_count += 1
                    
                    # Send coarse search progress with coarse totals
                    if progress_callback:
                        progress_callback(
                            f"phase:coarse,completed:k={result['k']},silhouette={result['silhouette']:.3f},composite={result['composite']:.3f},inertia={result['inertia']:.0f}",
                            completed_count, coarse_tests, int((completed_count / coarse_tests) * 100)
                        )
                        
                except Exception as e:
                    k = future_to_k[future]
                    print(f"❌ Error in coarse search k={k}: {e}")
                    completed_count += 1
        
        # Find promising regions after coarse search
        best_coarse_composite = max(coarse_results, key=lambda x: x['composite'])
        best_coarse_separation = max(coarse_results, key=lambda x: x['separation'])
        best_coarse_cohesion = max(coarse_results, key=lambda x: x['cohesion'])
        
        # Calculate actual fine search regions
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
        
        # Now we know the actual total tests - send initial progress with real numbers
        actual_fine_tests = len(fine_search_regions)
        actual_total_tests = coarse_tests + actual_fine_tests
        
        if progress_callback:
            progress_callback(f"Phase 1: Completed coarse search ({coarse_tests} tests)", coarse_tests, actual_total_tests, int((coarse_tests / actual_total_tests) * 100))
        
        print(f"🔄 Phase 2: Parallel fine search ({actual_fine_tests} points)")
        if progress_callback:
            progress_callback(
                f"phase:fine,planned_tests:{','.join(map(str, sorted(fine_search_regions)))}", 
                completed_count, actual_total_tests, int((completed_count / actual_total_tests) * 100)
            )
        
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
                        completed_count += 1
                        
                        if progress_callback:
                            progress_callback(
                                f"completed:k={result['k']},silhouette={result['silhouette']:.3f},composite={result['composite']:.3f},inertia={result['inertia']:.0f}",
                                completed_count, actual_total_tests, int((completed_count / actual_total_tests) * 100)
                            )
                            
                    except Exception as e:
                        k = future_to_k[future]
                        print(f"❌ Error in fine search k={k}: {e}")
                        completed_count += 1
        
        # Combine and sort results
        all_results = coarse_results + fine_results
        all_results.sort(key=lambda x: x['k'])
        
        elapsed = time.time() - start_time
        print(f"⚡ Multi-resolution search completed in {elapsed:.2f} seconds")
        print(f"📊 Tested {len(all_results)} points vs {search_range} exhaustive ({100*(1-len(all_results)/search_range):.1f}% reduction)")
        
        if progress_callback:
            progress_callback(f"Multi-resolution search completed! Tested {actual_total_tests} cluster sizes in {elapsed:.1f}s", actual_total_tests, actual_total_tests, 100)
        
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

    def find_optimal_cluster_sizes_binary(self, embeddings, max_clusters=None, min_clusters=2, progress_callback=None):
        """Find optimal cluster sizes using efficient binary search approach"""
        n_goals = len(embeddings)
        
        # Set reasonable bounds
        if max_clusters is None:
            max_clusters = min(n_goals - 1, n_goals // 2 + 50)
        max_clusters = min(max_clusters, n_goals - 1)
        
        if max_clusters < min_clusters:
            return None, None, None
        
        # Precompute similarity matrices once for all operations
        self._precompute_similarity_matrices(embeddings)
        
        print(f"🎯 Using efficient binary search optimization for {n_goals} goals")
        print(f"   Search range: {min_clusters} to {max_clusters} clusters")
        
        if progress_callback:
            progress_callback(f"Starting efficient binary search optimization...", 0, 0, 0)
        
        # Binary search optimization
        results_data = []
        test_count = 0
        
        # Phase 1: Initial broad sampling (thirds)
        initial_points = [
            min_clusters + int((max_clusters - min_clusters) * 1/3),  # 1/3 point
            min_clusters + int((max_clusters - min_clusters) * 2/3),  # 2/3 point
        ]
        
        # Always include min and max for boundary reference
        test_points = [min_clusters, max_clusters] + initial_points
        test_points = sorted(list(set(test_points)))  # Remove duplicates and sort
        
        if progress_callback:
            progress_callback(f"binary_search:initial,planned_tests:{','.join(map(str, test_points))}", 0, 0, 0)
        
        print(f"🔄 Phase 1: Testing initial points: {test_points}")
        
        # Test initial points
        for k in test_points:
            result = self._evaluate_cluster_size_fast(embeddings, k)
            results_data.append(result)
            test_count += 1
            
            if progress_callback:
                progress_callback(
                    f"binary_search:initial,completed:k={result['k']},silhouette={result['silhouette']:.3f},composite={result['composite']:.3f},inertia={result['inertia']:.0f}",
                    test_count, 0, 0
                )
        
        # Find the best initial point
        best_result = max(results_data, key=lambda x: x['composite'])
        best_k = best_result['k']
        
        print(f"   📊 Best initial point: k={best_k} (composite={best_result['composite']:.3f})")
        
        # Phase 2: Binary search refinement
        search_iterations = 0
        max_iterations = 6  # Limit iterations to prevent infinite loop
        convergence_threshold = 2  # Stop when search range is <= 2
        
        # Determine search boundaries around the best point
        left_bound = min_clusters
        right_bound = max_clusters
        
        # Find the position of best_k and set initial bounds
        for i, result in enumerate(sorted(results_data, key=lambda x: x['k'])):
            if result['k'] == best_k:
                sorted_results = sorted(results_data, key=lambda x: x['k'])
                if i > 0:
                    left_bound = sorted_results[i-1]['k']
                if i < len(sorted_results) - 1:
                    right_bound = sorted_results[i+1]['k']
                break
        
        if progress_callback:
            progress_callback(f"Starting binary search refinement around k={best_k}...", test_count, 0, 0)
        
        while search_iterations < max_iterations and (right_bound - left_bound) > convergence_threshold:
            search_iterations += 1
            
            # Calculate midpoints for binary search
            mid_left = left_bound + int((best_k - left_bound) / 2)
            mid_right = best_k + int((right_bound - best_k) / 2)
            
            # Remove already tested points
            tested_ks = {r['k'] for r in results_data}
            new_test_points = []
            
            if mid_left not in tested_ks and mid_left != best_k and mid_left > left_bound:
                new_test_points.append(mid_left)
            if mid_right not in tested_ks and mid_right != best_k and mid_right < right_bound:
                new_test_points.append(mid_right)
            
            if not new_test_points:
                print(f"   ✅ Converged: No new points to test in range [{left_bound}, {right_bound}]")
                break
            
            print(f"   🔄 Iteration {search_iterations}: Testing points {new_test_points} around k={best_k}")
            
            if progress_callback:
                progress_callback(f"binary_search:refine,planned_tests:{','.join(map(str, new_test_points))}", test_count, 0, 0)
            
            # Test new points
            iteration_results = []
            for k in new_test_points:
                result = self._evaluate_cluster_size_fast(embeddings, k)
                results_data.append(result)
                iteration_results.append(result)
                test_count += 1
                
                if progress_callback:
                    progress_callback(
                        f"binary_search:refine,completed:k={result['k']},silhouette={result['silhouette']:.3f},composite={result['composite']:.3f},inertia={result['inertia']:.0f}",
                        test_count, 0, 0
                    )
            
            # Find the best result from this iteration
            all_candidates = [best_result] + iteration_results
            new_best = max(all_candidates, key=lambda x: x['composite'])
            
            # Update search bounds and best point
            if new_best['k'] != best_k:
                print(f"   📈 Found better point: k={new_best['k']} (composite={new_best['composite']:.3f}) vs k={best_k} (composite={best_result['composite']:.3f})")
                best_result = new_best
                best_k = new_best['k']
                
                # Adjust bounds around new best point
                left_candidates = [r['k'] for r in results_data if r['k'] < best_k]
                right_candidates = [r['k'] for r in results_data if r['k'] > best_k]
                
                left_bound = max(left_candidates) if left_candidates else min_clusters
                right_bound = min(right_candidates) if right_candidates else max_clusters
            else:
                # Current best is still best, narrow the search range
                print(f"   ✅ Current best k={best_k} remains optimal, narrowing search...")
                if iteration_results:
                    # Adjust bounds based on tested points
                    left_candidates = [r['k'] for r in iteration_results if r['k'] < best_k]
                    right_candidates = [r['k'] for r in iteration_results if r['k'] > best_k]
                    
                    if left_candidates:
                        left_bound = max(left_candidates)
                    if right_candidates:
                        right_bound = min(right_candidates)
        
        print(f"⚡ Binary search completed in {search_iterations} iterations, {test_count} total tests")
        
        if progress_callback:
            progress_callback(f"Binary search completed! Found optimal in {test_count} tests vs traditional ~{max_clusters-min_clusters} tests", test_count, test_count, 100)
        
        return self._process_optimization_results(results_data)

    def find_optimal_cluster_sizes_elbow(self, embeddings, max_clusters=None, min_clusters=2, progress_callback=None):
        """Find optimal cluster sizes using interactive elbow detection approach"""
        n_goals = len(embeddings)
        
        # Set reasonable bounds
        if max_clusters is None:
            max_clusters = min(n_goals - 1, n_goals // 2 + 50)
        max_clusters = min(max_clusters, n_goals - 1)
        
        if max_clusters < min_clusters:
            return None, None, None
        
        # Precompute similarity matrices once for all operations
        self._precompute_similarity_matrices(embeddings)
        
        print(f"🎯 Using interactive elbow detection for {n_goals} goals")
        print(f"   Search range: {min_clusters} to {max_clusters} clusters")
        
        if progress_callback:
            progress_callback(f"Starting minimal elbow exploration...", 0, 0, 0)
        
        results_data = []
        test_count = 0
        search_range = max_clusters - min_clusters + 1
        
        # Phase 1: Minimal initial sampling (3-4 points max)
        # Start with just the bounds and 1-2 strategic middle points
        initial_points = [min_clusters, max_clusters]
        
        # Add 1-2 strategic middle points
        if search_range > 10:
            mid1 = min_clusters + int(search_range * 0.3)  # ~30% point
            mid2 = min_clusters + int(search_range * 0.7)  # ~70% point
            initial_points.extend([mid1, mid2])
        elif search_range > 4:
            mid = min_clusters + int(search_range * 0.5)  # 50% point
            initial_points.append(mid)
        
        initial_points = sorted(list(set(initial_points)))  # Remove duplicates and sort
        
        if progress_callback:
            progress_callback(f"elbow_search:initial,planned_tests:{','.join(map(str, initial_points))}", 0, 0, 0)
        
        print(f"🔄 Phase 1: Quick sampling with {len(initial_points)} points: {initial_points}")
        print(f"   Each test may take several minutes with {n_goals} goals...")
        
        # Test initial points
        for k in initial_points:
            result = self._evaluate_cluster_size_fast(embeddings, k)
            results_data.append(result)
            test_count += 1
            
            if progress_callback:
                progress_callback(
                    f"elbow_search:initial,completed:k={result['k']},silhouette={result['silhouette']:.3f},composite={result['composite']:.3f},inertia={result['inertia']:.0f}",
                    test_count, 0, 0
                )
        
        print(f"⚡ Initial sampling completed with {test_count} tests")
        print(f"📊 Ready for user interaction - they can add more points as needed")
        
        if progress_callback:
            progress_callback(f"Initial curve ready! Add more points around areas of interest, or select a point to use.", test_count, test_count, 100)
        
        return self._process_optimization_results(results_data)
    
    def add_elbow_refinement_points(self, embeddings, existing_results, target_region_center, region_radius=None, max_clusters=None, min_clusters=2, progress_callback=None):
        """Add more points around a target region for elbow refinement"""
        n_goals = len(embeddings)
        
        if max_clusters is None:
            max_clusters = min(n_goals - 1, n_goals // 2 + 50)
        max_clusters = min(max_clusters, n_goals - 1)
        
        if region_radius is None:
            search_range = max_clusters - min_clusters + 1
            region_radius = max(3, search_range // 20)  # Adaptive radius
        
        # Define refinement region around target
        region_start = max(min_clusters, target_region_center - region_radius)
        region_end = min(max_clusters, target_region_center + region_radius)
        
        # Get already tested points
        tested_ks = {r['k'] for r in existing_results}
        
        # Generate new test points in the region
        new_test_points = []
        for k in range(region_start, region_end + 1):
            if k not in tested_ks:
                new_test_points.append(k)
        
        # Limit to reasonable number of new points
        if len(new_test_points) > 6:
            # Sample evenly across the region
            step = len(new_test_points) // 6
            new_test_points = new_test_points[::step][:6]
        
        if not new_test_points:
            print(f"   No new points to test in region {region_start}-{region_end}")
            return existing_results
        
        print(f"🔄 Adding {len(new_test_points)} refinement points around k={target_region_center}: {new_test_points}")
        
        if progress_callback:
            progress_callback(f"elbow_search:refine,planned_tests:{','.join(map(str, new_test_points))}", 0, 0, 0)
        
        # Test new points
        new_results = []
        for i, k in enumerate(new_test_points):
            result = self._evaluate_cluster_size_fast(embeddings, k)
            new_results.append(result)
            
            if progress_callback:
                progress_callback(
                    f"elbow_search:refine,completed:k={result['k']},silhouette={result['silhouette']:.3f},composite={result['composite']:.3f},inertia={result['inertia']:.0f}",
                    i + 1, len(new_test_points), int(((i + 1) / len(new_test_points)) * 100)
                )
        
        # Combine with existing results
        all_results = existing_results + new_results
        
        print(f"⚡ Refinement completed, now have {len(all_results)} total points")
        
        if progress_callback:
            progress_callback(f"Refinement complete! Added {len(new_test_points)} points around k={target_region_center}", len(new_test_points), len(new_test_points), 100)
        
        return all_results

    def build_hierarchical_tree(self, embeddings, goals, sources, n_levels=8, linkage_method='ward', sampling_method='intelligent'):
        """Build a complete hierarchical clustering tree with multiple levels and quality metrics"""
        start_time = time.time()
        
        print(f"🌳 Building hierarchical tree with {n_levels} levels using {linkage_method} linkage ({sampling_method} sampling)")
        
        # Compute distance matrix and linkage
        print("📊 Computing distance matrix...")
        distances = pdist(embeddings, metric='euclidean')
        
        print(f"🔗 Building linkage matrix with {linkage_method} method...")
        linkage_matrix = linkage(distances, method=linkage_method)
        
        # Build tree structure by cutting at different levels
        tree_result = self._build_tree_levels(
            embeddings, goals, sources, linkage_matrix, n_levels, sampling_method
        )
        
        elapsed = time.time() - start_time
        print(f"⚡ Hierarchical tree built in {elapsed:.2f} seconds")
        
        return {
            'tree_structure': tree_result['nodes'],
            'level_metrics': tree_result['level_metrics'],
            'total_goals': len(goals),
            'n_levels': n_levels,
            'linkage_method': linkage_method,
            'sampling_method': sampling_method,
            'processing_time': round(elapsed, 2)
        }
    
    def _build_tree_levels(self, embeddings, goals, sources, linkage_matrix, n_levels, sampling_method):
        """Build the actual tree structure with nested nodes ensuring complete hierarchical paths"""
        n_samples = len(embeddings)
        
        print(f"📊 Building {n_levels} levels using {sampling_method} sampling")
        
        level_labelings = {}
        level_metrics = {}
        
        if sampling_method == 'natural':
            # Original approach: Use sequential distance thresholds (0, 1, 2, 3...)
            print("  Using natural dendrogram distances (original approach)")
            
            for level in range(n_levels):
                distance_threshold = level  # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9...
                
                try:
                    cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
                    level_labelings[level] = cluster_labels - 1  # Convert to 0-based indexing
                    
                    # Calculate silhouette score for this level
                    n_clusters_at_level = len(set(level_labelings[level]))
                    
                    # Only calculate metrics if we have multiple clusters and valid clustering
                    if n_clusters_at_level > 1 and n_clusters_at_level < n_samples:
                        silhouette_score = self.calculate_cluster_quality_fast(embeddings, level_labelings[level])
                    else:
                        silhouette_score = 0.0  # Default for single cluster or invalid clustering
                    
                    level_metrics[level] = {
                        'clusters': n_clusters_at_level,
                        'silhouette': silhouette_score,
                        'distance_threshold': distance_threshold
                    }
                    
                    print(f"    Level {level} (distance={distance_threshold}): {n_clusters_at_level} clusters, silhouette={silhouette_score:.3f}")
                    
                except Exception as e:
                    print(f"    Warning: Could not create level {level} with distance {distance_threshold}: {e}")
                    # Fallback: use previous level's clustering or create single cluster
                    if level > 0:
                        level_labelings[level] = level_labelings[level - 1]
                        level_metrics[level] = level_metrics[level - 1].copy()
                        level_metrics[level]['distance_threshold'] = distance_threshold
                    else:
                        # Create single cluster as fallback
                        level_labelings[level] = np.zeros(n_samples, dtype=int)
                        level_metrics[level] = {
                            'clusters': 1,
                            'silhouette': 0.0,
                            'distance_threshold': distance_threshold
                        }
        
        else:  # intelligent sampling
            # Intelligent approach: Sample dendrogram to get even cluster count distribution
            print("  Using intelligent dendrogram sampling (gap-filling approach)")
            
            # Get all unique distances from the linkage matrix to understand the structure
            all_distances = np.sort(linkage_matrix[:, 2])
            
            # We want cluster counts roughly distributed across the range
            # Target cluster counts: from n_samples down to 2, distributed across n_levels
            target_max_clusters = min(n_samples, n_samples // 1)  # Start high
            target_min_clusters = max(2, min(10, n_samples // 100))  # End reasonable
            
            # Create a logarithmic distribution of target cluster counts
            if n_levels > 1:
                target_cluster_counts = np.logspace(
                    np.log10(target_max_clusters), 
                    np.log10(target_min_clusters), 
                    n_levels
                ).astype(int)
                # Ensure uniqueness and sort descending (finest to coarsest)
                target_cluster_counts = sorted(list(set(target_cluster_counts)), reverse=True)
                # Ensure we have exactly n_levels by padding if needed
                while len(target_cluster_counts) < n_levels:
                    target_cluster_counts.append(target_cluster_counts[-1])
                target_cluster_counts = target_cluster_counts[:n_levels]
            else:
                target_cluster_counts = [target_min_clusters]
            
            print(f"    Target cluster counts: {target_cluster_counts}")
            
            # Build hierarchical labeling by finding distances that give target cluster counts
            for level in range(n_levels):
                target_clusters = target_cluster_counts[level]
                
                # Find the distance threshold that gives closest to target_clusters
                best_distance = None
                best_cluster_count = None
                best_diff = float('inf')
                
                # Sample different distances to find the one closest to our target
                for distance in all_distances:
                    try:
                        cluster_labels = fcluster(linkage_matrix, distance, criterion='distance')
                        n_clusters_at_distance = len(set(cluster_labels))
                        
                        # Find distance that gives cluster count closest to target
                        diff = abs(n_clusters_at_distance - target_clusters)
                        if diff < best_diff:
                            best_diff = diff
                            best_distance = distance
                            best_cluster_count = n_clusters_at_distance
                            
                    except Exception:
                        continue
                
                # If we couldn't find a good distance, try using maxclust as fallback
                if best_distance is None:
                    try:
                        cluster_labels = fcluster(linkage_matrix, target_clusters, criterion='maxclust')
                        best_cluster_count = len(set(cluster_labels))
                        best_distance = 0  # Placeholder
                    except Exception:
                        # Final fallback: create dummy clustering
                        cluster_labels = np.zeros(n_samples, dtype=int)
                        best_cluster_count = 1
                        best_distance = 0
                else:
                    # Use the best distance we found
                    cluster_labels = fcluster(linkage_matrix, best_distance, criterion='distance')
                    best_cluster_count = len(set(cluster_labels))
                
                level_labelings[level] = cluster_labels - 1  # Convert to 0-based indexing
                
                # Calculate silhouette score for this level
                if best_cluster_count > 1 and best_cluster_count < n_samples:
                    silhouette_score = self.calculate_cluster_quality_fast(embeddings, level_labelings[level])
                else:
                    silhouette_score = 0.0  # Default for single cluster or invalid clustering
                
                level_metrics[level] = {
                    'clusters': best_cluster_count,
                    'silhouette': silhouette_score,
                    'distance_threshold': best_distance,
                    'target_clusters': target_clusters
                }
                
                print(f"    Level {level}: target={target_clusters}, actual={best_cluster_count}, distance={best_distance:.3f}, silhouette={silhouette_score:.3f}")
        
        # Build the complete tree structure ensuring every goal has a path through all levels
        root_nodes = self._build_complete_tree_nodes(
            embeddings, goals, sources, level_labelings, n_levels
        )
        
        # Reverse level_metrics so that frontend Level A (0) = coarsest, Level H = finest
        reversed_level_metrics = {}
        for level in range(n_levels):
            frontend_level = n_levels - 1 - level  # Reverse mapping
            reversed_level_metrics[frontend_level] = level_metrics[level]
        
        return {
            'nodes': root_nodes,
            'level_metrics': reversed_level_metrics
        }
    
    def _build_complete_tree_nodes(self, embeddings, goals, sources, level_labelings, n_levels):
        """Build complete tree nodes ensuring every goal has a full hierarchical path"""
        
        # Start with the coarsest level (fewest clusters) for root nodes
        # This is the LAST level in our level_labelings (n_levels - 1)
        n_levels = len(level_labelings)
        coarsest_level = n_levels - 1
        root_labels = level_labelings[coarsest_level]
        
        # Create root nodes
        root_nodes = []
        unique_root_labels = sorted(set(root_labels))
        
        for i, root_label in enumerate(unique_root_labels):
            # Get indices for this root cluster
            root_indices = [idx for idx, label in enumerate(root_labels) if label == root_label]
            
            # Create root node
            root_node = {
                'id': f"A{i+1}",
                'label': f"A{i+1}",
                'level': 0,
                'size': len(root_indices),
                'indices': root_indices,
                'children': [],
                'goals': None,
                'sources': None,
                'quality': None,
                'representative_goal': ''
            }
            
            # Build complete children ensuring every goal gets a full path
            # Start from coarsest_level - 1 and go down to 0 (finest)
            self._build_complete_children_recursive(
                root_node, embeddings, goals, sources, level_labelings, 
                n_levels, coarsest_level - 1
            )
            
            # Calculate representative goal for root node
            root_node['representative_goal'] = self._find_representative_goal_for_node(
                root_node, goals
            )
            
            root_nodes.append(root_node)
        
        return root_nodes
    
    def _build_complete_children_recursive(self, parent_node, embeddings, goals, sources, 
                                          level_labelings, n_levels, current_level):
        """Recursively build children ensuring every goal gets a complete hierarchical path"""
        
        if current_level < 0:
            # Leaf level - add goals and calculate quality
            parent_node['goals'] = [goals[i] for i in parent_node['indices']]
            parent_node['sources'] = [sources[i] for i in parent_node['indices']]
            
            # Add representative goal (longest one)
            if parent_node['goals']:
                parent_node['representative_goal'] = max(parent_node['goals'], key=len)
            else:
                parent_node['representative_goal'] = ''
            
            # Calculate quality metrics for leaf nodes
            if len(parent_node['indices']) > 1:
                leaf_embeddings = embeddings[parent_node['indices']]
                parent_node['quality'] = self._calculate_node_quality(
                    leaf_embeddings, embeddings, parent_node['indices']
                )
            
            return
        
        # Get labels for current level within parent's scope
        current_labels = level_labelings[current_level]
        parent_indices = parent_node['indices']
        
        # Find unique clusters within this parent
        parent_labels = [current_labels[i] for i in parent_indices]
        unique_labels = sorted(set(parent_labels))
        
        if len(unique_labels) <= 1:
            # No subdivision at this level - create a single child with all the same goals
            # This ensures every goal still gets a complete path through all levels
            child_node_label = self._generate_child_label(parent_node['label'], 0)
            
            child_node = {
                'id': child_node_label,
                'label': child_node_label,
                'level': parent_node['level'] + 1,
                'size': len(parent_indices),
                'indices': parent_indices,
                'children': [],
                'goals': None,
                'sources': None,
                'quality': None,
                'representative_goal': ''
            }
            
            # Continue building this child's children
            self._build_complete_children_recursive(
                child_node, embeddings, goals, sources, level_labelings,
                n_levels, current_level - 1
            )
            
            parent_node['children'].append(child_node)
        else:
            # Normal subdivision - create children for each unique label
            for j, child_label in enumerate(unique_labels):
                child_indices = [
                    parent_indices[i] for i, label in enumerate(parent_labels) 
                    if label == child_label
                ]
                
                if len(child_indices) == 0:
                    continue
                
                # Generate hierarchical label
                child_node_label = self._generate_child_label(parent_node['label'], j)
                
                child_node = {
                    'id': child_node_label,
                    'label': child_node_label,
                    'level': parent_node['level'] + 1,
                    'size': len(child_indices),
                    'indices': child_indices,
                    'children': [],
                    'goals': None,
                    'sources': None,
                    'quality': None,
                    'representative_goal': ''
                }
                
                # Recursively build this child's children
                self._build_complete_children_recursive(
                    child_node, embeddings, goals, sources, level_labelings,
                    n_levels, current_level - 1
                )
                
                parent_node['children'].append(child_node)
        
        # Calculate quality for non-leaf nodes and find representative goal
        if parent_node['children']:
            parent_embeddings = embeddings[parent_node['indices']]
            parent_node['quality'] = self._calculate_node_quality(
                parent_embeddings, embeddings, parent_node['indices']
            )
            
            # Find representative goal from all children
            parent_node['representative_goal'] = self._find_representative_goal_for_node(
                parent_node, goals
            )
    
    def _generate_child_label(self, parent_label, child_index):
        """Generate hierarchical labels like A1, A1B1, A1B1C1, etc."""
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        level = len([c for c in parent_label if c.isalpha()])
        
        if level < len(alphabet):
            next_letter = alphabet[level]
            return f"{parent_label}{next_letter}{child_index + 1}"
        else:
            # Fallback for very deep trees
            return f"{parent_label}_{child_index + 1}"
    
    def _calculate_node_quality(self, node_embeddings, all_embeddings, node_indices):
        """Calculate quality metrics for a tree node"""
        if len(node_embeddings) < 2:
            return {
                'silhouette': 0.0,
                'cohesion': 1.0,
                'separation': 0.0
            }
        
        try:
            # Create dummy labels for silhouette calculation
            # All points in this node get label 1, all others get label 0
            labels = [1 if i in node_indices else 0 for i in range(len(all_embeddings))]
            
            # Calculate silhouette score
            silhouette = self.calculate_cluster_quality_fast(all_embeddings, labels)
            
            # Calculate cohesion (average similarity within cluster)
            if len(node_embeddings) > 1:
                similarities = cosine_similarity(node_embeddings)
                # Get upper triangle (excluding diagonal)
                n = len(node_embeddings)
                upper_indices = np.triu_indices(n, k=1)
                cohesion = np.mean(similarities[upper_indices])
            else:
                cohesion = 1.0
            
            # Calculate separation (distance to other clusters)
            if len(node_indices) < len(all_embeddings):
                other_indices = [i for i in range(len(all_embeddings)) if i not in node_indices]
                other_embeddings = all_embeddings[other_indices]
                
                # Calculate average distance between this cluster and others
                node_centroid = np.mean(node_embeddings, axis=0).reshape(1, -1)
                other_centroid = np.mean(other_embeddings, axis=0).reshape(1, -1)
                separation = cosine_distances(node_centroid, other_centroid)[0, 0]
            else:
                separation = 0.0
            
            return {
                'silhouette': round(float(silhouette), 3),
                'cohesion': round(float(cohesion), 3),
                'separation': round(float(separation), 3)
            }
            
        except Exception as e:
            print(f"Warning: Could not calculate quality metrics for node: {e}")
            return {
                'silhouette': 0.0,
                'cohesion': 0.0,
                'separation': 0.0
            }
    
    def _find_representative_goal_for_node(self, node, all_goals):
        """Find a representative goal for a non-leaf node by looking at all goals in its subtree"""
        
        def collect_all_goals_from_node(n):
            if n.get('goals'):
                return n['goals']
            
            all_goals_in_subtree = []
            for child in n.get('children', []):
                all_goals_in_subtree.extend(collect_all_goals_from_node(child))
            return all_goals_in_subtree
        
        subtree_goals = collect_all_goals_from_node(node)
        if subtree_goals:
            # Return the longest goal as representative
            return max(subtree_goals, key=len)
        else:
            # Fallback: use goals from indices
            node_goals = [all_goals[i] for i in node['indices']]
            if node_goals:
                return max(node_goals, key=len)
            return ''
    
    def find_tree_node_by_id(self, tree_structure, node_id):
        """Find a specific node in the tree by its ID"""
        def search_node(node):
            if node['id'] == node_id:
                return node
            for child in node.get('children', []):
                result = search_node(child)
                if result:
                    return result
            return None
        
        for root_node in tree_structure:
            result = search_node(root_node)
            if result:
                return result
        return None
    
    def flatten_tree_node(self, node):
        """Flatten a tree node into a simple cluster format"""
        def collect_goals_recursive(n):
            if n.get('goals'):
                return n['goals'], n['sources']
            
            all_goals = []
            all_sources = []
            for child in n.get('children', []):
                child_goals, child_sources = collect_goals_recursive(child)
                all_goals.extend(child_goals)
                all_sources.extend(child_sources)
            return all_goals, all_sources
        
        goals, sources = collect_goals_recursive(node)
        
        return {
            'id': node['id'],
            'label': node['label'],
            'size': len(goals),
            'goals': goals,
            'sources': sources,
            'quality': node.get('quality', {})
        }

    # HDBSCAN Clustering Methods
    def cluster_hdbscan(self, embeddings, min_cluster_size=3, min_samples=None, alpha=1.0, epsilon=0.0, max_cluster_size=None):
        """
        Perform HDBSCAN clustering with comprehensive parameter control
        
        Parameters:
        - min_cluster_size: minimum points required to form a cluster (default: 3, less strict)
        - min_samples: conservative clustering parameter (defaults to min_cluster_size)  
        - alpha: robustness parameter for outlier detection
        - epsilon: clustering hierarchy cutoff (0.0 = use full hierarchy)
        - max_cluster_size: maximum cluster size (None = no limit)
        """
        try:
            import hdbscan
        except ImportError:
            print("❌ HDBSCAN not installed. Please install it with: pip install hdbscan")
            return np.full(len(embeddings), -1)  # Return all outliers
        
        start_time = time.time()
        
        # Set default min_samples if not provided
        if min_samples is None:
            min_samples = min_cluster_size
        
        # Validate parameters
        if min_cluster_size < 2:
            min_cluster_size = 2
            print("⚠️ Adjusted min_cluster_size to minimum value of 2")
            
        if min_samples < 1:
            min_samples = 1
            print("⚠️ Adjusted min_samples to minimum value of 1")
        
        print(f"🎯 HDBSCAN clustering with min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        
        try:
            # Initialize HDBSCAN with optimized parameters
            hdb = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                alpha=alpha,
                cluster_selection_epsilon=epsilon,
                max_cluster_size=max_cluster_size,
                metric='euclidean',  # Works well with normalized embeddings
                cluster_selection_method='eom',  # Excess of Mass - good for varying cluster sizes
                prediction_data=True  # Enable soft clustering predictions
            )
            
            # Fit and predict cluster labels
            cluster_labels = hdb.fit_predict(embeddings)
            
            # Store the fitted clusterer for access to additional metrics
            self._last_hdbscan_clusterer = hdb
            
            elapsed = time.time() - start_time
            
            # Count clusters and outliers
            unique_labels = set(cluster_labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Exclude noise
            n_outliers = sum(1 for label in cluster_labels if label == -1)
            
            print(f"⚡ HDBSCAN completed in {elapsed:.2f} seconds")
            print(f"📊 Found {n_clusters} clusters and {n_outliers} outliers")
            
            # If all points are outliers and min_cluster_size > 2, suggest reducing it
            if n_clusters == 0 and n_outliers == len(embeddings) and min_cluster_size > 2:
                print(f"⚠️ All points marked as outliers. Consider reducing min_cluster_size (currently {min_cluster_size})")
            
            return cluster_labels
            
        except Exception as e:
            print(f"❌ HDBSCAN clustering failed: {e}")
            # Fallback to noise for all points
            return np.full(len(embeddings), -1)
    
    def get_hdbscan_cluster_stability(self):
        """Get cluster stability scores from the last HDBSCAN run"""
        if not hasattr(self, '_last_hdbscan_clusterer') or self._last_hdbscan_clusterer is None:
            return {}
        
        clusterer = self._last_hdbscan_clusterer
        
        try:
            # Get cluster stability scores
            stability_scores = {}
            
            if hasattr(clusterer, 'cluster_persistence_') and clusterer.cluster_persistence_ is not None:
                # Cluster persistence is a measure of stability
                for i, persistence in enumerate(clusterer.cluster_persistence_):
                    stability_scores[i] = float(persistence)
            
            # Alternative: use condensed tree data if available
            elif hasattr(clusterer, 'condensed_tree_') and clusterer.condensed_tree_ is not None:
                # Extract stability from condensed tree
                condensed_tree = clusterer.condensed_tree_
                # Get clusters and their stability from condensed tree
                clusters = condensed_tree[condensed_tree['child_size'] > 1]
                for row in clusters:
                    cluster_id = row['parent']
                    stability = row['lambda_val']
                    if cluster_id not in stability_scores or stability > stability_scores[cluster_id]:
                        stability_scores[cluster_id] = float(stability)
            
            return stability_scores
            
        except Exception as e:
            print(f"⚠️ Could not extract stability scores: {e}")
            return {}
    
    def get_hdbscan_outlier_scores(self):
        """Get outlier scores for each point from the last HDBSCAN run"""
        if not hasattr(self, '_last_hdbscan_clusterer') or self._last_hdbscan_clusterer is None:
            return np.array([])
        
        clusterer = self._last_hdbscan_clusterer
        
        try:
            if hasattr(clusterer, 'outlier_scores_') and clusterer.outlier_scores_ is not None:
                return clusterer.outlier_scores_
            else:
                # Fallback: use membership probabilities to calculate outlier scores
                if hasattr(clusterer, 'probabilities_') and clusterer.probabilities_ is not None:
                    # Higher probability = less likely to be outlier
                    return 1.0 - clusterer.probabilities_
                else:
                    return np.array([])
        except Exception as e:
            print(f"⚠️ Could not extract outlier scores: {e}")
            return np.array([])
    
    def get_hdbscan_membership_probabilities(self):
        """Get membership probabilities for each point from the last HDBSCAN run"""
        if not hasattr(self, '_last_hdbscan_clusterer') or self._last_hdbscan_clusterer is None:
            return np.array([])
        
        clusterer = self._last_hdbscan_clusterer
        
        try:
            if hasattr(clusterer, 'probabilities_') and clusterer.probabilities_ is not None:
                return clusterer.probabilities_
            else:
                return np.array([])
        except Exception as e:
            print(f"⚠️ Could not extract membership probabilities: {e}")
            return np.array([])
    
    def calculate_hdbscan_quality_metrics(self, embeddings, labels):
        """Calculate comprehensive quality metrics for HDBSCAN clustering"""
        metrics = {}
        
        try:
            # Basic cluster statistics
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_outliers = sum(1 for label in labels if label == -1)
            n_total = len(labels)
            
            metrics['n_clusters'] = n_clusters
            metrics['n_outliers'] = n_outliers
            metrics['outlier_percentage'] = (n_outliers / n_total * 100) if n_total > 0 else 0
            
            # Calculate silhouette score (excluding outliers)
            if n_clusters > 1:
                # Filter out outliers for silhouette calculation
                non_outlier_mask = np.array(labels) != -1
                if np.sum(non_outlier_mask) > n_clusters:  # Need more points than clusters
                    filtered_embeddings = embeddings[non_outlier_mask]
                    filtered_labels = [label for label in labels if label != -1]
                    
                    try:
                        silhouette = self.calculate_cluster_quality_fast(filtered_embeddings, filtered_labels)
                        metrics['silhouette_score'] = silhouette
                    except Exception as e:
                        print(f"⚠️ Silhouette calculation failed: {e}")
                        metrics['silhouette_score'] = 0.0
                else:
                    metrics['silhouette_score'] = 0.0
            else:
                metrics['silhouette_score'] = 0.0
            
            # Calculate separation and cohesion metrics (excluding outliers)
            if n_clusters > 1:
                try:
                    non_outlier_mask = np.array(labels) != -1
                    if np.sum(non_outlier_mask) > n_clusters:
                        filtered_embeddings = embeddings[non_outlier_mask]
                        filtered_labels = [label for label in labels if label != -1]
                        
                        separation, cohesion = self.calculate_cluster_separation_metrics_fast(
                            filtered_embeddings, filtered_labels
                        )
                        metrics['inter_cluster_separation'] = separation
                        metrics['intra_cluster_cohesion'] = cohesion
                    else:
                        metrics['inter_cluster_separation'] = 0.0
                        metrics['intra_cluster_cohesion'] = 0.0
                except Exception as e:
                    print(f"⚠️ Separation metrics calculation failed: {e}")
                    metrics['inter_cluster_separation'] = 0.0
                    metrics['intra_cluster_cohesion'] = 0.0
            else:
                metrics['inter_cluster_separation'] = 0.0
                metrics['intra_cluster_cohesion'] = 0.0
            
            # Get HDBSCAN-specific metrics
            stability_scores = self.get_hdbscan_cluster_stability()
            metrics['cluster_stability'] = stability_scores
            
            if stability_scores:
                metrics['avg_cluster_stability'] = np.mean(list(stability_scores.values()))
                metrics['min_cluster_stability'] = np.min(list(stability_scores.values()))
                metrics['max_cluster_stability'] = np.max(list(stability_scores.values()))
            else:
                metrics['avg_cluster_stability'] = 0.0
                metrics['min_cluster_stability'] = 0.0
                metrics['max_cluster_stability'] = 0.0
            
            # Get outlier scores
            outlier_scores = self.get_hdbscan_outlier_scores()
            if len(outlier_scores) > 0:
                metrics['avg_outlier_score'] = np.mean(outlier_scores)
                metrics['max_outlier_score'] = np.max(outlier_scores)
                
                # Calculate outlier threshold (points with scores above this are likely outliers)
                outlier_threshold = np.percentile(outlier_scores, 95)  # Top 5% as potential outliers
                metrics['outlier_threshold'] = outlier_threshold
            else:
                metrics['avg_outlier_score'] = 0.0
                metrics['max_outlier_score'] = 0.0
                metrics['outlier_threshold'] = 0.0
            
            # Calculate composite score adapted for HDBSCAN
            # For HDBSCAN, we weight outlier detection capability alongside clustering quality
            silhouette_weight = 0.3
            separation_weight = 0.2
            cohesion_weight = 0.3
            stability_weight = 0.2
            
            composite_score = (
                silhouette_weight * metrics['silhouette_score'] +
                separation_weight * min(metrics['inter_cluster_separation'], 1.0) +
                cohesion_weight * metrics['intra_cluster_cohesion'] +
                stability_weight * min(metrics['avg_cluster_stability'], 1.0)
            )
            
            metrics['composite_score'] = composite_score
            
            return metrics
            
        except Exception as e:
            print(f"❌ HDBSCAN quality metrics calculation failed: {e}")
            return {
                'n_clusters': 0,
                'n_outliers': len(labels),
                'outlier_percentage': 100.0,
                'silhouette_score': 0.0,
                'inter_cluster_separation': 0.0,
                'intra_cluster_cohesion': 0.0,
                'cluster_stability': {},
                'avg_cluster_stability': 0.0,
                'composite_score': 0.0
            }
    
    def optimize_hdbscan_parameters(self, embeddings, min_cluster_size_range=None, min_samples_range=None, 
                                  progress_callback=None):
        """
        Optimize HDBSCAN parameters through systematic parameter search
        
        Parameters:
        - min_cluster_size_range: tuple (min, max) for min_cluster_size search
        - min_samples_range: tuple (min, max) for min_samples search  
        - progress_callback: function to report progress
        """
        n_goals = len(embeddings)
        
        # Set reasonable parameter ranges if not provided
        if min_cluster_size_range is None:
            min_size = max(2, n_goals // 200)  # At least 2, be less strict (was // 100)
            max_size = min(30, n_goals // 20)  # At most 30 or 5% of data (was // 10)
            min_cluster_size_range = (min_size, min(max_size, min_size + 15))
        
        if min_samples_range is None:
            # min_samples typically ranges from 1 to min_cluster_size
            min_samples_range = (1, min_cluster_size_range[1])
        
        print(f"🎯 Optimizing HDBSCAN parameters for {n_goals} goals")
        print(f"   min_cluster_size range: {min_cluster_size_range}")
        print(f"   min_samples range: {min_samples_range}")
        
        # Generate parameter combinations
        min_cluster_sizes = list(range(min_cluster_size_range[0], min_cluster_size_range[1] + 1, 2))
        min_samples_values = list(range(min_samples_range[0], min_samples_range[1] + 1, 2))
        
        # Limit combinations to prevent excessive computation
        if len(min_cluster_sizes) > 6:
            step = len(min_cluster_sizes) // 6
            min_cluster_sizes = min_cluster_sizes[::step]
        
        if len(min_samples_values) > 4:
            step = len(min_samples_values) // 4
            min_samples_values = min_samples_values[::step]
        
        param_combinations = []
        for min_cluster_size in min_cluster_sizes:
            for min_samples in min_samples_values:
                # Ensure both values are integers and min_samples <= min_cluster_size
                if (isinstance(min_cluster_size, int) and isinstance(min_samples, int) and 
                    min_samples <= min_cluster_size):
                    param_combinations.append((min_cluster_size, min_samples))
        
        total_tests = len(param_combinations)
        print(f"🔄 Testing {total_tests} parameter combinations...")
        
        if progress_callback:
            progress_callback(f"Starting HDBSCAN parameter optimization with {total_tests} combinations", 0, total_tests, 0)
        
        results = []
        
        for i, (min_cluster_size, min_samples) in enumerate(param_combinations):
            try:
                print(f"  Testing min_cluster_size={min_cluster_size}, min_samples={min_samples}")
                
                # Perform clustering
                labels = self.cluster_hdbscan(embeddings, min_cluster_size, min_samples)
                
                # Calculate quality metrics
                metrics = self.calculate_hdbscan_quality_metrics(embeddings, labels)
                
                result = {
                    'min_cluster_size': min_cluster_size,
                    'min_samples': min_samples,
                    'n_clusters': metrics['n_clusters'],
                    'n_outliers': metrics['n_outliers'],
                    'outlier_percentage': metrics['outlier_percentage'],
                    'silhouette_score': metrics['silhouette_score'],
                    'composite_score': metrics['composite_score'],
                    'avg_cluster_stability': metrics['avg_cluster_stability']
                }
                
                results.append(result)
                
                print(f"    Result: {result['n_clusters']} clusters, {result['n_outliers']} outliers, "
                      f"composite={result['composite_score']:.3f}")
                
                if progress_callback:
                    progress_callback(
                        f"Tested min_cluster_size={min_cluster_size}, min_samples={min_samples}: "
                        f"{result['n_clusters']} clusters, composite={result['composite_score']:.3f}",
                        i + 1, total_tests, int(((i + 1) / total_tests) * 100)
                    )
                    
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                if progress_callback:
                    progress_callback(f"Failed: min_cluster_size={min_cluster_size}, min_samples={min_samples}", 
                                    i + 1, total_tests, int(((i + 1) / total_tests) * 100))
        
        if not results:
            print("❌ No successful parameter combinations found")
            return None, None
        
        # Find best parameters based on different criteria
        best_composite = max(results, key=lambda x: x['composite_score'])
        best_silhouette = max(results, key=lambda x: x['silhouette_score'])
        best_stability = max(results, key=lambda x: x['avg_cluster_stability'])
        
        print(f"🎯 Parameter optimization complete:")
        print(f"  🏆 Best composite: min_cluster_size={best_composite['min_cluster_size']}, "
              f"min_samples={best_composite['min_samples']} (score: {best_composite['composite_score']:.3f})")
        print(f"  🎯 Best silhouette: min_cluster_size={best_silhouette['min_cluster_size']}, "
              f"min_samples={best_silhouette['min_samples']} (score: {best_silhouette['silhouette_score']:.3f})")
        
        if progress_callback:
            progress_callback(f"Parameter optimization complete! Best: min_cluster_size={best_composite['min_cluster_size']}, "
                            f"min_samples={best_composite['min_samples']}", total_tests, total_tests, 100)
        
        return results, best_composite
    
    def format_hdbscan_clusters(self, goals, sources, labels, include_outliers=True):
        """Format HDBSCAN clustering results for display"""
        clusters = {}
        outliers = []
        
        for i, (goal, source, label) in enumerate(zip(goals, sources, labels)):
            if label == -1:  # Outlier
                if include_outliers:
                    outliers.append({
                        'goal': goal,
                        'source': source,
                        'index': i
                    })
            else:
                if label not in clusters:
                    clusters[label] = {
                        'goals': [],
                        'sources': [],
                        'size': 0
                    }
                
                clusters[label]['goals'].append(goal)
                clusters[label]['sources'].append(source)
                clusters[label]['size'] += 1
        
        # Generate representative goals for clusters
        formatted_clusters = []
        for cluster_id, cluster_data in clusters.items():
            representative_goal = max(cluster_data['goals'], key=len)
            
            formatted_clusters.append({
                'id': int(cluster_id),
                'size': cluster_data['size'],
                'goals': cluster_data['goals'],
                'sources': cluster_data['sources'],
                'representative_goal': representative_goal
            })
        
        # Sort clusters by size (largest first)
        formatted_clusters.sort(key=lambda x: x['size'], reverse=True)
        
        return formatted_clusters, outliers
    
    # End of HDBSCAN methods 