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