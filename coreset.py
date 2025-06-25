import torch 
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset, Subset, WeightedRandomSampler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import johnson_lindenstrauss_min_dim

class Coreset:
    def __init__(self, dataset: Dataset,fraction:float=0.1, seed:int=540986710):
        self.dataset = dataset
        self.n = len(dataset)
        self.coreset_size = int(len(dataset) * fraction)
    

    def select_coreset(self):
        """Base method to select coreset indices and weights. Should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement select_coreset method")

    def get_loader(self, batch_size, num_workers, pin_memory=True):
        idx, sample_weight = self.select_coreset()
        # Create per-sample weights tensor on selected indices
        weights = torch.full((len(idx),), sample_weight, dtype=torch.float)
        sampler = WeightedRandomSampler(weights=weights,
                                        num_samples=len(idx),
                                        replacement=False)
        loader = DataLoader(self.dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            num_workers=num_workers,
                            pin_memory=pin_memory)
        return loader, sample_weight



class UniformRandomCoreset(Coreset):
    def __init__(self, dataset: Dataset, fraction:float=0.1, seed:int=540986710):
        super().__init__(dataset, fraction, seed)
        if seed is not None:
            torch.manual_seed(seed)

    def select_coreset(self):
        """Uniformly select indices without replacement and compute sampling weight"""
        idx = torch.randperm(self.n)[:self.coreset_size].tolist()
        # Each sample in the coreset represents dataset_size / sample_size samples from the full dataset
        weight = self.n / self.coreset_size
        return idx, weight


class SensitivityCoreset(Coreset):
    def __init__(self, dataset, coreset_fraction, k_clusters_fraction, pilot_fraction, model, z=2, lambda_=1.0, epsilon=0.1, seed=None):
        super().__init__(dataset, fraction=coreset_fraction, seed=seed)
        self.model = model.eval()
        self.total_coreset_size = int(coreset_fraction * len(dataset))
        self.k_clusters = int(k_clusters_fraction * self.total_coreset_size)
        self.pilot_size = int(pilot_fraction * self.total_coreset_size)
        self.num_points_to_sample = self.total_coreset_size - self.k_clusters - self.pilot_size
        self.z = z  # Distance exponent (1→k-median, 2→k-means)
        self.lambda_ = lambda_  # Balances loss vs distance in sensitivity
        self.epsilon = epsilon  # Determines pilot_size and theoretical guarantees
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

                # User-friendly check
        if self.pilot_size <= self.k_clusters:
            raise ValueError(
                f"pilot_size ({self.pilot_size}) from pilot_fraction{pilot_fraction} must be greater than k_clusters ({self.k_clusters}),"
                f"evaluated from k_clusters_fraction{k_clusters_fraction}."
                "Please increase pilot_fraction or decrease k_clusters_fraction."
            )

  
    def select_coreset(self):
        """Select coreset indices based on sensitivity"""
        return self.build(self.model)

    def compute_embeddings(self, loader):
        print("In compute_embeddings method")
        embs = []
        with torch.no_grad():
            for x, _ in loader:  # only inputs needed
                x = x.to(next(self.model.parameters()).device)
                # Get embeddings from penultimate layer (before the output head)
                hidden = self.model.get_embeddings(x)  # This gives us the penultimate layer output
                # Average over the sequence length to get a single embedding per sample
                # Or you could use the last token: hidden[:, -1, :]
                hidden = hidden.mean(dim=1)  # Average pooling over sequence dimension to get sequence embedding
                embs.append(hidden.cpu().numpy())
        return np.concatenate(embs, axis=0)

    def compute_losses_on_indices(self, indices, collate_fn=None) -> list[float]:
        """Compute losses for given indices using the model"""
        if not indices:
            return []
        
        device = next(self.model.parameters()).device
        criterion_none = nn.CrossEntropyLoss(reduction='none')
        self.model.eval()
        
        loader = DataLoader(Subset(self.dataset, indices),
                          batch_size=len(indices), 
                          collate_fn=collate_fn, 
                          shuffle=False,
                          pin_memory=True)
        
        losses = []
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                # Apply the full model to get classification logits
                logits = self.model(x_batch)  # final classification logits
                logits = logits[:, -1, :]  # Take the last token for classification
                batch_losses = criterion_none(logits, y_batch).cpu().numpy()
                losses.extend(batch_losses.tolist())
        
        return losses

    def build(self, model: nn.Module, batch_size=32, collate_fn=None):
        """
        Construct a sensitivity-based coreset by estimating per-sample sensitivities using model embeddings, clustering, and loss information.

        This method performs the following steps:
            1. Computes embeddings for the entire dataset using the model's penultimate layer.
            2. Uniformly samples a pilot subset of embeddings and fits a PCA for dimensionality reduction.
            3. Projects all embeddings into the reduced space using the pilot-fitted PCA.
            4. Performs k-means clustering on the reduced pilot embeddings to obtain cluster centroids.
            5. Assigns all data points to clusters and identifies the closest in-dataset point to each centroid (the in-data center).
            6. Computes the model loss for each in-data center.
            7. Estimates a Hölder constant (lambda) for each cluster, quantifying how loss varies with distance from the center.
            8. For each data point, computes its sensitivity as the sum of its cluster center loss and a distance-weighted term.
            9. Normalizes sensitivities to obtain a probability distribution over the dataset.
           10. Samples coreset indices according to these probabilities, and computes per-sample weights for unbiased estimation.
           11. Stores cluster-level sampling probabilities and weights for later analysis.

        Args:
            model (nn.Module): The neural network model used to compute embeddings and losses. Should implement get_embeddings(x).
            batch_size (int, optional): Batch size for embedding computation. Default is 32.
            collate_fn (callable, optional): Optional collate function for DataLoader.

        Returns:
            tuple:
                - sample_idx (list[int]): List of selected coreset indices (with replacement, length = total_coreset_size).
                - weights (torch.Tensor): Tensor of per-sample weights for the selected indices (shape: [total_coreset_size]).

        Side Effects:
            - Sets self.approx_cluster_sampling_probs: dict mapping cluster index to its sampling probability.
            - Sets self.approx_cluster_sampling_weights: dict mapping cluster index to its sampling weight.

        Notes:
            - The pilot_fraction is the fraction of the total coreset size that is used to get the clustering.
            - The method assumes the model has a get_embeddings(x) method returning penultimate layer outputs.
            - The coreset is constructed with replacement, and weights are set for unbiased estimation.
            - The method prints diagnostic information about the clustering, losses, and sampling.
        """
        self._collate_fn = collate_fn
        print("In build method")

        # step 0: compute all embeddings
        full_loader = DataLoader(self.dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        full_embs = self.compute_embeddings(full_loader)
        print(f"Full embeddings shape: {full_embs.shape}")

        # step 1. [heuristic] uniformly sample a small pilot subset that we will use for projection
        pilot_idxs = np.random.choice(self.n, size=self.pilot_size, replace=False)
        pilot_embs = full_embs[pilot_idxs]
        # Fit PCA on pilot embeddings
        pca = PCA(n_components=0.975, svd_solver='auto', random_state=44875)
        X_reduced = pca.fit_transform(pilot_embs)  # Fit on pilot, transform pilot
        X_full_reduced = pca.transform(full_embs)  # Transform full data using pilot-fitted PCA
        print(f"Reduced from {pilot_embs.shape[1]} to {X_reduced.shape[1]} dimensions (95% variance)")
        print(f"X_full_reduced shape: {X_full_reduced.shape}")

        # step 2. k-means clustering on the pilot data
        km = KMeans(n_clusters=self.k_clusters, init='k-means++', n_init=10, random_state=42)
        km.fit(X_reduced)
        out_of_data_centroids = km.cluster_centers_
        X_full_reduced_clustered = km.predict(X_full_reduced)
        points_in_clusters = {i : np.where(X_full_reduced_clustered == i)[0] for i in range(self.k_clusters)}
        in_data_centers = []
        # Compute all distances at once
        dists_to_centroids = np.linalg.norm(X_full_reduced[:, None, :] - out_of_data_centroids[None, :, :], axis=2, ord=self.z)

        # Create a mask to set distances to infinity for points not in each cluster
        masked_dists = dists_to_centroids.copy()
        for i in range(self.k_clusters):
            cluster_mask = (X_full_reduced_clustered == i)
            masked_dists[~cluster_mask, i] = np.inf

        # Find closest point to each centroid in out_of_data_centroids
        in_data_centers = np.argmin(masked_dists, axis=0)
        in_data_centroid_vectors = X_full_reduced[in_data_centers]
        print(f"In-data centers: {in_data_centers}")
        print(f"In-data centroids shape: {in_data_centroid_vectors.shape}")
        
        # Compute losses for the center points
        losses_centers = self.compute_losses_on_indices(in_data_centers.tolist(), self._collate_fn)
        print(f"Losses centers: {losses_centers}")
        
        # Create mapping from cluster index to center loss
        center_losses = dict(zip(range(self.k_clusters), losses_centers))
        print(f"Center losses mapping: {center_losses}")
        
        # Calculate sum of all losses: center_loss * number_of_points_in_cluster
        sum_x_losses = sum([center_losses[i] * len(points_in_clusters[i]) for i in range(self.k_clusters)])
        print(f"Sum of all losses: {sum_x_losses}")
        
        
        # Estimate Hölder constants using optimized method
        cluster_lambdas = self.estimate_holder_lambda_logsample(
            embeddings=X_full_reduced,
            center_losses=center_losses,
            centroids=out_of_data_centroids,
            centroid_to_points=points_in_clusters,
            dists_to_centroids=dists_to_centroids,  # Pass precomputed distances
            p=0.9
        )
        print("Cluster λ and sizes:")
        print(f"{'Cluster':>8} | {'Size':>8} | {'Lambda':>12}")
        print('-' * 34)
        for cluster_idx in range(self.k_clusters):
            size = len(points_in_clusters[cluster_idx])
            lambda_val = cluster_lambdas[cluster_idx]
            print(f"{cluster_idx:8d} | {size:8d} | {lambda_val:12.6f}")
        lambda_arr = np.array([cluster_lambdas[i] for i in range(self.k_clusters)]) # (k,)
        per_point_lambdas = lambda_arr[X_full_reduced_clustered] # (n,) --> lambda value for each point
        per_point_centroid_vector = in_data_centroid_vectors[X_full_reduced_clustered] # (n, d) --> centroid vector for each point
        all_embedding_dists = np.linalg.norm(X_full_reduced - per_point_centroid_vector, axis=1, ord=self.z) ** self.z # (n,) --> distance to centroid for each point
        weighted_costs = per_point_lambdas * all_embedding_dists # (n,) --> weighted cost for each point
        global_weighted_costs = weighted_costs.sum()
        print(f"Global weighted costs: {global_weighted_costs}")
        
        # ---------- NEW: sensitivity, probabilities, sampling ----------
        # s(e) = l̂(e) + Λ_i ‖e - c_i‖^z
        center_losses_arr = np.array([center_losses[i] for i in range(self.k_clusters)])
        per_point_center_loss = center_losses_arr[X_full_reduced_clustered]       # shape (n,)
        sensitivities = per_point_center_loss + per_point_lambdas * all_embedding_dists  # shape (n,)
        denominator = global_weighted_costs + per_point_center_loss.sum()
        probs = sensitivities / denominator                            # p_e
        assert np.isclose(probs.sum(), 1.0), f"PROBABILITY SUM: {probs.sum()}"
        assert np.all(probs >= 0), f"PROBABILITY MIN: {probs.min()}"
        assert np.all(probs <= 1), f"PROBABILITY MAX: {probs.max()}"
        assert probs.shape == (self.n,), f"PROBABILITY SHAPE: {probs.shape}"
        
        # first get the losses for all of the cluster centres that 
        sample_idx = np.random.choice(self.n, size=self.total_coreset_size, replace=True, p=probs) # can optimise by taking cluster centres
        weights   = 1.0 / (self.total_coreset_size * probs[sample_idx])                                  # w(e)

        # Save cluster sampling weights (proportion of points in each cluster)
        self.approx_cluster_sampling_probs = {i : center_losses_arr[i]/denominator for i in range(self.k_clusters)}
        self.approx_cluster_sampling_weights = {k : 1.0 / (self.total_coreset_size * v) for k,v in self.approx_cluster_sampling_probs.items()}

        return sample_idx.tolist(), torch.tensor(weights, dtype=torch.float)

    def estimate_holder_lambda_logsample(
        self, embeddings, center_losses, centroids, centroid_to_points, dists_to_centroids=None, p=0.9
    ) -> dict[int, float]:
        """
        Estimate the Hölder constant Λ for each cluster using a log-sized sample.

        Args:
            embeddings (np.ndarray): (N, D) array of all embeddings.
            center_losses (dict): mapping from cluster indices to their center losses.
            centroids (np.ndarray): (K, D) array of cluster centroids.
            centroid_to_points (dict): cluster index -> list/array of point indices.
            dists_to_centroids (np.ndarray): (N, K) array of distances from all points to all centers (optional).
            p (float): Probability for log-sized sample (default 0.9).

        Returns:
            cluster_lambdas (dict): Λ for each cluster.
        """
        n = self.n
        num_clusters = centroids.shape[0]
        cluster_lambdas = {}
        sample_points_per_cluster = int(np.ceil(-np.log(100 * num_clusters) / np.log(1 - p)))
        print(f"Sampling at most {sample_points_per_cluster} points from all clusters for lambda estimation (excluding center).")

        for i in range(num_clusters):
            in_cluster_indices = np.array(centroid_to_points[i])
            if len(in_cluster_indices) == 0:
                cluster_lambdas[i] = 0.0
                continue

            # Get the center loss for this cluster
            loss_center = center_losses[i]
            
            # Find the actual center point index for this cluster
            # We need to find which point in the cluster is closest to the centroid
            if dists_to_centroids is not None:
                # Use precomputed distances to find the center
                cluster_dists = dists_to_centroids[in_cluster_indices, i]
                center_idx_in_cluster = np.argmin(cluster_dists)
                center_idx = in_cluster_indices[center_idx_in_cluster]
            else:
                # Fallback: compute distances for cluster points
                cluster_embeddings = embeddings[in_cluster_indices]
                cluster_dists = np.linalg.norm(cluster_embeddings - centroids[i], axis=1) ** self.z
                center_idx_in_cluster = np.argmin(cluster_dists)
                center_idx = in_cluster_indices[center_idx_in_cluster]

            # Exclude the center point from sampling to avoid division by zero
            sampling_indices = in_cluster_indices[in_cluster_indices != center_idx]
            
            if len(sampling_indices) == 0:
                cluster_lambdas[i] = 0.0
                continue

            # log-sized sample (but not more than available points)
            
            s_i = min(sample_points_per_cluster, len(sampling_indices))
            
            sample_indices = np.random.choice(sampling_indices, size=s_i, replace=False)

            # Compute losses for all sample points in one batch for efficiency
            sample_losses = self.compute_losses_on_indices(sample_indices.tolist(), self._collate_fn)
            
            # Use precomputed distances if available, otherwise compute them
            if dists_to_centroids is not None:
                # Use the already computed distances
                dists = dists_to_centroids[sample_indices, i]
            else:
                # Fallback: compute distances for sample points
                sample_embeddings = embeddings[sample_indices]
                dists = np.linalg.norm(sample_embeddings - centroids[i], axis=1) ** self.z
            
            r_vals = np.abs(np.array(sample_losses) - loss_center) / (dists + 1E-4)
            cluster_lambdas[i] = max(r_vals) * np.log(n)

        return cluster_lambdas

    def get_cluster_sampling_weights(self, mode='prob', pretty_print=False):
        """
        Return the cluster sampling probabilities or weights.
        mode: 'prob' for probabilities, 'weight' for weights.
        If pretty_print=True, print them in a formatted table.
        """
        if not hasattr(self, 'approx_cluster_sampling_probs') or not hasattr(self, 'approx_cluster_sampling_weights'):
            raise AttributeError("Cluster sampling probabilities/weights not computed yet. Run build() first.")
        if mode == 'prob':
            d = self.approx_cluster_sampling_probs
            label = 'Prob'
        elif mode == 'weight':
            d = self.approx_cluster_sampling_weights
            label = 'Weight'
        else:
            raise ValueError("mode must be 'prob' or 'weight'")
        if pretty_print:
            print(f"{'Cluster':>8} | {label:>12}")
            print('-' * 23)
            for k in sorted(d.keys()):
                print(f"{k:8d} | {d[k]:12.6f}")
        return d




