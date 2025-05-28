import numpy as np 
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_samples
from sklearn_extra.cluster import KMedoids


def compute_match(Signatures : pd.DataFrame, Signatures_true : pd.DataFrame, index : int) -> pd.DataFrame:
    """
    Compute the cosine similarity between the extracted signatures and the true signatures and return a dataframe with the similarity values.

    Parameters:
    Signatures (pd.DataFrame): Extracted signatures of shape 96 x k
    Signatures_true (pd.DataFrame): True signatures

    Returns:
    match_df (pd.DataFrame): Dataframe with columns 'Extracted', 'True', and 'Similarity' showing the similarity values between the extracted and true signatures.
    """


    if not isinstance(Signatures, pd.DataFrame):
        Signatures = pd.DataFrame(Signatures)
    
    if not isinstance(Signatures_true, pd.DataFrame):
        Signatures_true = pd.DataFrame(Signatures_true)

    
    cost = cosine_similarity(Signatures.T, Signatures_true.T)


    row_ind, col_ind = linear_sum_assignment(1 - cost)


    Signatures_sorted = Signatures.iloc[:, row_ind]
    Signatures_true_sorted = Signatures_true.iloc[:, col_ind]

    simils = np.diag(cosine_similarity(Signatures_sorted.T, Signatures_true_sorted.T))

    match_df = pd.DataFrame({
        f'Extracted_{index}': Signatures_sorted.columns,
        f'True_{index}' : Signatures_true_sorted.columns,
        f'Similarity_{index}' : simils
    })

    mean_similarity = np.mean(match_df[f'Similarity_{index}'])

    return match_df, mean_similarity



def compute_all_matches(all_signatures : np.ndarray, cosmic : pd.DataFrame, k:int = 4) -> pd.DataFrame:
    """
    Compute the cosine similarity between the extracted signatures and the true signatures and return a dataframe with the similarity values.

    Parameters:
    all_signatures (np.ndarray): Extracted signatures of shape 96 x k
    cosmic (pd.DataFrame): True signatures
    k (int): Number of signatures to compare at once

    Returns:
    match_df (pd.DataFrame): Dataframe with columns 'Extracted', 'True', and 'Similarity' showing the similarity values between the extracted and true signatures.
    """
    all_matches = pd.DataFrame()
    for i in range(0, all_signatures.shape[1], k):
    
        signature_block = all_signatures[:, i:i+k]

        match, _ = compute_match(signature_block, cosmic, index = i//k)

        all_matches = pd.concat([all_matches, match.iloc[:,1:]],  axis=1)

    return all_matches


def compute_consensus_signatures(df, avg_threshold=0.0, min_threshold=0.0):
    best_signature_overall = None
    
    for k, group in df.groupby("k"):
        # Stack all muse_signatures for given k
        all_signatures = np.hstack(group["muse_signatures"].values)  # Shape (features, samples)
        
        # Apply K-Medoids clustering
        pam = KMedoids(n_clusters=k, metric='cosine', random_state=42).fit(all_signatures.T)
        labels = pam.labels_
        medoid_indices = pam.medoid_indices_
        consensus_signatures = all_signatures[:, medoid_indices]
        
        # Compute silhouette scores
        silhouette_scores = silhouette_samples(all_signatures.T, labels, metric='cosine')
        
        # Assign each consensus signature to an iteration based on source index
        for idx, medoid in enumerate(medoid_indices):
            # Determine the corresponding iteration and muse_error
            total_samples = 0
            for _, row in group.iterrows():
                if total_samples + row["muse_signatures"].shape[1] > medoid:
                    iteration = row["iteration"]
                    muse_error = row["muse_error"]
                    break
                total_samples += row["muse_signatures"].shape[1]
            
            # Compute silhouette scores for medoid
            medoid_silhouette_scores = silhouette_scores[labels == labels[medoid]]
            avg_silhouette = np.mean(medoid_silhouette_scores)
            min_silhouette = np.min(medoid_silhouette_scores)
            
            # Prune signatures below threshold
            if avg_silhouette >= avg_threshold and min_silhouette >= min_threshold:
                candidate_signature = {
                    "k": k,
                    "iteration": iteration,
                    "muse_error": muse_error,
                    "consensus_signature": consensus_signatures[:, idx],
                    "avg_silhouette": avg_silhouette,
                    "min_silhouette": min_silhouette
                }
                
                # Select the best consensus signature overall based on lowest muse_error
                if best_signature_overall is None or candidate_signature["muse_error"] < best_signature_overall["muse_error"]:
                    best_signature_overall = candidate_signature
    
    return best_signature_overall


def find_best_k(df, avg_threshold=0.0, min_threshold=0.0):
    best_k = None
    lowest_muse_error = float("inf")
    
    for k, group in df.groupby("k"):
        # Stack all muse_signatures for given k
        all_signatures = np.hstack(group["muse_signatures"].values)  # Shape (features, samples)
        
        # Apply K-Medoids clustering
        pam = KMedoids(n_clusters=k, metric='cosine').fit(all_signatures.T)
        labels = pam.labels_
        medoid_indices = pam.medoid_indices_
        
        # Compute silhouette scores
        silhouette_scores = silhouette_samples(all_signatures.T, labels, metric='cosine')
        
        # Track if this k has at least one valid medoid
        valid_k = False
        
        for medoid in medoid_indices:
            # Compute silhouette scores for medoid
            medoid_silhouette_scores = silhouette_scores[labels == labels[medoid]]
            avg_silhouette = np.mean(medoid_silhouette_scores)
            min_silhouette = np.min(medoid_silhouette_scores)
            
            # Prune signatures below threshold
            if avg_silhouette >= avg_threshold and min_silhouette >= min_threshold:
                # Find the minimum muse_error for this k
                muse_errors = group["muse_error"].values
                k_muse_error = np.min(muse_errors)  # Minimum error across all iterations for this k
                
                # Update the best k if this one is better
                if k_muse_error < lowest_muse_error:
                    lowest_muse_error = k_muse_error
                    best_k = k
                    valid_k = True
        
        if valid_k:
            print(f"Valid k={k} found with min muse_error={lowest_muse_error} and silhouette avg={avg_silhouette} and min={min_silhouette}")

    return best_k