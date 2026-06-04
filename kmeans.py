# Copyright (c) 2026 Danel Slabbert, Stellenbosch University. All rights reserved.

"""
kmeans.py

Clusters pre-computed segment embeddings into k acoustic classes using a
two-stage k-means procedure:

  1. Centroid initialisation – sklearn's k-means++ is run on a random subset
     of min(3k, N) segments to produce well-spread initial centroids.  The
     centroids are cached to disk so the expensive initialisation step can be
     skipped on reruns.

  2. Full k-means – FAISS (GPU/CPU) refines the centroids over 15 iterations
     with 3 restarts, using the k-means++ centroids as the starting point.

Wall-clock time and peak memory are tracked separately for the initialisation
and training stages and embedded in the output filenames for easy comparison
across experimental runs.

Output:
  A plain-text partition file ``kmeans_k<k>_<time>.txt`` written by
  ``utils.write_partition_to_file``, listing each cluster with its member
  segments (filename, start_sec, end_sec).
"""

import time
from pathlib import Path

import faiss
import numpy as np
from sklearn.cluster import kmeans_plusplus

from pooling import load_pooled_features
from utils import write_partition_to_file, convert_labels_to_dict

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    """Run two-stage k-means clustering and write the partition to disk.

    Stage 1 – k-means++ initialisation (sklearn, on a random subset):
      Cached to ``<output_dir>/tmp-kmeans/`` so it is only computed once per
      (k, feature_dir) combination.  Timing and peak memory are embedded in
      the cache filename.

    Stage 2 – Full k-means (FAISS):
      Refines the k-means++ centroids over 15 iterations with 3 restarts.
      The nearest centroid for every segment is then looked up with a single
      FAISS index search to produce the final cluster assignments.

    Timing and peak memory for Stage 2 are added to the Stage 1 figures and
    embedded in the output partition filename for reproducibility.
    """

    # Load and normalise segment embeddings (shape: N × D).
    features, filenames, intervals = load_pooled_features(args.feature_dir)

    start_time = time.time()

    # Mirror the feature directory structure under the output root.
    output_dir = args.output_dir / "/".join(args.feature_dir.parts[-6:])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving partition to {output_dir}")

    # Temporary directory for caching k-means++ centroids between runs.
    tmp_path = output_dir / "tmp-kmeans"
    tmp_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1: k-means++ centroid initialisation
    # ------------------------------------------------------------------

    # Initialise the FAISS k-means object (training happens in Stage 2).
    acoustic_model = faiss.Kmeans(
        features.shape[1],  # feature dimensionality
        args.num_clusters,
        niter=15,
        nredo=3,
        verbose=True,
    )
    print("Initializing k-means centroids with sklearn's kmeans++")

    centroid_cache = list(
        tmp_path.rglob(f"kmeans_initial_centroids{args.num_clusters}_*.npy")
    )

    if centroid_cache:
        # Re-use previously computed centroids; recover timing/memory from
        # the filename so they can be added to the Stage-2 totals.
        tmp_save_path = centroid_cache[0]
        print(f"Loading existing initial centroids from {tmp_save_path}")
        initial_centroids = np.load(tmp_save_path)
        if len(tmp_save_path.stem.split("_")) > 3:
            centroid_time = tmp_save_path.stem.split("_")[-2]
        else:
            centroid_time = tmp_save_path.stem.split("_")[-1]
        print(f"Initial centroids loaded in {centroid_time} seconds with ")
        # Reset counters so Stage-2 measurement is clean.
        start_time = time.time()
    else:
        # Sub-sample to at most 3k segments for efficiency; k-means++ on the
        # full corpus is unnecessary as a small subset already gives good spread.
        subset_size = min(args.num_clusters * 3, features.shape[0])
        subset_idx = np.random.choice(features.shape[0], subset_size, replace=False)
        subset_features = features[subset_idx]

        sk_kmeans, _ = kmeans_plusplus(
            subset_features, args.num_clusters, random_state=0
        )
        initial_centroids = sk_kmeans.astype(np.float32)

        centroid_time = time.time() - start_time
        print(f"Initial centroids computed in {centroid_time:.2f} seconds.")

        # Cache centroids with timing in the filename.
        tmp_save_path = (
            tmp_path
            / f"kmeans_initial_centroids{args.num_clusters}_{centroid_time:.2f}.npy"
        )
        np.save(tmp_save_path, initial_centroids)

    # ------------------------------------------------------------------
    # Stage 2: FAISS k-means refinement
    # ------------------------------------------------------------------

    start_time = time.time()

    print("Training k-means model with faiss")
    acoustic_model.train(features, init_centroids=initial_centroids)

    # Assign every segment to its nearest centroid.
    _, Index = acoustic_model.index.search(features, 1)
    labels = Index.flatten()

    partition = convert_labels_to_dict(labels)

    # Aggregate timing and peak memory across both stages.
    total_time = time.time() - start_time + float(centroid_time)

    print(f"K-means clustering completed in {total_time:.2f} seconds.")

    # Embed timing and memory in the filename for easy experiment comparison.
    partition_path = output_dir / f"kmeans_k{args.num_clusters}_{total_time:.2f}.txt"
    write_partition_to_file(partition, filenames, intervals, partition_path)
    print(f"Partition saved to {partition_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cluster segment embeddings with k-means (k-means++ init + FAISS)."
    )
    parser.add_argument(
        "feature_dir",
        type=Path,
        help="Directory containing pooled segment embedding files (*.npy).",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Root directory under which the partition file will be saved.",
    )
    parser.add_argument(
        "num_clusters",
        type=int,
        help="Number of k-means clusters (k).",
    )
    args = parser.parse_args()
    main(args)
