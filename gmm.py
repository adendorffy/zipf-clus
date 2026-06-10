# Copyright (c) 2026 Danel Slabbert, Stellenbosch University. All rights reserved.

"""
gmm.py

Clusters pre-computed segment embeddings using sklearn's Gaussian Mixture
Model (GMM) with spherical covariance.

Fitting is done via the EM algorithm (up to 1 000 iterations) with a fixed
random seed for reproducibility.  Unlike the Bayesian FBGMM, the cluster
count is fixed to ``num_clusters`` and does not adapt during training.

Output:
  A plain-text partition file ``gmm_k<k>_<time>.txt`` written by
  ``utils.write_partition_to_file``, listing each cluster with its member
  segments (filename, start_sec, end_sec).
"""

import time
from pathlib import Path

from sklearn.mixture import GaussianMixture

from pooling import load_pooled_features
from utils import convert_labels_to_dict, write_partition_to_file


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    """Fit a GMM on segment embeddings and write the partition.

    Args:
        args: Parsed CLI arguments with attributes:
            - ``feature_dir`` (Path): Directory of pooled segment embeddings.
            - ``output_dir`` (Path): Root directory for output files.
            - ``num_clusters`` (int): Number of Gaussian components.
    """
    features, filenames, intervals = load_pooled_features(args.feature_dir)

    output_dir = args.output_dir / "/".join(args.feature_dir.parts[-6:])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving partition to {output_dir}")

    start_time = time.time()
    print("Running GMM with sklearn")

    gmm_model = GaussianMixture(
        n_components=args.num_clusters,
        covariance_type="spherical",
        max_iter=1000,
        random_state=42,
    )

    labels = gmm_model.fit_predict(features)
    partition = convert_labels_to_dict(labels)

    total_time = time.time() - start_time
    print(f"GMM clustering completed in {total_time:.2f} seconds.")

    partition_path = output_dir / f"gmm_k{args.num_clusters}_{total_time:.2f}.txt"
    write_partition_to_file(partition, filenames, intervals, partition_path)
    print(f"Partition saved to {partition_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cluster segment embeddings with a Gaussian Mixture Model (EM)."
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
        help="Number of Gaussian mixture components.",
    )
    args = parser.parse_args()
    main(args)
