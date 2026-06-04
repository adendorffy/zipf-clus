# Copyright (c) 2026 Danel Slabbert, Stellenbosch University. All rights reserved.

"""
BIRCH.py

Clusters pre-computed segment embeddings using sklearn's BIRCH algorithm
(Balanced Iterative Reducing and Clustering using Hierarchies).

BIRCH builds a compact in-memory Clustering Feature Tree (CF-Tree) from the
data in a single pass, making it substantially faster and more memory-efficient
than k-means for large datasets.  The final cluster count is controlled by the
``num_clusters`` argument, which is passed directly to sklearn's ``Birch``
as ``n_clusters``.

Output:
  A plain-text partition file ``birch_k<k>_<time>.txt`` written by
  ``utils.write_partition_to_file``, listing each cluster with its member
  segments (filename, start_sec, end_sec).
"""

import time
from pathlib import Path

import numpy as np
from sklearn.cluster import Birch

from pooling import load_pooled_features
from utils import convert_labels_to_dict, write_partition_to_file


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    """Run BIRCH clustering on segment embeddings and write the partition.

    Args:
        args: Parsed CLI arguments with attributes:
            - ``feature_dir`` (Path): Directory of pooled segment embeddings.
            - ``output_dir`` (Path): Root directory for output files.
            - ``num_clusters`` (int): Target number of clusters.
    """
    # Load and normalise segment embeddings (shape: N × D).
    features, filenames, intervals = load_pooled_features(args.feature_dir)
    features = features.astype(np.float32)

    # Mirror the feature directory structure under the output root.
    output_dir = args.output_dir / "/".join(args.feature_dir.parts[-6:])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving partition to {output_dir}")

    start_time = time.time()

    print("Running BIRCH with sklearn")
    birch_model = Birch(n_clusters=args.num_clusters)

    # fit_predict builds the CF-Tree and assigns each segment to a cluster
    # in a single pass — no separate fit() + predict() calls needed.
    labels = birch_model.fit_predict(features)
    partition = convert_labels_to_dict(labels)

    total_time = time.time() - start_time
    print(f"BIRCH clustering completed in {total_time:.2f} seconds.")

    # Embed timing in the filename for easy comparison across runs.
    partition_path = output_dir / f"birch_k{args.num_clusters}_{total_time:.2f}.txt"
    write_partition_to_file(partition, filenames, intervals, partition_path)
    print(f"Partition saved to {partition_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cluster segment embeddings with BIRCH."
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
        help="Number of clusters (passed to sklearn Birch as n_clusters).",
    )
    args = parser.parse_args()
    main(args)
