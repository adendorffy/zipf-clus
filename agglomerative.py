# Copyright (c) 2026 Your Name
# Licensed under the MIT License – see LICENSE file for details.

"""
agglomerative_clustering.py

Clusters pre-computed segment embeddings using sklearn's AgglomerativeClustering
(bottom-up hierarchical clustering).

Two modes are supported, selected by whether ``--threshold`` is provided:

  - **Fixed-k mode** (default): cluster count is set to ``num_clusters``.
  - **Threshold mode**: the dendrogram is cut at ``--threshold``; the number
    of resulting clusters is determined automatically.

The linkage criterion and distance metric are configurable via CLI flags.
Note that sklearn's AgglomerativeClustering only supports ``ward`` linkage
with the ``euclidean`` metric.

Output:
  A plain-text partition file written by ``utils.write_partition_to_file``,
  listing each cluster with its member segments (filename, start_sec, end_sec).
  The filename encodes the run hyperparameters and wall-clock time for easy
  comparison across experiments.
"""

import time
from pathlib import Path

from sklearn.cluster import AgglomerativeClustering

from pooling import load_pooled_features
from utils import convert_labels_to_dict, write_partition_to_file


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    """Run agglomerative clustering on segment embeddings and write the partition.

    Args:
        args: Parsed CLI arguments with attributes:
            - ``feature_dir`` (Path): Directory of pooled segment embeddings.
            - ``output_dir`` (Path): Root directory for output files.
            - ``num_clusters`` (int): Target cluster count (fixed-k mode).
            - ``linkage_criterion`` (str): Linkage criterion
              (``"average"``, ``"complete"``, or ``"ward"``).
            - ``distance_metric`` (str): Distance metric
              (``"euclidean"`` or ``"cosine"``).
            - ``threshold`` (float | None): Distance threshold for cutting the
              dendrogram.  If ``None``, fixed-k mode is used.
    """
    features, filenames, intervals = load_pooled_features(args.feature_dir)

    output_dir = args.output_dir / "/".join(args.feature_dir.parts[-6:])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving partition to {output_dir}")

    start_time = time.time()
    print("Running Agglomerative clustering with sklearn")

    if args.threshold is not None:
        # Threshold mode: n_clusters must be None when distance_threshold is set.
        agg_model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=args.threshold,
            metric=args.distance_metric,
            linkage=args.linkage_criterion,
        )
    else:
        # Fixed-k mode: cut the dendrogram to produce exactly num_clusters.
        agg_model = AgglomerativeClustering(
            n_clusters=args.num_clusters,
            metric=args.distance_metric,
            linkage=args.linkage_criterion,
            compute_full_tree="auto",
            compute_distances=False,
        )

    labels = agg_model.fit_predict(features)
    partition = convert_labels_to_dict(labels)

    total_time = time.time() - start_time
    print(f"Agglomerative clustering completed in {total_time:.2f} seconds.")

    if args.threshold is not None:
        partition_path = (
            output_dir / f"agglomerative_th{args.threshold}_{args.distance_metric}"
            f"_{args.linkage_criterion}_{total_time:.2f}.txt"
        )
    else:
        partition_path = (
            output_dir / f"agglomerative_k{args.num_clusters}_{args.distance_metric}"
            f"_{args.linkage_criterion}_{total_time:.2f}.txt"
        )

    write_partition_to_file(partition, filenames, intervals, partition_path)
    print(f"Partition saved to {partition_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cluster segment embeddings with agglomerative (hierarchical) clustering."
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
        help="Number of clusters (fixed-k mode; ignored if --threshold is set).",
    )
    parser.add_argument(
        "--linkage_criterion",
        type=str,
        choices=["average", "complete", "ward"],
        default="average",
        help="Linkage criterion for agglomerative clustering (default: average).",
    )
    parser.add_argument(
        "--distance_metric",
        type=str,
        choices=["euclidean", "cosine"],
        default="euclidean",
        help="Distance metric (default: euclidean). Note: ward linkage requires euclidean.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Distance threshold for cutting the dendrogram (threshold mode). "
        "If set, num_clusters is ignored.",
    )
    args = parser.parse_args()
    main(args)
