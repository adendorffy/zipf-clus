# Copyright (c) 2026 Danel Slabbert, Stellenbosch University. All rights reserved.

"""
edit_graph.py

Builds a normalised edit-distance graph over quantised unit sequences and
partitions it into clusters using the Leiden community-detection algorithm.

Unlike the cosine graph (which operates on continuous embeddings), this script
works on discrete unit sequences produced by ``quantiser.py``.  Edges are
added between pairs of segments whose normalised edit similarity (1 - normalised_edit_distance) is at or above
``--threshold``, with edge weight = normalised edit similarity.

Two early-exit heuristics are applied before the full edit-distance call to
prune obviously distant pairs cheaply:
  1. Skip pairs where either sequence is empty.
  2. Skip pairs where the relative length difference alone already exceeds
     the threshold (a necessary condition for low edit distance).

Edge computation is O(N²) in the number of segments and is parallelised
across row-batches with joblib.  The resulting graph is cached as an igraph
pickle for reuse across runs.

Output:
  A plain-text partition file ``edit_t<threshold>_r<res>_<time>.txt`` written by
  ``utils.write_partition_to_file``.
"""

import time
from pathlib import Path

import editdistance
import igraph as ig
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from quantiser import load_quantised_segments
from utils import batch_indices, partition_graph, write_partition_to_file


# ---------------------------------------------------------------------------
# Edge computation
# ---------------------------------------------------------------------------


def compute_edges_batch(batch, features, lengths, threshold):
    """Compute edges for a batch of source nodes using normalised edit similarity.

    For each source node ``i`` in *batch*, iterates over all nodes ``j > i``
    and adds an edge when the normalised edit similarity between token sequences
    ``features[i]`` and ``features[j]`` is at least *threshold*. A length-ratio
    filter is applied first to prune pairs whose length difference alone
    guarantees the similarity threshold cannot be met.

    Parameters
    ----------
    batch : sequence of int
        Indices of source nodes to process in this batch.
    features : list of sequence
        Quantised token sequences, one per segment.
    lengths : numpy.ndarray of shape (N,), dtype int32
        Pre-computed lengths of each token sequence.
    threshold : float
        Minimum normalised edit similarity
        ``1 - editdistance / max(len_i, len_j)`` for an edge to be retained.
        Also used as the length-ratio filter bound.

    Returns
    -------
    list of tuple(int, int, float)
        Each tuple is ``(src_index, dst_index, similarity)``.
    """

    edges = []
    n = len(features)

    for i in batch:
        fi = features[i]
        li = lengths[i]

        for j in range(i + 1, n):
            lj = lengths[j]
            max_len = li if li >= lj else lj

            if max_len == 0:
                continue

            if abs(li - lj) / max_len > threshold:
                continue

            dist = editdistance.eval(fi, features[j]) / max_len
            similarity = 1.0 - dist
            if similarity >= threshold:
                edges.append((i, j, similarity))

    return edges


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    """Run the full edit-distance graph construction and Leiden partitioning pipeline.

    Loads quantised token sequences from ``args.feature_dir`` and constructs a
    similarity graph where edges connect segment pairs whose normalised edit
    similarity meets ``args.threshold``. If a cached graph pickle exists under
    ``args.output_dir``, it is loaded directly to skip edge computation.
    Otherwise, edges are computed in parallel batches and the graph is
    serialised as a pickle for future reuse.

    The Leiden algorithm is then run via :func:`utils.partition_graph`, which
    tunes the resolution parameter ``gamma`` to target ``args.num_clusters``
    clusters within ``args.tolerance``. The final partition is written to a
    plain-text file named after the threshold, resolution, and total runtime.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments. Expected attributes:

        - ``feature_dir`` (*Path*): directory containing quantised segment files.
        - ``output_dir`` (*Path*): root output directory.
        - ``num_clusters`` (*int*): target number of Leiden clusters.
        - ``threshold`` (*float*): minimum normalised edit similarity for an edge.
        - ``resolution`` (*float*): initial Leiden resolution parameter ``gamma``.
        - ``max_iter`` (*int*): maximum resolution-tuning iterations.
        - ``tolerance`` (*float*): acceptable deviation from ``num_clusters``.
    """

    features, filenames, intervals = load_quantised_segments(args.feature_dir)
    lengths = np.array([len(f) for f in features], dtype=np.int32)

    graph_dir = args.output_dir / "/".join(args.feature_dir.parts[-6:]) / "graphs"
    existing_graphs = list(graph_dir.glob(f"edit_t{args.threshold}_*.pkl"))

    # ------------------------------------------------------------------
    # Path A: load cached igraph pickle
    # ------------------------------------------------------------------
    if existing_graphs:
        print(f"Graph already exists at {existing_graphs[0]}, skipping computation.")
        graph = ig.Graph.Read_Pickle(existing_graphs[0])

        # Recover graph construction time from the filename convention
        # edit_t<threshold>_<time>.pkl  (memory not tracked for this graph).
        stem_parts = existing_graphs[0].stem.split(
            "_"
        )  
        if len(stem_parts) >= 3:  
            graph_time = float(stem_parts[-2])
        else:
            graph_time = float(stem_parts[-1])

    # ------------------------------------------------------------
    # Path B: compute edges from scratch
    # ------------------------------------------------------------------
    else:
        start_time = time.time()
        graph_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving graph to {graph_dir}")

        graph = ig.Graph()
        graph.add_vertices(len(features))

        batches = list(batch_indices(len(features), batch_size=1_000))

        print("Using exact computation for edge creation...")
        batch_edges = Parallel(
            n_jobs=-1,
            backend="loky",
            prefer="processes",
            verbose=0,
        )(
            delayed(compute_edges_batch)(batch, features, lengths, args.threshold)
            for batch in tqdm(batches, desc="Computing edges")
        )

        edges = [edge for batch in batch_edges for edge in batch]

        graph.add_edges([(src, dst) for src, dst, _ in edges])
        graph.es["weight"] = [weight for _, _, weight in edges]

        graph_time = time.time() - start_time
        print(f"Graph construction completed in {graph_time:.2f} seconds.")

        graph_path = graph_dir / f"edit_t{args.threshold}_{graph_time:.2f}.pkl"
        graph.write_pickle(graph_path)
        print(f"Graph saved to {graph_path}")

    # ------------------------------------------------------------------
    # Leiden partitioning
    # ------------------------------------------------------------------

    start_time = time.time()
    partition, resolution = partition_graph(
        graph,
        args.num_clusters,
        resolution=args.resolution,
        max_iterations=args.max_iter,
        tolerance=args.tolerance,
    )
    partition_time = time.time() - start_time
    print(f"Leiden partitioning completed in {partition_time:.2f} seconds.")

    total_time = graph_time + partition_time
    print(f"Total time (graph construction + partitioning): {total_time:.2f} seconds.")

    partition_path = (
        graph_dir.parent
        / f"edit_t{args.threshold}_r{resolution:.4f}_{total_time:.2f}.txt"
    )
    write_partition_to_file(partition, filenames, intervals, partition_path)
    print(f"Partition saved to {partition_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a normalised edit-distance graph and cluster with Leiden."
    )
    parser.add_argument(
        "feature_dir",
        type=Path,
        help="Directory containing quantised segment files (*.npy).",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Root directory under which graphs and partition files will be saved.",
    )
    parser.add_argument(
        "num_clusters",
        type=int,
        help="Target number of clusters for the Leiden resolution tuner.",
    )
    parser.add_argument(
        "--pca_components",
        type=int,
        default=350,
        help="(Unused) Number of PCA components — reserved for future preprocessing.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Maximum normalised edit distance for an edge to be created (default: 0.5).",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.5,
        help="Initial resolution parameter for the Leiden algorithm (default: 0.5).",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=15,
        help="Maximum resolution-tuning iterations for Leiden (default: 15).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5,
        help="Acceptable cluster-count deviation from num_clusters (default: 5).",
    )
    args = parser.parse_args()
    main(args)
