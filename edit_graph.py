# Copyright (c) 2026 Danel Slabbert, Stellenbosch University. All rights reserved.

"""
edit_graph.py

Builds a normalised edit-distance graph over quantised unit sequences and
partitions it into clusters using the Leiden community-detection algorithm.

Unlike the cosine graph (which operates on continuous embeddings), this script
works on discrete unit sequences produced by ``quantiser.py``.  Edges are
added between pairs of segments whose normalised edit distance is at or below
``--threshold``, with edge weight = 1 − normalised_edit_distance.

Two early-exit heuristics are applied before the full edit-distance call to
prune obviously distant pairs cheaply:
  1. Skip pairs where either sequence is empty.
  2. Skip pairs where the relative length difference alone already exceeds
     the threshold (a necessary condition for low edit distance).

Edge computation is O(N²) in the number of segments and is parallelised
across row-batches with joblib.  The resulting graph is cached as an igraph
pickle for reuse across runs.

Output:
  A plain-text partition file ``edit_t<t>_r<res>_<time>.txt`` written by
  ``utils.write_partition_to_file``.
"""

import time
from pathlib import Path

import editdistance
import igraph as ig
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from quantise import load_quantised_segments
from utils import batch_indices, partition_graph, write_partition_to_file


# ---------------------------------------------------------------------------
# Edge computation
# ---------------------------------------------------------------------------


def compute_edges_batch(batch, features, lengths, threshold):
    """Compute edit-distance edges for a batch of source nodes.

    For each source node ``i`` in ``batch``, compares against all nodes
    ``j > i`` (upper triangle only, to avoid duplicate edges).  Two
    pruning steps are applied before the O(|fi|·|fj|) edit-distance call:
      1. Skip pairs where ``max(|fi|, |fj|) == 0``.
      2. Skip pairs where ``|li − lj| / max(li, lj) > threshold`` — if the
         length ratio alone violates the threshold, edit distance can only be
         worse.

    Edge weight is set to ``1 − normalised_edit_distance`` so that more
    similar segments receive higher weights (consistent with the cosine graph).

    Args:
        batch (range | list[int]): Source node indices for this batch.
        features (list[np.ndarray]): Quantised unit sequences, one per segment.
        lengths (np.ndarray[int32]): Pre-computed sequence lengths,
            shape ``(N,)``.
        threshold (float): Maximum normalised edit distance for an edge.

    Returns:
        list[tuple[int, int, float]]: Each tuple is
            ``(src_node_id, dst_node_id, edge_weight)``.
    """
    edges = []
    n = len(features)

    for i in batch:
        fi = features[i]
        li = lengths[i]

        for j in range(i + 1, n):
            lj = lengths[j]
            max_len = li if li >= lj else lj

            # Skip degenerate empty sequences.
            if max_len == 0:
                continue

            # Length-ratio pruning: if the sequences differ too much in
            # length they cannot be within the edit-distance threshold.
            if abs(li - lj) / max_len > threshold:
                continue

            # Full normalised edit distance.
            dist = editdistance.eval(fi, features[j]) / max_len
            if dist <= threshold:
                edges.append((i, j, 1.0 - dist))

    return edges


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    """Build the edit-distance graph (or load a cached version) and run Leiden.

    Args:
        args: Parsed CLI arguments (see ``__main__`` block for full list).

    Note:
        ``--pca_components`` is accepted by the parser but is not currently
        used in this script.  It may be intended for a future preprocessing
        step applied to the quantised features before graph construction.
    """
    # Load quantised unit sequences and pre-compute their lengths to avoid
    # repeated len() calls inside the inner loop of compute_edges_batch.
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
        )  # was existing_graphs.stem — bug fix
        if len(stem_parts) >= 3:  # was len(... >= 3) — bug fix
            graph_time = float(stem_parts[-2])
        else:
            graph_time = float(stem_parts[-1])

    # ------------------------------------------------------------------
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

        # Flatten the per-batch edge lists into a single list.
        edges = [edge for batch in batch_edges for edge in batch]

        graph.add_edges([(src, dst) for src, dst, _ in edges])
        graph.es["weight"] = [weight for _, _, weight in edges]

        graph_time = time.time() - start_time
        print(f"Graph construction completed in {graph_time:.2f} seconds.")

        # Cache the graph for future runs.
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

    # Embed threshold, tuned resolution, and total time in the filename.
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
