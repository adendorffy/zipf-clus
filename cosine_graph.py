# Copyright (c) 2026 Danel Slabbert, Stellenbosch University. All rights reserved.

"""
cosine_graph.py

Builds a cosine-similarity graph over segment embeddings and partitions it
into clusters using the Leiden community-detection algorithm.

Pipeline:
  1. Graph construction – for each pair of segments whose cosine *distance*
     (1 − similarity) is at or below ``--threshold``, an undirected weighted
     edge is added with the cosine similarity as its weight.  Edge computation
     is parallelised across row-batches with joblib.

  2. Caching – the graph (or its edge list for very large graphs) is written
     to disk so the expensive construction step can be skipped on reruns:
       - ≤ 10 M edges  → igraph pickle  (``cosine_t<t>_<time>.pkl``)
       - >  10 M edges → compressed NPZ (``cosine_t<t>_<time>_edges.npz``)

  3. Leiden partitioning – ``utils.partition_graph`` is called with adaptive
     resolution tuning to steer the partition towards ``num_clusters`` clusters.

Wall-clock time is tracked for each stage and embedded in
all output filenames for reproducibility.

Output:
  A plain-text partition file ``cosine_t<t>_r<res>_<time>.txt`` written
  by ``utils.write_partition_to_file``.
"""

import time
from pathlib import Path

import igraph as ig
import leidenalg as la
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from pooling import load_pooled_features
from utils import batch_indices, partition_graph, write_partition_to_file


# ---------------------------------------------------------------------------
# Edge computation
# ---------------------------------------------------------------------------


def compute_edges_batch(batch_idx, features, threshold):
    """Compute all edges originating from a batch of source nodes.

    For each source node in ``batch_idx``, computes cosine similarities
    against all nodes via a matrix-vector product (features are assumed to be
    L2-normalised, so the dot product equals cosine similarity).  An edge is
    emitted for every target node whose cosine *distance* (1 − similarity) is
    at or below ``threshold``.  Only upper-triangle edges (dst > src) are
    returned to avoid duplicates.

    Args:
        batch_idx (range | list[int]): Indices of source nodes in this batch.
        features (np.ndarray): L2-normalised segment embeddings, shape
            ``(N, D)``, shared across all parallel workers.
        threshold (float): Maximum cosine distance for an edge to be created.

    Returns:
        list[tuple[int, int, float]]: Each tuple is
            ``(src_node_id, dst_node_id, cosine_similarity)``.
    """
    batch_feat = features[batch_idx]

    # Dot product of L2-normalised vectors equals cosine similarity.
    sims = batch_feat @ features.T

    # Find pairs whose cosine distance is within the threshold.
    src_rows, dst_cols = np.where((1 - sims) <= threshold)

    edges = []
    for row, dst in zip(src_rows, dst_cols):
        src = batch_idx[row]
        # Keep only upper-triangle to avoid duplicate edges.
        if dst > src:
            edges.append((src, int(dst), float(sims[row, dst])))

    return edges


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    """Build the cosine graph (or load a cached version) and run Leiden.

    Three execution paths are taken depending on what is already cached:
      - Cached pickle  → load graph directly.
      - Cached NPZ     → reconstruct graph from edge list.
      - No cache       → compute edges in parallel, build graph, cache result.

    Timing and peak memory metadata are recovered from cached filenames so
    they can be aggregated into the final output filename.

    Args:
        args: Parsed CLI arguments (see ``__main__`` block for full list).
    """

    # Load and L2-normalise segment embeddings (shape: N × D).
    features, filenames, intervals = load_pooled_features(args.feature_dir)
    features = features.astype(np.float32)

    graph_dir = args.output_dir / "/".join(args.feature_dir.parts[-6:]) / "graphs"

    # Check for previously cached graph artefacts.
    existing_graphs = list(graph_dir.glob(f"cosine_t{args.threshold}_*.pkl"))
    existing_edges = list(graph_dir.glob(f"cosine_t{args.threshold}_*_edges.npz"))

    # ------------------------------------------------------------------
    # Path A: load cached igraph pickle
    # ------------------------------------------------------------------
    if existing_graphs:
        print(f"Graph already exists at {existing_graphs[0]}, skipping computation.")
        graph = ig.Graph.Read_Pickle(existing_graphs[0])

        # Recover timing from the filename convention
        # cosine_t<threshold>_<time>_<mem>.pkl
        stem_parts = existing_graphs[0].stem.split("_")
        if len(stem_parts) > 3:
            graph_time = float(stem_parts[-2])
        else:
            graph_time = float(stem_parts[-1])

    # ------------------------------------------------------------------
    # Path B: reconstruct graph from cached edge list (NPZ)
    # ------------------------------------------------------------------
    elif existing_edges:
        print(
            f"Edges already exist at {existing_edges[0]}, "
            "loading and constructing graph."
        )
        data = np.load(existing_edges[0], allow_pickle=True)
        edges = data["edges"]
        src = edges[:, 0].astype(np.int32)
        dst = edges[:, 1].astype(np.int32)
        weights = edges[:, 2].astype(np.float32)

        graph = ig.Graph()
        graph.add_vertices(len(features))
        graph.add_edges(list(zip(src, dst)))
        graph.es["weight"] = weights.tolist()

        # Recover timing from the filename convention
        # cosine_t<threshold>_<time>_<mem>_edges.npz
        stem_parts = existing_edges[0].stem.split("_")
        if len(stem_parts) > 4:
            graph_time = float(stem_parts[-3])
        else:
            graph_time = float(stem_parts[-2])

    # ------------------------------------------------------------------
    # Path C: compute edges from scratch
    # ------------------------------------------------------------------
    else:
        start_time = time.time()
        graph_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving graph to {graph_dir}")

        graph = ig.Graph()
        graph.add_vertices(len(features))

        # Parallelise similarity computation across row-batches.
        n_batches = (len(features) + args.batch_size - 1) // args.batch_size
        batch_edges = Parallel(n_jobs=args.n_jobs)(
            delayed(compute_edges_batch)(batch_idx, features, args.threshold)
            for batch_idx in tqdm(
                batch_indices(len(features), args.batch_size),
                total=n_batches,
                desc="Computing edges",
            )
        )

        # Free the feature matrix before building the graph to reduce peak memory.
        del features

        # Flatten the list-of-lists returned by the parallel workers.
        edges = [edge for batch in batch_edges for edge in batch]
        del batch_edges

        graph.add_edges([(src, dst) for src, dst, _ in edges])
        graph.es["weight"] = [weight for _, _, weight in edges]

        graph_time = time.time() - start_time

        print(f"Graph construction completed in {graph_time:.2f} seconds.")

        if len(edges) > 10_000_000:
            # Very dense graphs are too large for pickle; save the edge list
            # as a compressed NPZ instead and reconstruct on reload.
            print(
                f"Warning: Graph has {len(edges):,} edges, "
                "saving edges as NPZ instead of pickle."
            )
            edges_path = (
                graph_dir / f"cosine_t{args.threshold}_{graph_time:.2f}_edges.npz"
            )
            np.savez_compressed(edges_path, edges=edges)
        else:
            graph_path = graph_dir / f"cosine_t{args.threshold}_{graph_time:.2f}.pkl"
            graph.write_pickle(graph_path)

    # ------------------------------------------------------------------
    # Leiden partitioning
    # ------------------------------------------------------------------

    # Map CLI string to leidenalg quality-function class.
    quality_function_map = {
        "modularity": la.ModularityVertexPartition,
        "cpm": la.CPMVertexPartition,
        "rb": la.RBConfigurationVertexPartition,
    }
    if args.quality_function not in quality_function_map:
        raise ValueError(f"Unsupported quality function: {args.quality_function}")
    quality_function = quality_function_map[args.quality_function]

    start_time = time.time()
    partition, resolution = partition_graph(
        graph,
        quality_function=quality_function,
        num_clusters=args.num_clusters,
        resolution=args.resolution,
        max_iterations=args.max_iter,
        tolerance=args.tolerance,
    )
    partition_time = time.time() - start_time
    print(f"Leiden partitioning completed in {partition_time:.2f} seconds.")

    total_time = graph_time + partition_time
    print(f"Total time (graph construction + partitioning): {total_time:.2f} seconds.")

    # Embed threshold, tuned resolution, timing, and memory in the filename.
    partition_path = (
        graph_dir.parent / f"cosine_t{args.threshold}_r{resolution:.4f}"
        f"_{total_time:.2f}.txt"
    )
    write_partition_to_file(partition, filenames, intervals, partition_path)
    print(f"Partition saved to {partition_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a cosine-similarity graph and cluster with Leiden."
    )
    parser.add_argument(
        "feature_dir",
        type=Path,
        help="Directory containing pooled segment embedding files (*.npy).",
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
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel workers for edge computation (default: all cores).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1_000,
        help="Number of source nodes processed per parallel batch (default: 1 000).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Maximum cosine distance for an edge to be created (default: 0.5).",
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
    parser.add_argument(
        "--quality_function",
        type=str,
        default="cpm",
        choices=["modularity", "cpm", "rb"],
        help="Leiden quality function (default: cpm).",
    )
    args = parser.parse_args()
    main(args)
