# Copyright (c) 2026 Danel Slabbert, Stellenbosch University. All rights reserved.

"""
cosine_graph.py

Builds a cosine-similarity graph over segment embeddings and partitions it
into clusters using the Leiden community-detection algorithm.

Pipeline:
  1. Graph construction – for each pair of segments whose cosine similarity
     is at or above ``--threshold``, an undirected weighted edge is added
     with the cosine similarity as its weight.  Edge computation is
     parallelised across row-batches with joblib.

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
    """Compute edges for a batch of source nodes using cosine similarity.

    Performs a batched matrix multiplication between the embeddings indexed by
    *batch_idx* and the full feature matrix to obtain pairwise cosine
    similarities (assumes rows are L2-normalised). Edges are retained when the
    cosine similarity meets *threshold*; only pairs where ``dst > src`` are
    kept to avoid duplicates.

    Parameters
    ----------
    batch_idx : array-like of int
        Row indices into *features* for the source nodes in this batch.
    features : numpy.ndarray of shape (N, D), dtype float32
        L2-normalised segment embeddings.
    threshold : float
        Minimum cosine similarity for an edge to be retained.

    Returns
    -------
    list of tuple(int, int, float)
        Each tuple is ``(src_index, dst_index, similarity)``.
    """

    batch_feat = features[batch_idx]

    # Dot product of L2-normalised vectors equals cosine similarity.
    sims = batch_feat @ features.T
    src_rows, dst_cols = np.where(sims >= threshold)

    edges = []
    for row, dst in zip(src_rows, dst_cols):
        src = batch_idx[row]
        if dst > src:
            edges.append((src, int(dst), float(sims[row, dst])))

    return edges


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    """Run the full cosine-similarity graph construction and Leiden partitioning pipeline.

    Loads L2-normalised pooled embeddings from ``args.feature_dir`` and constructs
    a similarity graph where edges connect segment pairs whose cosine similarity
    meets ``args.threshold``. Three execution paths are supported depending on
    what is cached under ``args.output_dir``: (A) load a previously serialised
    igraph pickle directly; (B) reconstruct the graph from a cached compressed
    edge list (NPZ), used when the edge count exceeded 10 million at construction
    time; or (C) compute edges from scratch in parallel batches, then serialise
    the result as a pickle or NPZ accordingly.

    The Leiden algorithm is then run via :func:`utils.partition_graph`, which
    tunes the resolution parameter ``gamma`` to target ``args.num_clusters``
    clusters within ``args.tolerance``. The final partition is written to a
    plain-text file named after the threshold, resolution, and total runtime.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments. Expected attributes:

        - ``feature_dir`` (*Path*): directory containing pooled embedding files.
        - ``output_dir`` (*Path*): root output directory.
        - ``num_clusters`` (*int*): target number of Leiden clusters.
        - ``threshold`` (*float*): minimum cosine similarity for an edge.
        - ``n_jobs`` (*int*): number of parallel workers for edge computation.
        - ``batch_size`` (*int*): source nodes per parallel batch.
        - ``resolution`` (*float*): initial Leiden resolution parameter ``gamma``.
        - ``max_iter`` (*int*): maximum resolution-tuning iterations.
        - ``tolerance`` (*float*): acceptable deviation from ``num_clusters``.
        - ``quality_function`` (*str*): Leiden quality function key
          (``"modularity"``, ``"cpm"``, or ``"rb"``).
    """


    features, filenames, intervals = load_pooled_features(args.feature_dir)
    features = features.astype(np.float32)

    graph_dir = args.output_dir / "/".join(args.feature_dir.parts[-6:]) / "graphs"

    existing_graphs = list(graph_dir.glob(f"cosine_t{args.threshold}_*.pkl"))
    existing_edges = list(graph_dir.glob(f"cosine_t{args.threshold}_*_edges.npz"))

    # ------------------------------------------------------------------
    # Path A: load cached igraph pickle
    # ------------------------------------------------------------------
    if existing_graphs:
        print(f"Graph already exists at {existing_graphs[0]}, skipping computation.")
        graph = ig.Graph.Read_Pickle(existing_graphs[0])

        # Recover timing from the filename convention
        # cosine_t<threshold>_<time>.pkl
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
        # cosine_t<threshold>_<time>edges.npz
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

        n_batches = (len(features) + args.batch_size - 1) // args.batch_size
        batch_edges = Parallel(n_jobs=args.n_jobs)(
            delayed(compute_edges_batch)(batch_idx, features, args.threshold)
            for batch_idx in tqdm(
                batch_indices(len(features), args.batch_size),
                total=n_batches,
                desc="Computing edges",
            )
        )

        del features

        edges = [edge for batch in batch_edges for edge in batch]
        del batch_edges

        graph.add_edges([(src, dst) for src, dst, _ in edges])
        graph.es["weight"] = [weight for _, _, weight in edges]

        graph_time = time.time() - start_time

        print(f"Graph construction completed in {graph_time:.2f} seconds.")

        if len(edges) > 10_000_000:
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
        default=0.5, ## tau in paper = 1 - threshold
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
