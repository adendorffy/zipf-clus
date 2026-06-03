import numpy as np
from pathlib import Path
import igraph as ig
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
import time
import tracemalloc
from pooling import load_pooled_features
from utils import (
    batch_indices,
    partition_graph,
    write_partition_to_file,
)
import leidenalg as la


def compute_edges_batch(batch_indices, features, threshold):
    batch_feat = features[batch_indices]
    sims = batch_feat @ features.T

    src_rows, dst_cols = np.where((1 - sims) <= threshold)

    edges = []
    for row, dst in zip(src_rows, dst_cols):
        src = batch_indices[row]
        if dst > src:
            edges.append((src, int(dst), float(sims[row, dst])))

    return edges


def main(args):

    tracemalloc.start()
    features, filenames, intervals = load_pooled_features(args.feature_dir)
    features = features.astype(np.float32)

    graph_dir = args.output_dir / "/".join(args.feature_dir.parts[-6:]) / "graphs"
    existing_graphs = list(graph_dir.glob(f"cosine_t{args.threshold}_*.pkl"))
    existing_edges = list(graph_dir.glob(f"cosine_t{args.threshold}_*_edges.npz"))

    if existing_graphs:
        print(f"Graph already exists at {existing_graphs[0]}, skipping computation.")
        graph = ig.Graph.Read_Pickle(existing_graphs[0])
        if len(existing_graphs[0].stem.split("_")) > 3:
            graph_time = float(existing_graphs[0].stem.split("_")[-2])
            graph_peak = float(existing_graphs[0].stem.split("_")[-1])
        else:
            graph_time = float(existing_graphs[0].stem.split("_")[-1])
            graph_peak = 0.0

    elif existing_edges:
        print(
            f"Edges already exist at {existing_edges[0]}, loading and constructing graph."
        )
        data = np.load(existing_edges[0], allow_pickle=True)
        edges = data["edges"]
        src = edges[:, 0].astype(np.int32)
        dst = edges[:, 1].astype(np.int32)
        weights = edges[:, 2].astype(np.float32)
        graph = ig.Graph()
        graph.add_vertices(len(features))
        graph.add_edges([(src, dst) for src, dst in zip(src, dst)])
        graph.es["weight"] = [weight for weight in weights]
        if len(existing_edges[0].stem.split("_")) > 4:
            graph_time = float(existing_edges[0].stem.split("_")[-3])
            graph_peak = float(existing_edges[0].stem.split("_")[-2])
        else:
            graph_time = float(existing_edges[0].stem.split("_")[-2])
            graph_peak = 0.0
    else:
        start_time = time.time()
        tracemalloc.reset_peak()
        graph_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving graph to {graph_dir}")
        graph = ig.Graph()
        graph.add_vertices(len(features))

        batch_edges = Parallel(n_jobs=args.n_jobs)(
            delayed(compute_edges_batch)(batch_idx, features, args.threshold)
            for batch_idx in tqdm(
                batch_indices(len(features), args.batch_size),
                total=(len(features) + args.batch_size - 1) // args.batch_size,
                desc="Computing edges",
            )
        )
        del features
        edges = [edge for batch in batch_edges for edge in batch]
        del batch_edges
        graph.add_edges([(src, dst) for src, dst, _ in edges])
        graph.es["weight"] = [weight for _, _, weight in edges]
        graph_time = time.time() - start_time
        current, graph_peak = tracemalloc.get_traced_memory()
        print(f"Graph construction completed in {graph_time:.2f} seconds.")
        print(f"Peak memory usage: {graph_peak / 10**6:.2f} MB")
        if len(edges) > 10_000_000:
            print(
                f"Warning: Graph has {len(edges):,} edges, which may lead to high memory usage."
            )
            edges_path = (
                graph_dir
                / f"cosine_t{args.threshold}_{graph_time:.2f}_{graph_peak / 10**6:.2f}_edges.npz"
            )
            np.savez_compressed(edges_path, edges=edges)
        else:
            graph_path = (
                graph_dir
                / f"cosine_t{args.threshold}_{graph_time:.2f}_{graph_peak / 10**6:.2f}.pkl"
            )
            graph.write_pickle(graph_path)

    start_time = time.time()
    if args.quality_function == "modularity":
        quality_function = la.ModularityVertexPartition
    elif args.quality_function == "cpm":
        quality_function = la.CPMVertexPartition
    elif args.quality_function == "rb":
        quality_function = la.RBConfigurationVertexPartition
    else:
        raise ValueError(f"Unsupported quality function: {args.quality_function}")
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
    current, peak = tracemalloc.get_traced_memory()
    if graph_peak > peak:
        peak = graph_peak

    total_time = graph_time + partition_time
    print(f"Total time (graph construction + partitioning): {total_time:.2f} seconds.")
    print(f"Peak memory usage: {peak / 10**6:.2f} MB")

    partition_path = (
        graph_dir.parent
        / f"cosine_t{args.threshold}_r{resolution:.4f}_{total_time:.2f}_{peak / 10**6:.2f}.txt"
    )
    write_partition_to_file(partition, filenames, intervals, partition_path)
    print(f"Partition saved to {partition_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run pooling on features")
    parser.add_argument(
        "feature_dir", type=Path, help="Directory containing feature .npy files"
    )
    parser.add_argument(
        "output_dir", type=Path, help="Directory to save pooled features"
    )
    parser.add_argument(
        "num_clusters", type=int, help="Number of clusters for Leiden algorithm"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for edge computation",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1_000, help="Batch size for edge computation"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Cosine distance threshold for edge creation",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.5,
        help="Resolution parameter for Leiden algorithm",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=15,
        help="Maximum iterations for Leiden algorithm",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5,
        help="Tolerance for Leiden algorithm convergence",
    )
    parser.add_argument(
        "--quality_function",
        type=str,
        default="cpm",
        choices=["modularity", "cpm", "rb"],
        help="Quality function for Leiden algorithm",
    )
    args = parser.parse_args()

    main(args)
