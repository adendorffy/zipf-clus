import numpy as np
from pathlib import Path
import igraph as ig
from tqdm import tqdm
import numpy as np
import time
import faiss
from pooling import load_pooled_features
from utils import partition_graph, write_partition_to_file   


def compute_edges(features, k, threshold):
    
    dim = features.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)

    index.add(features)
    Sim, I = index.search(features, k + 1)

    I = I[:, 1:]
    Sim = Sim[:, 1:]

    sources = np.repeat(np.arange(len(features)), k)
    targets = I.ravel()
    weights = Sim.ravel()

    valid_mask = (targets != -1) & (weights >= threshold)
    sources = sources[valid_mask]
    targets = targets[valid_mask]
    weights = weights[valid_mask] 

    edges = list(zip(sources, targets, weights))

    return edges


def main(args):

    features, filenames, intervals = load_pooled_features(args.feature_dir)

    graph_dir = args.output_dir / "/".join(args.feature_dir.parts[-6:]) / "graphs"
    existing_graphs = list(graph_dir.glob(f"knn_cosine_k{args.k}_t{args.threshold}_*.pkl"))

    if existing_graphs:
        print(f"Graph already exists at {existing_graphs[0]}, skipping computation.")
        graph = ig.Graph.Read_Pickle(existing_graphs[0])
        graph_time = float(existing_graphs[0].stem.split("_")[-1])
    else:
        start_time = time.time()
        graph_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving graph to {graph_dir}")
        graph = ig.Graph()
        graph.add_vertices(len(features))

        edges = compute_edges(features, args.k, args.threshold)

        graph.add_edges([(src, dst) for src, dst, _ in edges])
        graph.es["weight"] = [weight for _, _, weight in edges]
        graph_time = time.time() - start_time
        print(f"Graph construction completed in {graph_time:.2f} seconds.")
        graph_path = graph_dir / f"knn_cosine_k{args.k}_t{args.threshold}_{graph_time:.2f}.pkl"
        graph.write_pickle(graph_path)
    
    start_time = time.time()
    partition, resolution = partition_graph(graph, num_clusters=args.num_clusters, resolution=args.resolution, max_iterations=args.max_iter, tolerance=args.tolerance)
    partition_time = time.time() - start_time
    print(f"Leiden partitioning completed in {partition_time:.2f} seconds.")
    
    total_time = graph_time + partition_time
    print(f"Total time (graph construction + partitioning): {total_time:.2f} seconds.")
    partition_path = graph_dir.parent / f"knn_cosine_k{args.k}_t{args.threshold}_r{resolution:.4f}_{total_time:.2f}.txt"
    write_partition_to_file(partition, filenames, intervals, partition_path)
    print(f"Partition saved to {partition_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run pooling on features")
    parser.add_argument("feature_dir", type=Path,  help="Directory containing feature .npy files")
    parser.add_argument("output_dir", type=Path, help="Directory to save pooled features")
    parser.add_argument("num_clusters", type=int, help="Number of clusters for Leiden algorithm")
    parser.add_argument("k", type=int, help="Number of nearest neighbors for graph construction")
    parser.add_argument("--pca_components", type=int, default=350, help="Number of PCA components to retain")
    parser.add_argument("--threshold", type=float, default=0.5, help="Cosine distance threshold for edge creation")
    parser.add_argument("--resolution", type=float, default=0.5, help="Resolution parameter for Leiden algorithm")
    parser.add_argument("--max_iter", type=int, default=15, help="Maximum iterations for Leiden algorithm")
    parser.add_argument("--tolerance", type=float, default=5, help="Tolerance for Leiden algorithm convergence")
    args = parser.parse_args()
    
    main(args)