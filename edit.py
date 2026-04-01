import numpy as np
from pathlib import Path
import igraph as ig
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
import time
import editdistance
from quantise import load_quantised_segments
from utils import batch_indices, partition_graph, write_partition_to_file   


def compute_edges_batch(batch_indices, features, threshold):
    
    batch_feat = features[batch_indices]

    edges = []
    for i in range(len(batch_indices)):
        src = batch_indices[i]
        for j in range(src + 1, len(features)):
            dist = editdistance.eval(batch_feat[i], features[j]) / max(len(batch_feat[i]), len(features[j]))
            if dist <= threshold:
                edges.append((src, j, 1 - dist))

    return edges


def main(args):

    features, filenames, intervals = load_quantised_segments(args.feature_dir)

    graph_dir = args.output_dir / "/".join(args.feature_dir.parts[-6:])
    existing_graphs = list(graph_dir.glob(f"edit_t{args.threshold}_*.pkl"))

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

        batch_edges = Parallel(n_jobs=-1)(
            delayed(compute_edges_batch)(batch, features, args.threshold)                
            for batch in tqdm(list(batch_indices(len(features), batch_size=1_000)), desc="Computing edges")
        )
        
        edges = [edge for batch in batch_edges for edge in batch]
        graph.add_edges([(src, dst) for src, dst, _ in edges])
        graph.es["weight"] = [weight for _, _, weight in edges]
        graph_time = time.time() - start_time
        print(f"Graph construction completed in {graph_time:.2f} seconds.")
        graph_path = graph_dir / f"cosine_t{args.threshold}_{graph_time:.2f}.pkl"
        graph.write_pickle(graph_path)
    
    start_time = time.time()
    partition = partition_graph(graph, resolution=args.resolution, max_iterations=args.max_iter, tolerance=args.tolerance)
    partition_time = time.time() - start_time
    print(f"Leiden partitioning completed in {partition_time:.2f} seconds.")
    
    total_time = graph_time + partition_time
    print(f"Total time (graph construction + partitioning): {total_time:.2f} seconds.")
    partition_path = graph_dir / f"edit_t{args.threshold}_r{args.resolution:.4f}_{total_time:.2f}.txt"
    write_partition_to_file(partition, filenames, intervals, partition_path)
    print(f"Partition saved to {partition_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run pooling on features")
    parser.add_argument("feature_dir", type=Path,  help="Directory containing feature .npy files")
    parser.add_argument("output_dir", type=Path, help="Directory to save pooled features")
    parser.add_argument("--pca_components", type=int, default=350, help="Number of PCA components to retain")
    parser.add_argument("--threshold", type=float, default=0.5, help="Cosine distance threshold for edge creation")
    parser.add_argument("--resolution", type=float, default=0.5, help="Resolution parameter for Leiden algorithm")
    parser.add_argument("--max_iter", type=int, default=15, help="Maximum iterations for Leiden algorithm")
    parser.add_argument("--tolerance", type=float, default=5, help="Tolerance for Leiden algorithm convergence")
    args = parser.parse_args()
    
    main(args)