import numpy as np
from pathlib import Path
import igraph as ig
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
import time
import editdistance
from datasketch import MinHash, MinHashLSH
from quantise import load_quantised_segments
from utils import batch_indices, partition_graph, write_partition_to_file   


def compute_edges_batch(batch, features, lengths, threshold):

    edges = []
    n = len(features)

    for i in batch:
        fi = features[i]
        li = lengths[i]

        for j in range(i + 1, n):
            lj = lengths[j]
            max_len = li if li >= lj else lj

            if abs(li - lj) / max_len > threshold:
                continue

            dist = editdistance.eval(fi, features[j]) / max_len
            if dist <= threshold:
                edges.append((i, j, 1.0 - dist))


    return edges

def compute_edges_lsh_minhash(features, lengths, threshold):

    minhashes = []
    for feat in tqdm(features, desc="Computing MinHash signatures for edit distance approximation"):
        m = MinHash(num_perm=128)

        if len(feat) >= 3:
            shingles = [tuple(feat[i:i+3]) for i in range(len(feat)-2)]
        elif len(feat) == 2:
            shingles = [tuple(feat)]
        elif len(feat) == 1:
            shingles = [(feat[0],)]
        else:
            shingles = []
       
        for s in shingles:
            m.update(str(s).encode('utf8'))
        minhashes.append(m)

    lsh = MinHashLSH(threshold=0.8-threshold, num_perm=128)
    for i, m in enumerate(minhashes):
        lsh.insert(f"node_{i}", m)

    edges = []
    for i, m in tqdm(enumerate(minhashes), desc="Querying LSH for neighbors", total=len(minhashes)):
        neighbors = lsh.query(m)
        for neighbor in neighbors:
            j = int(neighbor.split("_")[1])
            if i >= j:
                continue

            max_len = max(lengths[i], lengths[j])
            if abs(lengths[i] - lengths[j]) / max_len > threshold:
                continue

            dist = editdistance.eval(features[i], features[j]) / max_len
            if dist <= threshold:
                edges.append((i, j, 1 - dist))

    return edges


def main(args):

    features, filenames, intervals = load_quantised_segments(args.feature_dir)
    lengths = np.array([len(f) for f in features], dtype=np.int32)

    graph_dir = args.output_dir / "/".join(args.feature_dir.parts[-6:]) / "graphs"
    if args.use_lsh:
        existing_graphs = list(graph_dir.glob(f"edit_t{args.threshold}_lsh_*.pkl"))
    else:
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

        batches = list(batch_indices(len(features), batch_size=1_000))

        if args.use_lsh:
            print("Using LSH with MinHash for edge computation...")
            edges = compute_edges_lsh_minhash(features, lengths, args.threshold)
        else:
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
        if args.use_lsh:
            graph_path = graph_dir / f"edit_t{args.threshold}_lsh_{graph_time:.2f}.pkl"
        else:
            graph_path = graph_dir / f"edit_t{args.threshold}_{graph_time:.2f}.pkl"
        graph.write_pickle(graph_path)
        print(f"Graph saved to {graph_path}")
    
    start_time = time.time()
    partition, resolution = partition_graph(graph, args.num_clusters, resolution=args.resolution, max_iterations=args.max_iter, tolerance=args.tolerance)
    partition_time = time.time() - start_time
    print(f"Leiden partitioning completed in {partition_time:.2f} seconds.")
    
    total_time = graph_time + partition_time
    print(f"Total time (graph construction + partitioning): {total_time:.2f} seconds.")
    if args.use_lsh:
        partition_path = graph_dir.parent / f"edit_t{args.threshold}_lsh_r{args.resolution:.4f}_{total_time:.2f}.txt"
    else:
        partition_path = graph_dir.parent / f"edit_t{args.threshold}_r{args.resolution:.4f}_{total_time:.2f}.txt"
    write_partition_to_file(partition, filenames, intervals, partition_path)
    print(f"Partition saved to {partition_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run pooling on features")
    parser.add_argument("feature_dir", type=Path,  help="Directory containing feature .npy files")
    parser.add_argument("output_dir", type=Path, help="Directory to save pooled features")
    parser.add_argument("num_clusters", type=int, help="Number of clusters for Leiden algorithm")
    parser.add_argument("--pca_components", type=int, default=350, help="Number of PCA components to retain")
    parser.add_argument("--threshold", type=float, default=0.5, help="Cosine distance threshold for edge creation")
    parser.add_argument("--resolution", type=float, default=0.5, help="Resolution parameter for Leiden algorithm")
    parser.add_argument("--max_iter", type=int, default=15, help="Maximum iterations for Leiden algorithm")
    parser.add_argument("--tolerance", type=float, default=5, help="Tolerance for Leiden algorithm convergence")
    parser.add_argument("--use_lsh", action="store_true", help="Whether to use LSH with MinHash for edge computation")
    args = parser.parse_args()
    
    main(args)