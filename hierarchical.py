import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.neighbors import kneighbors_graph


from collections import defaultdict
import time
import tracemalloc
from pooling import load_pooled_features
from utils import write_partition_to_file

def convert_labels_to_dict(labels):
    partition_dict = defaultdict(list)

    for node_id, cluster_id in tqdm(enumerate(labels), desc="Converting labels to partition dict", unit="nodes", total=len(labels)):
        partition_dict[cluster_id].append(node_id)

    partition = list(partition_dict.values())
    return partition

def main(args):

    tracemalloc.start()
    features, filenames, intervals = load_pooled_features(args.feature_dir)
    features = features.astype(np.float32)
    start_time = time.time()

    output_dir = args.output_dir / "/".join(args.feature_dir.parts[-6:])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving partition to {output_dir}")

    start_time = time.time()
    tracemalloc.reset_peak()
    print("Running hierarchical clustering with sklearn")
    if args.algorithm == "agglomerative":
        if args.threshold is not None:
            agg_model = AgglomerativeClustering(n_clusters=None, distance_threshold=args.threshold, metric=args.distance_metric, linkage='average')
        else:   
            agg_model = AgglomerativeClustering(
                n_clusters=args.num_clusters,
                metric='euclidean',
                linkage='average',        
                compute_full_tree='auto',
                compute_distances=False
            )
        labels = agg_model.fit_predict(features)
    elif args.algorithm == "birch":
        birch_model = Birch(n_clusters=args.num_clusters)
        labels = birch_model.fit_predict(features)
    elif args.algorithm == "both":
        # Stage 1: Birch compresses 86k → ~5k subclusters cheaply
        birch = Birch(n_clusters=None, threshold=0.6)
        birch.fit(features)
        print(f"Birch found {len(birch.subcluster_centers_)} subclusters")
        subcluster_centers = birch.subcluster_centers_

        # Stage 2: Agglomerative on the compressed representation
        agg = AgglomerativeClustering(n_clusters=args.num_clusters, linkage='average', metric=args.distance_metric)
        agg_labels = agg.fit_predict(subcluster_centers)

        # Map back to original points
        labels = agg_labels[birch.predict(features)]

    partition = convert_labels_to_dict(labels)

    total_time = time.time() - start_time
    peak_memory = tracemalloc.get_traced_memory()[1] / 10**6
    print(f"{args.algorithm.capitalize()} clustering completed in {total_time:.2f} seconds.")
    print(f"Peak memory usage during {args.algorithm}: {peak_memory:.2f} MB")
    partition_path = output_dir / f"{args.algorithm}_k{args.num_clusters}_{total_time:.2f}_{peak_memory:.2f}.txt"
    if args.algorithm == "agglomerative":
        partition_path = output_dir / f"{args.algorithm}_k{args.num_clusters}_{args.distance_metric}_{total_time:.2f}_{peak_memory:.2f}_avg.txt"
        if args.threshold is not None:
            partition_path = output_dir / f"{args.algorithm}_th{args.threshold}_{args.distance_metric}_{total_time:.2f}_{peak_memory:.2f}_avg.txt"
    write_partition_to_file(partition, filenames, intervals, partition_path)
    print(f"Partition saved to {partition_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run pooling on features")
    parser.add_argument("feature_dir", type=Path,  help="Directory containing feature .npy files")
    parser.add_argument("output_dir", type=Path, help="Directory to save pooled features")
    parser.add_argument("num_clusters", type=int, help="Number of clusters for algorithm")
    parser.add_argument("--algorithm", type=str, choices=["agglomerative", "birch", "both"], default="birch", help="Hierarchical clustering algorithm to use")
    parser.add_argument("--distance_metric", type=str, choices=["euclidean", "cosine"], default="euclidean", help="Distance metric for clustering")
    parser.add_argument("--threshold", type=float, default=None, help="Threshold for clustering (if applicable)")
    args = parser.parse_args()
    main(args)