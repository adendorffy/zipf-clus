import numpy as np
from pathlib import Path
from tqdm import tqdm
import faiss
from sklearn.cluster import kmeans_plusplus
from collections import defaultdict
import time
from pooling import load_pooled_features
from utils import write_partition_to_file

def convert_labels_to_dict(labels):
    partition_dict = defaultdict(list)

    for node_id, cluster_id in tqdm(enumerate(labels), desc="Converting labels to partition dict", unit="nodes", total=len(labels)):
        partition_dict[cluster_id].append(node_id)

    partition = list(partition_dict.values())
    return partition

def main(args):

    features, filenames, intervals = load_pooled_features(args.feature_dir)
    start_time = time.time()

    output_dir = args.output_dir / "/".join(args.feature_dir.parts[-6:])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving partition to {output_dir}")

    tmp_path = output_dir / "tmp-kmeans"
    tmp_path.mkdir(parents=True, exist_ok=True)
    acoustic_model = faiss.Kmeans(
        features.shape[1], args.num_clusters, niter=15, nredo=3, verbose=True
    )
    print("Initializing k-means centroids with sklearn's kmeans++")

    tmp_save_path = tmp_path / f"kmeans_initial_centroids_{args.num_clusters}.npy"
    if tmp_save_path.exists():
        print(f"Loading existing initial centroids from {tmp_save_path}")
        initial_centroids = np.load(tmp_save_path)
    else:
        subset_features = features[np.random.choice(features.shape[0], min(args.num_clusters*3, features.shape[0]), replace=False)]
        sk_kmeans, _ = kmeans_plusplus(subset_features, args.num_clusters, random_state=0)
        initial_centroids = sk_kmeans.astype(np.float32)
        np.save(tmp_save_path, initial_centroids)

    print("Training k-means model with faiss")
    acoustic_model.train(features, init_centroids=initial_centroids)
    _, Index = acoustic_model.index.search(features, 1)
    labels = Index.flatten()
    partition = convert_labels_to_dict(labels)

    total_time = time.time() - start_time
    print(f"K-means clustering completed in {total_time:.2f} seconds.")
    partition_path = output_dir / f"kmeans_k{args.num_clusters}_{total_time:.2f}.txt"
    write_partition_to_file(partition, filenames, intervals, partition_path)
    print(f"Partition saved to {partition_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run pooling on features")
    parser.add_argument("feature_dir", type=Path,  help="Directory containing feature .npy files")
    parser.add_argument("output_dir", type=Path, help="Directory to save pooled features")
    parser.add_argument("num_clusters", type=int, help="Number of clusters for k-means algorithm")
    args = parser.parse_args()
    main(args)