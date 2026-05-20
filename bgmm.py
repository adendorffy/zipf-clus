import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.mixture import BayesianGaussianMixture

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

    start_time = time.time()
    print("Running Bayesian GMM clustering with sklearn")

    gmm_model = BayesianGaussianMixture(n_components=args.num_clusters + 2000, covariance_type='diag', random_state=42)
    labels = gmm_model.fit_predict(features)
    partition = convert_labels_to_dict(labels)

    total_time = time.time() - start_time
    print(f"Bayesian GMM clustering completed in {total_time:.2f} seconds.")
    partition_path = output_dir / f"bgmm_k{args.num_clusters}_{total_time:.2f}.txt"
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