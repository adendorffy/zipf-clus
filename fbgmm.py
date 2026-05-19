import numpy as np
from pathlib import Path
from tqdm import tqdm
from bayes_gmm.fbgmm import FBGMM
from bayes_gmm.niw import NIW

from collections import defaultdict
import time
# import tracemalloc
from pooling import load_pooled_features
from utils import write_partition_to_file

def set_prior_diag(X):
    N, D = X.shape

    m_0 = np.mean(X, axis=0)
    k_0 = 0.01
    v_0 = D + 3

    var = np.var(X, axis=0) + 1e-6
    S_0 = var * v_0   # vector, not matrix

    return NIW(m_0, k_0, v_0, S_0)

def convert_labels_to_dict(labels):
    partition_dict = defaultdict(list)

    for node_id, cluster_id in tqdm(enumerate(labels), desc="Converting labels to partition dict", unit="nodes", total=len(labels)):
        partition_dict[cluster_id].append(node_id)

    partition = list(partition_dict.values())
    return partition

def main(args):

    # tracemalloc.start()
    features, filenames, intervals = load_pooled_features(args.feature_dir)
    start_time = time.time()

    output_dir = args.output_dir / "/".join(args.feature_dir.parts[-6:])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving partition to {output_dir}")

    start_time = time.time()
    # tracemalloc.reset_peak()
    print("Running FBGMM")

    prior = set_prior_diag(features)
    fbgmm_model = FBGMM(features, prior, alpha=1.0, K=args.num_clusters, assignments="rand", covariance_type="diag")
    n_iter = 5
    record = fbgmm_model.gibbs_sample(n_iter)

    labels = fbgmm_model.components.assignments
    active_clusters = len(np.unique(labels))

    print(f"Completed {n_iter} Gibbs iterations.")
    print(f"Active clusters: {active_clusters}")
    print(f"Log marginal trace: {record['log_marg']}")

    partition = convert_labels_to_dict(labels)

    total_time = time.time() - start_time

    print(f"FBGMM clustering completed in {total_time:.2f} seconds.")

    partition_path = output_dir / (
        f"fbgmm_k{args.num_clusters}_iter{n_iter}_"
        f"active{active_clusters}_{total_time:.2f}.txt"
    )

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