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

def print_iteration_info(i, labels, log_marginal=None, K_target=None):
    labels = np.asarray(labels)

    unique, counts = np.unique(labels, return_counts=True)
    active_clusters = len(unique)
    empty_clusters = None if K_target is None else K_target - active_clusters

    min_size = counts.min()
    max_size = counts.max()
    mean_size = counts.mean()
    median_size = np.median(counts)

    msg = (
        f"iter={i:03d} | "
        f"active={active_clusters}"
    )

    if K_target is not None:
        msg += f"/{K_target} | empty={empty_clusters}"

    msg += (
        f" | size min/median/mean/max="
        f"{min_size}/{median_size:.1f}/{mean_size:.1f}/{max_size}"
    )

    if log_marginal is not None:
        msg += f" | log_marginal={float(log_marginal):.2f}"

    print(msg)

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
    fbgmm_model = FBGMM(features, prior, alpha=1000.0, K=args.num_clusters, assignments="rand", covariance_type="diag")
    K_target = args.num_clusters
    log_marginal_trace = []

    for i in range(1, args.n_iter + 1):
        fbgmm_model.gibbs_sample(1)   

        labels = fbgmm_model.components.assignments

        # Use the correct attribute/method from your FBGMM implementation
        log_marginal = fbgmm_model.log_marg()
        log_marginal_trace.append(log_marginal)

        print_iteration_info(
            i=i,
            labels=labels,
            log_marginal=log_marginal,
            K_target=K_target,
        )
    active_clusters = len(np.unique(labels))   
    n_iter = len(log_marginal_trace)

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
    parser.add_argument("num_clusters", type=int, help="Number of clusters for fbgmm algorithm")
    parser.add_argument("--n_iter", type=int, default=5, help="Number of iterations for Gibbs sampling")
    args = parser.parse_args()
    main(args)