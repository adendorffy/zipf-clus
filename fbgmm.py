import numpy as np
from pathlib import Path
from tqdm import tqdm
from bayes_gmm.fbgmm import FBGMM
from bayes_gmm.gaussian_components_fixedvar import FixedVarPrior
from bayes_gmm.niw import NIW

from collections import defaultdict
import pickle

import time
from pooling import load_pooled_features
from utils import write_partition_to_file


def convert_labels_to_dict(labels):
    partition_dict = defaultdict(list)

    for node_id, cluster_id in tqdm(enumerate(labels), desc="Converting labels to partition dict", unit="nodes", total=len(labels)):
        partition_dict[cluster_id].append(node_id)

    partition = list(partition_dict.values())
    return partition

def print_iteration_info(i, labels, log_marginal=None, K_target=None, time=None):
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
    
    if time is not None:
        msg += f" | time={time:.2f}s"

    print(msg)

def set_prior(features, k_0, s_0, covariance_type):
    if covariance_type == "fixed":
        _, D = features.shape
        mu_0 = np.zeros(D)
        data_var = np.var(features, axis=0) + 1e-6
        var = data_var * s_0
        var_0 = var / k_0
        prior = FixedVarPrior(var, mu_0, var_0)
    else:
        prior = NIW(
            mu_0=np.zeros(features.shape[1]),
            k_0=k_0,
            s_0=s_0,
            cov_0=np.eye(features.shape[1])
        )
    return prior


def main(args):

    features, filenames, intervals = load_pooled_features(args.feature_dir)
    start_time = time.time()

    output_dir = args.output_dir / "/".join(args.feature_dir.parts[-6:])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving partition to {output_dir}")

    start_time = time.time()
    print("Running FBGMM")

    if args.resume:
        checkpoint_path = output_dir / "models" / (
            f"fbgmm_k*_alpha{args.alpha}_KO{args.k_0}_SO{args.s_0}_*_{args.covariance_type}{'_each-in-own' if args.each_in_own else ''}.pkl"
        )
        exists = list(checkpoint_path.parent.glob(checkpoint_path.name))
        if exists:
            checkpoint_path = exists[0]
            print(f"Loading FBGMM checkpoint from {checkpoint_path}")
            with open(checkpoint_path, "rb") as f:
                checkpoint = pickle.load(f)

            fbgmm_model = checkpoint["model"]
            log_marginal_trace = checkpoint.get("log_marginal_trace", [])
            start_iter = checkpoint.get("iteration", 0)

            K_target = checkpoint["K_target"]

    else:
        prior = set_prior(features, args.k_0, args.s_0, args.covariance_type)
        K_target = args.num_clusters if not args.each_in_own else features.shape[0]
        assignments = "rand" if not args.each_in_own else "each-in-own"

        fbgmm_model = FBGMM(
            features,
            prior,
            alpha=args.alpha,
            K=K_target,
            assignments=assignments,
            covariance_type=args.covariance_type
        )

        log_marginal_trace = []
        start_iter = 0
    
    for i in range(start_iter + 1, start_iter + args.n_iter + 1):
        time_start = time.time()
        fbgmm_model.gibbs_sample(1)   

        labels = fbgmm_model.components.assignments

        log_marginal = fbgmm_model.log_marg()
        log_marginal_trace.append(log_marginal)
        time_total = time.time() - time_start

        print_iteration_info(
            i=i,
            labels=labels,
            log_marginal=log_marginal,
            K_target=K_target,
            time=time_total
        )

        checkpoint = {
            "model": fbgmm_model,
            "iteration": i,
            "log_marginal_trace": log_marginal_trace,
            "K_target": K_target
        }
        checkpoint_path = output_dir / "models" / (
            f"fbgmm_k{K_target}_alpha{args.alpha}_KO{args.k_0}_SO{args.s_0}_{time_total}_{args.covariance_type}{'_each-in-own' if args.each_in_own else ''}.pkl"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)

    print(f"Checkpoint saved after iteration {i}: {checkpoint_path}")

    active_clusters = len(np.unique(labels))
    n_iter = len(log_marginal_trace)

    partition = convert_labels_to_dict(labels)

    total_time = time.time() - start_time

    print(f"FBGMM clustering completed in {total_time:.2f} seconds.")

    partition_path = output_dir / (
        f"fbgmm_k{active_clusters}_iter{n_iter}_"
        f"alpha{args.alpha}_KO{args.k_0}_SO{args.s_0}_{total_time:.2f}_{args.covariance_type}{'_each-in-own' if args.each_in_own else ''}.txt"
    )

    write_partition_to_file(partition, filenames, intervals, partition_path)

    print(f"Partition saved to {partition_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run pooling on features")
    parser.add_argument("feature_dir", type=Path,  help="Directory containing feature .npy files")
    parser.add_argument("output_dir", type=Path, help="Directory to save pooled features")
    parser.add_argument("num_clusters", type=int, help="Number of clusters for fbgmm algorithm")
    parser.add_argument("--n_iter", type=int, default=1, help="Number of iterations for Gibbs sampling")
    parser.add_argument("--alpha", type=float, default=1.0, help="Concentration parameter for Dirichlet Process")
    parser.add_argument("--k_0", type=float, default=0.05, help="Scale parameter for the prior: mu")
    parser.add_argument("--s_0", type=float, default=0.001, help="Scale parameter for the prior: covar")
    parser.add_argument("--covariance_type", type=str, choices=["full", "diag", "fixed"], default="fixed", help="Covariance type for Gaussian components")
    parser.add_argument("--each_in_own", action="store_true", help="Initialize each data point in its own cluster")
    parser.add_argument("--checkpoint", type=Path, default=None,
                    help="Path to save/load FBGMM checkpoint")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint instead of creating a new model")
    args = parser.parse_args()
    main(args)