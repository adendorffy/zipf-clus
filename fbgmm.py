# Copyright (c) 2026 Danel Slabbert, Stellenbosch University. All rights reserved.

"""
fbgmm.py

Clusters pre-computed segment embeddings with a Finite Bayesian Gaussian
Mixture Model (FBGMM) trained via collapsed Gibbs sampling.

The FBGMM places a Dirichlet Process prior (concentration ``alpha``) over
cluster assignments, allowing the effective number of active clusters to
shrink or grow during sampling even when initialised with ``K`` components.

Two initialisation strategies are supported via ``--each_in_own``:
  - ``rand``         : randomly assign each segment to one of K clusters.
  - ``each-in-own``  : start every segment in its own cluster (K = N);
                       the model then merges clusters during sampling.

Two covariance types are supported via ``--covariance_type``:
  - ``fixed`` : diagonal covariance fixed to a data-driven estimate scaled
                by ``s_0`` and ``k_0``; uses ``FixedVarPrior``.
  - ``full`` / ``diag`` : conjugate Normal-Inverse-Wishart (NIW) prior.

Checkpointing:
  The full model state is pickled after every Gibbs iteration to
  ``<output_dir>/models/``.  Training can be resumed from the latest
  checkpoint with ``--resume``.  Note that each iteration writes a new file
  (the per-iteration wall time is embedded in the name), so old checkpoints
  are not automatically deleted.

Output:
  A plain-text partition file written by ``utils.write_partition_to_file``,
  named with the active cluster count, iteration count, and all
  hyperparameters for reproducibility.
"""

import pickle
import time
from pathlib import Path

import numpy as np

# FBGMM implementation by Herman Kamper – https://github.com/kamperh/bayes_gmm
from bayes_gmm.fbgmm import FBGMM
from bayes_gmm.gaussian_components_fixedvar import FixedVarPrior
from bayes_gmm.niw import NIW

from pooling import load_pooled_features
from utils import convert_labels_to_dict, write_partition_to_file


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------


def print_iteration_info(i, labels, log_marginal=None, K_target=None, time=None):
    """Print a one-line summary of the current Gibbs iteration.

    Args:
        i (int): Current iteration index.
        labels (array-like): Cluster assignment for each segment.
        log_marginal (float | None): Log marginal likelihood of the current
            assignment; omitted from the line if ``None``.
        K_target (int | None): Initial / maximum cluster count; used to
            report the number of empty clusters.  Omitted if ``None``.
        time (float | None): Wall-clock seconds for this iteration; omitted
            if ``None``.
    """
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)

    active_clusters = len(unique)
    min_size = counts.min()
    max_size = counts.max()
    mean_size = counts.mean()
    median_size = np.median(counts)

    msg = f"iter={i:03d} | active={active_clusters}"

    if K_target is not None:
        empty_clusters = K_target - active_clusters
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


# ---------------------------------------------------------------------------
# Prior construction
# ---------------------------------------------------------------------------


def set_prior(features, k_0, s_0, covariance_type):
    """Construct the prior distribution for the FBGMM.

    For ``covariance_type="fixed"``, a ``FixedVarPrior`` is built from the
    empirical per-dimension variance of the features scaled by ``s_0`` and
    ``k_0``.  For ``"full"`` or ``"diag"``, a standard Normal-Inverse-Wishart
    (NIW) prior is used with a spherical identity covariance.

    Args:
        features (np.ndarray): Segment embeddings, shape ``(N, D)``.
        k_0 (float): Scale parameter controlling the prior precision on the
            cluster mean (smaller → more diffuse mean prior).
        s_0 (float): Scale parameter controlling the prior on the covariance
            (smaller → tighter covariance prior).
        covariance_type (str): One of ``"fixed"``, ``"full"``, ``"diag"``.

    Returns:
        FixedVarPrior | NIW: Constructed prior object.
    """
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
            cov_0=np.eye(features.shape[1]),
        )
    return prior


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    """Run FBGMM Gibbs sampling and write the final partition.

    Resumes from the latest checkpoint if ``--resume`` is set and a matching
    checkpoint file is found; otherwise initialises a fresh model.

    Args:
        args: Parsed CLI arguments (see ``__main__`` block for full list).

    Note:
        ``--checkpoint`` is accepted by the parser but is not currently used;
        checkpoint paths are derived automatically from the hyperparameters.
    """
    # Load and normalise segment embeddings (shape: N × D).
    features, filenames, intervals = load_pooled_features(args.feature_dir)

    output_dir = args.output_dir / "/".join(args.feature_dir.parts[-6:])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving partition to {output_dir}")

    start_time = time.time()
    print("Running FBGMM")

    name_suffix = (
        f"alpha{args.alpha}_KO{args.k_0}_SO{args.s_0}_"
        f"{args.covariance_type}"
        f"{'_each-in-own' if args.each_in_own else ''}"
    )

    # ------------------------------------------------------------------
    # Initialisation: resume from checkpoint or create a fresh model
    # ------------------------------------------------------------------
    if args.resume:
        checkpoint_glob = output_dir / "models" / f"fbgmm_k*_{name_suffix}_*.pkl"
        exists = list(checkpoint_glob.parent.glob(checkpoint_glob.name))

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
            print("No checkpoint found; starting fresh despite --resume flag.")
            args.resume = False

    if not args.resume:
        prior = set_prior(features, args.k_0, args.s_0, args.covariance_type)
        K_target = args.num_clusters if not args.each_in_own else features.shape[0]

        assignments = "each-in-own" if args.each_in_own else "rand"

        fbgmm_model = FBGMM(
            features,
            prior,
            alpha=args.alpha,
            K=K_target,
            assignments=assignments,
            covariance_type=args.covariance_type,
        )
        log_marginal_trace = []
        start_iter = 0

    # ------------------------------------------------------------------
    # Gibbs sampling loop
    # ------------------------------------------------------------------
    for i in range(start_iter + 1, start_iter + args.n_iter + 1):
        iter_start = time.time()

        fbgmm_model.gibbs_sample(1)

        labels = fbgmm_model.components.assignments
        log_marginal = fbgmm_model.log_marg()
        log_marginal_trace.append(log_marginal)
        iter_time = time.time() - iter_start

        print_iteration_info(
            i=i,
            labels=labels,
            log_marginal=log_marginal,
            K_target=K_target,
            time=iter_time,
        )

        checkpoint = {
            "model": fbgmm_model,
            "iteration": i,
            "log_marginal_trace": log_marginal_trace,
            "K_target": K_target,
        }
        checkpoint_path = (
            output_dir
            / "models"
            / f"fbgmm_k{K_target}_{name_suffix}_{iter_time:.2f}.pkl"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)

    print(f"Checkpoint saved after iteration {i}: {checkpoint_path}")

    # ------------------------------------------------------------------
    # Write final partition
    # ------------------------------------------------------------------
    active_clusters = len(np.unique(labels))
    n_iter = len(log_marginal_trace)
    partition = convert_labels_to_dict(labels)
    total_time = time.time() - start_time

    print(f"FBGMM clustering completed in {total_time:.2f} seconds.")

    partition_path = (
        output_dir
        / f"fbgmm_k{active_clusters}_iter{n_iter}_{name_suffix}_{total_time:.2f}.txt"
    )
    write_partition_to_file(partition, filenames, intervals, partition_path)
    print(f"Partition saved to {partition_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cluster segment embeddings with a Finite Bayesian GMM (Gibbs sampling)."
    )
    parser.add_argument(
        "feature_dir",
        type=Path,
        help="Directory containing pooled segment embedding files (*.npy).",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Root directory under which partition files and checkpoints will be saved.",
    )
    parser.add_argument(
        "num_clusters",
        type=int,
        help="Initial number of clusters K (ignored if --each_in_own is set).",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="Number of Gibbs sampling iterations to run (default: 1).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Dirichlet Process concentration parameter (default: 1.0).",
    )
    parser.add_argument(
        "--k_0",
        type=float,
        default=0.05,
        help="Prior scale on cluster mean precision (default: 0.05).",
    )
    parser.add_argument(
        "--s_0",
        type=float,
        default=0.1,
        help="Prior scale on cluster covariance (default: 0.001).",
    )
    parser.add_argument(
        "--covariance_type",
        type=str,
        choices=["full", "diag", "fixed"],
        default="fixed",
        help="Gaussian component covariance type (default: fixed).",
    )
    parser.add_argument(
        "--each_in_own",
        action="store_true",
        help="Initialise each segment in its own cluster (K = N); model merges during sampling.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="(Unused) Reserved for explicit checkpoint path; currently auto-derived from hyperparameters.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume Gibbs sampling from the latest matching checkpoint.",
    )
    args = parser.parse_args()
    main(args)
