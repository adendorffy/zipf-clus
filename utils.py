# Copyright (c) 2026 Danel Slabbert, Stellenbosch University. All rights reserved.

"""
utils.py

Shared utility functions used across the pipeline.

Contents:
  - Frame / second conversion helpers.
  - A batch-index generator for memory-efficient iteration.
  - A resolution-adaptive Leiden graph partitioner that steers the number of
    discovered clusters towards a user-specified target.
  - A writer that serialises a partition to a human-readable class file.
"""

import leidenalg as la
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Frame / time conversion
# ---------------------------------------------------------------------------


def frames_to_seconds(frame_num, ms_per_frame=20):
    """Convert a frame index to a timestamp in seconds.

    Args:
        frame_num (int | float): Zero-based frame index.
        ms_per_frame (int): Frame shift in milliseconds. Defaults to 20 ms
            (50 Hz).

    Returns:
        float: Timestamp in seconds, rounded to two decimal places.
    """
    return round(frame_num * ms_per_frame / 1000.0, 2)


def seconds_to_frames(seconds, ms_per_frame=20):
    """Convert a timestamp in seconds to the nearest frame index.

    Rounds to one decimal place before applying nearest-integer rounding
    (adding 0.5 then flooring) to match standard rounding behaviour while
    avoiding floating-point drift.

    Args:
        seconds (float | np.ndarray): Timestamp(s) in seconds.
        ms_per_frame (int): Frame shift in milliseconds. Defaults to 20 ms
            (50 Hz).

    Returns:
        np.int32 | np.ndarray[np.int32]: Corresponding frame index/indices.
    """
    return np.floor(np.round((seconds / ms_per_frame * 1000), 1) + 0.5).astype(np.int32)


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def batch_indices(total_length, batch_size):
    """Yield successive non-overlapping index ranges over a sequence.

    Example::

        list(batch_indices(10, 3))
        # → [range(0, 3), range(3, 6), range(6, 9), range(9, 10)]

    Args:
        total_length (int): Total number of elements.
        batch_size (int): Maximum size of each batch.

    Yields:
        range: Index range for each batch.
    """
    for i in range(0, total_length, batch_size):
        yield range(i, min(i + batch_size, total_length))


# ---------------------------------------------------------------------------
# Partition conversion
# ---------------------------------------------------------------------------


def convert_labels_to_dict(labels):
    """Convert a flat array of cluster labels into a list-of-lists partition.

    Args:
        labels (array-like): Integer cluster assignment for each segment,
            indexed by segment id.

    Returns:
        list[list[int]]: Each inner list contains the segment indices that
            belong to one cluster.  Cluster order matches ascending cluster id.
    """
    partition_dict = defaultdict(list)
    for node_id, cluster_id in tqdm(
        enumerate(labels),
        desc="Converting labels to partition dict",
        unit="nodes",
        total=len(labels),
    ):
        partition_dict[cluster_id].append(node_id)

    return list(partition_dict.values())


# ---------------------------------------------------------------------------
# Graph partitioning
# ---------------------------------------------------------------------------


def partition_graph(
    graph,
    num_clusters,
    quality_function=la.CPMVertexPartition,
    resolution=0.5,
    max_iterations=15,
    tolerance=5,
):
    """Partition a weighted graph into approximately ``num_clusters`` clusters
    using the Leiden algorithm with adaptive resolution tuning.

    The Leiden ``resolution_parameter`` controls cluster granularity.
    Starting from ``resolution``, the function iteratively re-partitions the
    graph and nudges the resolution up or down with a decaying learning rate
    until the cluster count is within ``tolerance`` of ``num_clusters``, the
    iteration budget is exhausted, or patience runs out.

    The best partition seen across all iterations (closest cluster count to
    ``num_clusters``) is returned even if the tolerance was never met.

    When ``max_iterations=0`` the initial partition is returned immediately
    without any resolution tuning.

    Args:
        graph (igraph.Graph): Weighted graph to partition.  Must have a
            ``"weight"`` edge attribute.
        num_clusters (int): Target number of clusters.
        quality_function: Leiden partition quality class.  Defaults to
            ``leidenalg.CPMVertexPartition``.
        resolution (float): Initial resolution parameter.  Defaults to 0.5.
        max_iterations (int): Maximum number of resolution-tuning iterations.
            Defaults to 15.
        tolerance (int): Acceptable absolute difference between the discovered
            cluster count and ``num_clusters``.  Also used as the patience
            limit before the learning rate is halved.  Defaults to 5.

    Returns:
        tuple[leidenalg.VertexPartition, float]:
            - Best partition found.
            - Resolution parameter used for that partition.
    """
    print(
        f"Using graph with {graph.vcount():,} vertices and "
        f"{graph.ecount():,} edges for partitioning with "
        f"{quality_function.__name__}."
    )

    lr = 0.1
    best_diff = float("inf")
    best_partition = None
    patience_counter = 0

    partition = la.find_partition(
        graph,
        quality_function,
        weights="weight",
        resolution_parameter=resolution,
        seed=42,
        max_comm_size=1000,
    )

    if max_iterations == 0:
        print(
            f"Initial partitioning done.\n"
            f"Clusters: {len(set(partition.membership))}, "
            f"Res: {partition.resolution_parameter:.8f}, "
            f"Diff: {len(set(partition.membership)) - num_clusters:+d}"
        )
        return partition, partition.resolution_parameter

    for i in range(max_iterations):
        curr_clusters = len(set(partition.membership))
        diff = curr_clusters - num_clusters

        print(
            f"[Iter {i + 1:02}] res={partition.resolution_parameter:.6f}, "
            f"clusters={curr_clusters:,}, diff={diff:+d}"
        )

        if abs(diff) <= tolerance:
            print(f"Acceptable resolution found. Res: {resolution:.8f}")
            best_partition = partition
            break

        if abs(diff) < best_diff:
            best_diff = abs(diff)
            best_partition = partition
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= tolerance:
            lr *= 0.5
            patience_counter = 0

        diff_clipped = max(min(diff, 2 * num_clusters), -2 * num_clusters)
        step = lr * diff_clipped / num_clusters

        resolution = min(max(resolution - step, -10), 10)
        lr *= 0.9

        partition = la.find_partition(
            graph,
            la.CPMVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=42,
        )

    return best_partition, best_partition.resolution_parameter


# ---------------------------------------------------------------------------
# Output serialisation
# ---------------------------------------------------------------------------


def write_partition_to_file(partition, filenames, intervals, output_path):
    """Write a partition to a human-readable text file.

    Output format::

        Class 0
        <filename> <start_sec> <end_sec>
        <filename> <start_sec> <end_sec>
        ...

        Class 1
        ...

    Args:
        partition (leidenalg.VertexPartition): Partition object whose
            iteration yields per-cluster lists of vertex indices.
        filenames (list[str]): Utterance stem for each vertex, indexed by
            vertex id.
        intervals (list[tuple[int, int]]): ``(start_frame, end_frame)`` pair
            for each vertex, indexed by vertex id.
        output_path (str | Path): Path to the output text file.
    """
    with open(output_path, "w") as f:
        for id, cluster in enumerate(partition):
            f.write(f"Class {id}\n")
            for idx in cluster:
                filename = filenames[idx]
                interval = intervals[idx]
                f.write(
                    f"{filename} "
                    f"{frames_to_seconds(interval[0])} "
                    f"{frames_to_seconds(interval[1])}\n"
                )
            f.write("\n")
