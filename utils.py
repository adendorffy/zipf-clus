import leidenalg as la
import numpy as np

def frames_to_seconds(frame_num, ms_per_frame=20):
    return round(frame_num * ms_per_frame / 1000.0, 2)

def seconds_to_frames(seconds, ms_per_frame=20):
    return np.floor(np.round((seconds / ms_per_frame * 1000), 1) + 0.5).astype(np.int32)

def batch_indices(total_length, batch_size):
    for i in range(0, total_length, batch_size):
        yield range(i, min(i + batch_size, total_length))

def partition_graph(graph, num_clusters, resolution=0.5, max_iterations=15, tolerance=5):
    print(f"Using graph with {graph.vcount():,} vertices and {graph.ecount():,} edges for partitioning.")

    tolerance = 5
    lr = 0.1

    best_diff = float('inf')
    best_partition = None
    patience_counter = 0    

    partition = la.find_partition(
        graph,
        la.CPMVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=42,
        max_comm_size=1000
    )   

    if max_iterations == 0:
        print(f"Initial partitioning done.\nClusters: {len(set(partition.membership))}, Res: {partition.resolution_parameter:.8f}, Diff: {len(set(partition.membership)) - num_clusters:+d}")
        return partition, partition.resolution_parameter
    

    for i in range(max_iterations):
        curr_clusters = len(set(partition.membership))
        diff = curr_clusters - num_clusters

        print(f"[Iter {i+1:02}] res={partition.resolution_parameter:.6f}, "
                f"clusters={curr_clusters:,}, diff={diff:+d}")
        
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

def write_partition_to_file(partition, filenames, intervals, output_path):
    
    with open(output_path, "w") as f:
        for id, cluster in enumerate(partition):
            f.write(f"Class {id}\n")
            for idx in cluster:
                filename = filenames[idx]
                interval = intervals[idx]
                f.write(f"{filename} {frames_to_seconds(interval[0])} {frames_to_seconds(interval[1])}\n")
            f.write("\n")