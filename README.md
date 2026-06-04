# Recovering the Zipfian Distribution in Unsupervised Term Discovery

This repository contains the code for the paper:

> **Recovering the Zipfian Distribution in Unsupervised Term Discovery**
> Danel Slabbert, Simon Malan, Herman Kamper

<!-- > _Conference/Journal Name, Year_ -->

<!-- [[Paper](link)] [[BibTeX](#citation)] -->

<!-- ---

## Citation

```bibtex
@inproceedings{yourname2026zipfian,
  title     = {Recovering the {Zipfian} Distribution in Unsupervised Term Discovery},
  author    = {Your Name and Co-author Name and Co-author Name},
  booktitle = {Proc. IEEE Spoken Language Technology Workshop (SLT)},
  year      = {2026},
}
``` -->

---

## Systems Overview

| Systems                      | Features                                | Clustering                   |
| ---------------------------- | --------------------------------------- | ---------------------------- |
| Cosine graph + Leiden        | Continuous (PCA-projected, mean-pooled) | Graph community detection    |
| Edit-distance graph + Leiden | Discrete (KMeans-quantised)             | Graph community detection    |
| KMeans                       | Continuous                              | k-means++ + FAISS            |
| BIRCH                        | Continuous                              | Balanced CF-Tree             |
| Agglomerative                | Continuous                              | Hierarchical                 |
| GMM                          | Continuous                              | EM (spherical covariance)    |
| FBGMM                        | Continuous                              | Bayesian GMM, Gibbs sampling |

---

## 1. Continuous Systems

These systems operate on continuous segment embeddings produced by
`pooling.py`, which fits a StandardScaler + PCA on frame-level SSL features
and mean-pools each boundary-delimited segment into a single fixed-length
vector.

### Step 1 (shared): Pool segments

```bash
python pooling.py [--n_components N_COMPONENTS] [--batch_size BATCH_SIZE] \
    feature_dir boundary_dir
```

| Argument         | Description                                                              |
| ---------------- | ------------------------------------------------------------------------ |
| `feature_dir`    | Directory of frame-level feature files (`*.npy`)                         |
| `boundary_dir`   | Directory of boundary files (`*.list`, timestamps in seconds)            |
| `--n_components` | Number of PCA components (default: 350)                                  |
| `--batch_size`   | Batch size for incremental fitting when > 10 000 files (default: 10 000) |

---

### Step 2a: Cosine graph + Leiden

Builds a cosine-similarity graph over the pooled embeddings (edge iff cosine
distance ≤ threshold) and partitions it with the Leiden algorithm using
adaptive resolution tuning towards a target cluster count.

```bash
python cosine_graph.py [--n_jobs N_JOBS] [--batch_size BATCH_SIZE] \
    [--threshold THRESHOLD] [--resolution RESOLUTION] [--max_iter MAX_ITER]  \
    [--tolerance TOLERANCE] [--quality_function {modularity,cpm,rb}]         \
    feature_dir output_dir num_clusters
```

| Argument             | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| `feature_dir`        | Directory of pooled segment embeddings                       |
| `output_dir`         | Root output directory                                        |
| `num_clusters`       | Target number of clusters                                    |
| `--threshold`        | Max cosine distance for an edge (default: 0.5)               |
| `--n_jobs`           | Parallel workers for edge computation (default: all cores)   |
| `--batch_size`       | Source nodes per parallel batch (default: 1 000)             |
| `--resolution`       | Initial Leiden resolution parameter (default: 0.5)           |
| `--max_iter`         | Max resolution-tuning iterations (default: 15)               |
| `--tolerance`        | Acceptable cluster-count deviation (default: 5)              |
| `--quality_function` | Leiden quality function: `cpm` (default), `modularity`, `rb` |

---

### Step 2b: KMeans

Two-stage clustering: k-means++ initialisation on a random subset, refined
by FAISS over 15 iterations with 3 restarts.

```bash
python kmeans.py feature_dir output_dir num_clusters
```

| Argument       | Description                            |
| -------------- | -------------------------------------- |
| `feature_dir`  | Directory of pooled segment embeddings |
| `output_dir`   | Root output directory                  |
| `num_clusters` | Number of clusters (k)                 |

---

### Step 2c: BIRCH

Builds a Clustering Feature Tree in a single pass for memory-efficient
clustering.

```bash
python BIRCH.py feature_dir output_dir num_clusters
```

| Argument       | Description                            |
| -------------- | -------------------------------------- |
| `feature_dir`  | Directory of pooled segment embeddings |
| `output_dir`   | Root output directory                  |
| `num_clusters` | Number of clusters                     |

---

### Step 2d: Agglomerative

Bottom-up hierarchical clustering. Supports fixed-k and distance-threshold
modes, configurable linkage criterion and distance metric.

```bash
python agglomerative.py [--linkage_criterion {average,complete,ward}] \
    [--distance_metric {euclidean,cosine}] [--threshold THRESHOLD]               \
    feature_dir output_dir num_clusters
```

| Argument              | Description                                                                |
| --------------------- | -------------------------------------------------------------------------- |
| `feature_dir`         | Directory of pooled segment embeddings                                     |
| `output_dir`          | Root output directory                                                      |
| `num_clusters`        | Number of clusters (ignored if `--threshold` is set)                       |
| `--linkage_criterion` | Linkage criterion (default: `average`). Note: `ward` requires `euclidean`. |
| `--distance_metric`   | Distance metric (default: `euclidean`)                                     |
| `--threshold`         | Distance threshold for cutting the dendrogram (threshold mode)             |

---

### Step 2e: GMM

Fits a Gaussian Mixture Model with spherical covariance via EM.

```bash
python gmm.py feature_dir output_dir num_clusters
```

| Argument       | Description                            |
| -------------- | -------------------------------------- |
| `feature_dir`  | Directory of pooled segment embeddings |
| `output_dir`   | Root output directory                  |
| `num_clusters` | Number of Gaussian components          |

---

### Step 2f: FBGMM

Finite Bayesian GMM trained via collapsed Gibbs sampling. The effective
cluster count can adapt during sampling even when initialised with a fixed K.
Supports checkpointing and resumption.

```bash
python fbgmm.py [--n_iter N_ITER] [--alpha ALPHA]                     \
    [--k_0 K_0] [--s_0 S_0] [--covariance_type {full,diag,fixed}] [--each_in_own] \
    [--resume]                                                                    \
    feature_dir output_dir num_clusters
```

| Argument            | Description                                               |
| ------------------- | --------------------------------------------------------- |
| `feature_dir`       | Directory of pooled segment embeddings                    |
| `output_dir`        | Root output directory                                     |
| `num_clusters`      | Initial number of clusters K (ignored if `--each_in_own`) |
| `--n_iter`          | Number of Gibbs sampling iterations (default: 1)          |
| `--alpha`           | Dirichlet Process concentration parameter (default: 1.0)  |
| `--k_0`             | Prior scale on cluster mean precision (default: 0.05)     |
| `--s_0`             | Prior scale on cluster covariance (default: 0.001)        |
| `--covariance_type` | Component covariance type (default: `fixed`)              |
| `--each_in_own`     | Initialise each segment in its own cluster (K = N)        |
| `--resume`          | Resume Gibbs sampling from the latest matching checkpoint |

---

## 2. Discrete System

This system quantises frame-level SSL features into sequences of discrete
acoustic units using KMeans, then computes edit distances between segments
to build a graph.

### Step 1: Quantise segments

```bash
python quantiser.py [--k K] [--batch_size BATCH_SIZE] \
    [--total_hours TOTAL_HOURS] [--collapsed]          \
    train_features_dir feature_dir boundary_dir
```

| Argument             | Description                                                 |
| -------------------- | ----------------------------------------------------------- |
| `train_features_dir` | Feature files used to fit the KMeans codebook               |
| `feature_dir`        | Feature files to quantise                                   |
| `boundary_dir`       | Directory of boundary files (`*.list`)                      |
| `--k`                | Codebook size (default: 500)                                |
| `--batch_size`       | Mini-batch size for MiniBatchKMeans (default: 1 000)        |
| `--total_hours`      | Max hours of audio for KMeans fitting (default: 2)          |
| `--collapsed`        | Deduplicate consecutive identical units within each segment |

### Step 2: Edit-distance graph + Leiden

Builds a normalised edit-distance graph over the quantised unit sequences
(edge iff normalised edit distance ≤ threshold) and partitions it with Leiden.

```bash
python edit_graph.py [--threshold THRESHOLD]     \
    [--resolution RESOLUTION] [--max_iter MAX_ITER] [--tolerance TOLERANCE] \
    feature_dir output_dir num_clusters
```

| Argument       | Description                                             |
| -------------- | ------------------------------------------------------- |
| `feature_dir`  | Directory of quantised segment files                    |
| `output_dir`   | Root output directory                                   |
| `num_clusters` | Target number of clusters                               |
| `--threshold`  | Max normalised edit distance for an edge (default: 0.5) |
| `--resolution` | Initial Leiden resolution parameter (default: 0.5)      |
| `--max_iter`   | Max resolution-tuning iterations (default: 15)          |
| `--tolerance`  | Acceptable cluster-count deviation (default: 5)         |

---

## Output Format

All systems write a plain-text partition file of the form:

```
Class 0
<filename> <start_sec> <end_sec>
<filename> <start_sec> <end_sec>

Class 1
<filename> <start_sec> <end_sec>
...
```

Partition filenames encode the key hyperparameters and wall-clock time so that results from different runs can
be compared without opening the files.
