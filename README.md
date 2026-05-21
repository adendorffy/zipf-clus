# Graph Clustering Pipelines
A compact overview of the two graph-clustering workflows in this project.

---

## 1. Continuous + Average Pooling + Cosine Graph

This pipeline works with continuous segment embeddings.

### Step 1: Pool word segments
Use `pooling.py` to average-pool frame-level features into segment-level embeddings.

```bash
python pooling.py [-h] [--n_components N_COMPONENTS] [--batch_size BATCH_SIZE] feature_dir boundary_dir
````

### Step 2: Build the cosine graph

Use the pooled segment embeddings to construct a cosine-similarity graph and run clustering.

```bash
python cosine.py [-h] [--n_jobs N_JOBS] [--batch_size BATCH_SIZE] [--threshold THRESHOLD] [--resolution RESOLUTION] [--max_iter MAX_ITER]
                 [--tolerance TOLERANCE]
                 feature_dir output_dir num_clusters
```

---

## 2. Discrete + Edit-Distance Graph

This pipeline works with quantised segment representations.

### Step 1: Quantise word segments

Use `quantise.py` to convert continuous features into discrete representations.

```bash
python quantise.py [-h] [--k K] [--batch_size BATCH_SIZE] [--total_hours TOTAL_HOURS] [--collapsed] train_features_dir feature_dir boundary_dir
```

### Step 2: Build the edit-distance graph

Use the quantised segments to construct an edit-distance graph and run clustering.

```bash
python edit.py [-h] [--pca_components PCA_COMPONENTS] [--threshold THRESHOLD] [--resolution RESOLUTION] [--max_iter MAX_ITER] [--tolerance TOLERANCE] feature_dir output_dir
```

---

## 3. Continuous + Average Pooling + FBGMM

This pipeline uses a Finite Bayesian Gaussian Mixture Model (FBGMM) to cluster pooled segment embeddings.

### Step 1: Pool word segments

Use `pooling.py` to average-pool frame-level features into segment-level embeddings.

```bash
python pooling.py [-h] [--n_components N_COMPONENTS] [--batch_size BATCH_SIZE] feature_dir boundary_dir
```

### Step 2: Run FBGMM clustering

Cluster the pooled embeddings using Gibbs sampling with a diagonal-covariance FBGMM.

```bash
python fbgmm.py [-h] [--n_iter N_ITER] feature_dir output_dir num_clusters
```

---

## Summary

* **Continuous pipeline:** pool segments → build cosine graph
* **Discrete pipeline:** quantise segments → build edit-distance graph
* **FBGMM:** pool segments → Bayesian mixture modelling → Gibbs sampling

