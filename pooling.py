# Copyright (c) 2026 Danel Slabbert, Stellenbosch University. All rights reserved.

"""
pooling.py

Transforms raw frame-level features into fixed-dimensional segment embeddings.

Pipeline per file:
  1. StandardScaler  – zero-mean, unit-variance normalisation across all frames.
  2. PCA             – linear dimensionality reduction to `pca_components` dims.
  3. Boundary-aware mean pooling – each speech segment (defined by a .list
     boundary file) is collapsed to a single vector by averaging its frames.

Two fitting strategies are provided:
  - fit()         : loads all .npy files into RAM at once (fast, < 10 k files).
  - fit_batched() : IncrementalPCA + partial_fit loop for larger corpora.

Fitted scaler and PCA models are serialised next to the pooled features so that
the same projection can be reused at inference time without refitting.
"""

from pathlib import Path

import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils import seconds_to_frames


# ---------------------------------------------------------------------------
# Pooling class
# ---------------------------------------------------------------------------


class Pooling:
    """Fits a scaler + PCA on frame-level features and pools them into
    fixed-length segment embeddings aligned to provided time boundaries.

    Args:
        feature_dir (str | Path): Root directory containing per-utterance
            frame-level feature files (``*.npy``), searched recursively.
        boundary_dir (str | Path): Root directory containing per-utterance
            boundary files (``*.list``), each holding a sequence of boundary
            timestamps in seconds (one per line).
        pca_components (int): Target dimensionality after PCA. Defaults to 350.
    """

    def __init__(self, feature_dir, boundary_dir, pca_components=350):
        self.feature_dir = Path(feature_dir)
        self.boundary_dir = Path(boundary_dir)

        self.scaler_model = None  # sklearn StandardScaler, set after fit()
        self.pca_model = None  # sklearn PCA / IncrementalPCA, set after fit()
        self.pca_components = pca_components

        # Derive and create output directory based on feature / boundary paths.
        self.output_dir = self.set_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Paths at which the fitted models are cached between runs.
        self.scaler_path = self.output_dir / "scaler_model.npy"
        self.pca_path = self.output_dir / "pca_model.npy"

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------

    def set_output_dir(self):
        """Construct the output directory path from the last four parts of
        ``feature_dir`` and the last two parts of ``boundary_dir``.

        This keeps outputs self-descriptive and co-located with the features
        they are derived from.

        Returns:
            Path: Resolved output directory path.
        """
        output_dir = "/".join(self.feature_dir.parts[-4:])
        output_dir = (
            Path("pooled_features")
            / output_dir
            / "/".join(self.boundary_dir.parts[-2:])
        )
        print(f"Setting output directory to: {output_dir}")
        return output_dir

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------

    def fit(self):
        """Fit (or load) the StandardScaler and PCA models.

        If both ``scaler_model.npy`` and ``pca_model.npy`` already exist in
        the output directory the models are loaded from disk and fitting is
        skipped.  Otherwise all feature files are stacked into a single matrix
        and used to fit both models sequentially.

        For corpora with more than 10 000 feature files the method
        automatically delegates to :meth:`fit_batched` to avoid memory
        exhaustion.
        """
        # Re-use cached models if available.
        if self.scaler_path.exists() and self.pca_path.exists():
            print("Loading existing scaler and PCA models...")
            self.scaler_model = np.load(self.scaler_path, allow_pickle=True).item()
            self.pca_model = np.load(self.pca_path, allow_pickle=True).item()
            return

        feature_files = sorted(self.feature_dir.rglob("*.npy"))

        # Fall back to memory-efficient incremental fitting for large corpora.
        if len(feature_files) > 10_000:
            print(
                f"Warning: Found {len(feature_files)} feature files. "
                "Using fit_batched() to avoid memory issues."
            )
            self.fit_batched()
            return

        print("Fitting scaler and PCA on all features at once...")

        # Stack all frame-level features into (N_frames, D) matrix.
        features = [np.load(f) for f in tqdm(feature_files, desc="Loading features")]
        features = np.vstack(features)

        # --- Fit StandardScaler ---
        if not self.scaler_path.exists():
            self.scaler_model = StandardScaler()
            self.scaler_model.fit(features)
            np.save(self.scaler_path, self.scaler_model)
        else:
            self.scaler_model = np.load(self.scaler_path, allow_pickle=True).item()

        scaled_features = self.scaler_model.transform(features)

        # --- Fit PCA ---
        if not self.pca_path.exists():
            self.pca_model = PCA(n_components=self.pca_components)
            self.pca_model.fit(scaled_features)
            np.save(self.pca_path, self.pca_model)
        else:
            self.pca_model = np.load(self.pca_path, allow_pickle=True).item()

    def fit_batched(self, batch_size=10_000):
        """Memory-efficient fitting using ``partial_fit`` / ``IncrementalPCA``.

        Iterates over feature files in chunks of ``batch_size``, updating the
        scaler and PCA incrementally.  Both models are saved to disk on
        completion.  If cached models already exist they are loaded instead.

        Args:
            batch_size (int): Number of feature files processed per chunk.
                Defaults to 10 000.
        """
        # Re-use cached models if available.
        if self.scaler_path.exists() and self.pca_path.exists():
            print("Loading existing scaler and PCA models...")
            self.scaler_model = np.load(self.scaler_path, allow_pickle=True).item()
            self.pca_model = np.load(self.pca_path, allow_pickle=True).item()
            return

        feature_files = sorted(self.feature_dir.rglob("*.npy"))

        # --- Pass 1: fit StandardScaler incrementally ---
        print(f"Fitting scaler in batches of {batch_size}...")
        self.scaler_model = StandardScaler()
        for batch_start in tqdm(
            range(0, len(feature_files), batch_size), desc="Fitting scaler"
        ):
            batch_files = feature_files[batch_start : batch_start + batch_size]
            batch_features = np.vstack([np.load(f) for f in batch_files])
            self.scaler_model.partial_fit(batch_features)
        np.save(self.scaler_path, self.scaler_model)

        # --- Pass 2: fit IncrementalPCA on scaled features ---
        self.pca_model = IncrementalPCA(n_components=self.pca_components)
        print(f"Fitting PCA in batches of {batch_size}...")
        for batch_start in tqdm(
            range(0, len(feature_files), batch_size), desc="Fitting PCA"
        ):
            batch_files = feature_files[batch_start : batch_start + batch_size]
            batch_features = np.vstack([np.load(f) for f in batch_files])
            batch_features_scaled = self.scaler_model.transform(batch_features)
            self.pca_model.partial_fit(batch_features_scaled)
        np.save(self.pca_path, self.pca_model)

    # ------------------------------------------------------------------
    # Feature transformation and pooling
    # ------------------------------------------------------------------

    def transform(self):
        """Apply the fitted scaler + PCA and pool features per boundary segment.

        For each utterance:
          1. Scale and project all frames with the fitted models.
          2. Look up its boundary file (matched by stem).
          3. Slice the projected frames at each boundary and mean-pool each
             slice into a single ``(pca_components,)`` vector.
          4. Save each pooled vector as a separate ``.npy`` file named
             ``<stem>_<start>_<end>.npy``.

        Utterances without a matching boundary file are pooled as a whole and
        saved as ``<stem>_0_<n_frames>.npy``.
        """
        feature_files = sorted(self.feature_dir.rglob("*.npy"))

        # Pre-load all boundary files into a dict keyed by utterance stem.
        all_boundaries = {}
        for boundary_file in sorted(self.boundary_dir.rglob("*.list")):
            all_boundaries[boundary_file.stem] = np.loadtxt(boundary_file)

        for feature_file in tqdm(feature_files, desc="Transforming features"):
            features = np.load(feature_file)

            # Skip empty files.
            if features.size == 0:
                continue

            # Ensure 2-D shape (n_frames, D) even for single-frame utterances.
            if features.ndim == 1:
                features = features.reshape(1, -1)

            # Project frames through scaler → PCA.
            scaled_features = self.scaler_model.transform(features)
            pca_features = self.pca_model.transform(scaled_features)

            segment_name = feature_file.stem

            if segment_name in all_boundaries:
                boundaries = all_boundaries[segment_name]

                # Normalise boundaries to a 1-D array regardless of how
                # many boundaries the file contains.
                if not isinstance(boundaries, np.ndarray):
                    boundaries = np.array(boundaries)
                if boundaries.ndim == 0:
                    boundaries = np.array([boundaries])

                # Convert boundary timestamps (seconds) to frame indices.
                boundaries = [seconds_to_frames(b) for b in boundaries]

                # Slice and pool each inter-boundary segment.
                start = 0
                for end in boundaries:
                    out_path = (
                        self.output_dir
                        / feature_file.relative_to(self.feature_dir).parent
                        / f"{feature_file.stem}_{start}_{end}.npy"
                    )

                    # Skip already-processed segments (allows resuming).
                    if out_path.exists():
                        print(
                            f"Warning: Skipping existing segment for "
                            f"{feature_file.stem} at boundary {end}"
                        )
                        start = end
                        continue

                    # Skip degenerate zero-length segments.
                    if end == start:
                        print(
                            f"Warning: Skipping zero-length segment for "
                            f"{feature_file.stem} at boundary {end}"
                        )
                        continue

                    segment = pca_features[start:end]
                    if segment.size == 0:
                        print(
                            f"Warning: Skipping zero-length segment for "
                            f"{feature_file.stem} at boundary {end}"
                        )
                        continue

                    # Mean-pool frames → single embedding vector.
                    pooled_segment = np.mean(segment, axis=0)
                    pooled_feature = np.asarray(pooled_segment, dtype=np.float32)

                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(out_path, pooled_feature)
                    start = end

            else:
                # No boundary file: pool the entire utterance as one segment.
                print(
                    f"Warning: No boundaries found for {feature_file.stem}. "
                    "Saving pooled feature for entire segment."
                )
                pooled_feature = np.mean(pca_features, axis=0)
                pooled_feature = np.asarray(pooled_feature, dtype=np.float32)
                out_path = (
                    self.output_dir
                    / feature_file.relative_to(self.feature_dir).parent
                    / f"{feature_file.stem}_0_{features.shape[0]}.npy"
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_path, pooled_feature)


# ---------------------------------------------------------------------------
# Loader utility
# ---------------------------------------------------------------------------


def load_pooled_features(feature_dir):
    """Load all pooled segment embeddings from a directory into NumPy arrays.

    Files whose stem contains ``"model"`` are skipped (they are cached
    scaler / PCA artefacts, not segment embeddings).

    Post-processing applied to the stacked feature matrix:
      - Mean-centre across segments (subtract per-dimension mean).
      - L2-normalise each segment vector (ε = 1e-10 for numerical stability).

    Args:
        feature_dir (Path): Directory produced by :meth:`Pooling.transform`.

    Returns:
        tuple:
            - **features** (*np.ndarray*, float32, shape ``(N, D)``): Stacked,
              normalised segment embeddings.
            - **filenames** (*list[str]*): Utterance stem for each segment
              (i.e. the filename without the ``_start_end`` suffix).
            - **intervals** (*list[tuple[int,int]]*): ``(start_frame,
              end_frame)`` pair for each segment.
    """
    feature_files = sorted(feature_dir.rglob("*.npy"))
    features, filenames, intervals = [], [], []

    for feature_file in tqdm(feature_files, desc="Loading pooled features"):
        # Skip serialised model artefacts stored alongside the embeddings.
        if "model" in feature_file.stem:
            continue

        pooled_feature = np.load(feature_file)
        features.append(pooled_feature)

        # Recover (start, end) frame indices from the filename convention
        # "<utterance_stem>_<start>_<end>.npy".
        start_frame, end_frame = map(int, feature_file.stem.split("_")[-2:])
        filenames.append(feature_file.stem.rsplit("_", 2)[0])
        intervals.append((start_frame, end_frame))

    features = np.asarray(features, dtype=np.float32)

    # Mean-centre then L2-normalise so cosine similarity == dot product.
    features = features - np.mean(features, axis=0)
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)

    print(
        f"Loaded {len(features)} pooled feature files. "
        f"Ex: {features[0].shape} from {filenames[0]} with interval {intervals[0]}"
    )
    return features, filenames, intervals


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(args):
    """Instantiate :class:`Pooling`, fit models, and transform all features."""
    pooling = Pooling(
        args.feature_dir, args.boundary_dir, pca_components=args.n_components
    )
    pooling.fit()
    pooling.transform()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pool features based on boundaries")
    parser.add_argument(
        "feature_dir",
        type=Path,
        help="Directory containing per-utterance frame-level feature files (*.npy).",
    )
    parser.add_argument(
        "boundary_dir",
        type=Path,
        help="Directory containing per-utterance boundary files (*.list).",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=350,
        help="Number of PCA components (default: 350).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10_000,
        help="Batch size for incremental scaler / PCA fitting (default: 10 000).",
    )
    args = parser.parse_args()
    main(args)
