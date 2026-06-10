# Copyright (c) 2026 Danel Slabbert, Stellenbosch University. All rights reserved.

"""
quantiser.py

Quantises frame-level SSL features into discrete unit sequences using
MiniBatchKMeans, then slices the resulting label sequences into per-segment
arrays aligned to provided time boundaries.

Pipeline:
  1. KMeans fitting  – trains a codebook of k centroids on up to
     `total_hours` of frame-level features from a (possibly separate)
     training split.
  2. Quantisation    – assigns each frame in the target split to its
     nearest centroid, yielding an integer label sequence.
  3. Boundary-aware slicing – cuts the label sequence at the provided
     boundary frames and saves each segment as a separate .npy file.
  4. Optional deduplication – consecutive identical labels within a segment
     are collapsed to a single occurrence (CTC-style).

The fitted KMeans model is serialised with joblib so it can be reloaded
without refitting on subsequent runs.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from utils import seconds_to_frames


# ---------------------------------------------------------------------------
# Quantiser class
# ---------------------------------------------------------------------------


class Quantiser:
    """Fits a MiniBatchKMeans codebook on SSL features and quantises
    frame-level features into boundary-aligned discrete unit segments.

    Args:
        train_features_dir (Path): Directory of ``*.npy`` feature files used
            *only* for fitting the KMeans model (can differ from the split
            being quantised, e.g. a dedicated training partition).
        feature_dir (Path): Directory of ``*.npy`` feature files to quantise
            (the target split).
        boundary_dir (Path): Directory of ``*.list`` boundary files, each
            containing segment boundary timestamps in seconds (one per line).
        k (int): Number of KMeans clusters (codebook size).
        batch_size (int): Mini-batch size passed to ``MiniBatchKMeans``.
            Defaults to 1 000.
        total_hours (int | float): Maximum hours of audio frames to load for
            KMeans fitting.  Assumes a 50 Hz frame rate.  Defaults to 50.
        collapsed (bool): If ``True``, consecutive identical units within each
            segment are deduplicated before saving.  Defaults to ``False``.
    """

    def __init__(
        self,
        train_features_dir,
        feature_dir,
        boundary_dir,
        k,
        batch_size=1_000,
        total_hours=50,
        collapsed=False,
    ):
        self.train_features_dir = train_features_dir
        self.feature_dir = feature_dir
        self.boundary_dir = boundary_dir
        self.k = k
        self.batch_size = batch_size
        self.total_hours = total_hours
        self.collapsed = collapsed

        self.output_dir = self.set_output_dir()

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------

    def set_output_dir(self):
        """Construct and create the output directory.

        The path is derived from the last four parts of ``feature_dir`` and
        the last two parts of ``boundary_dir``, rooted at either
        ``discrete_features/`` or ``discrete_features_collapsed/`` depending
        on the ``collapsed`` flag.

        Returns:
            Path: Created output directory.
        """
        output_dir = "/".join(self.feature_dir.parts[-4:])
        output_dir = (
            Path("discrete_features")
            / output_dir
            / "/".join(self.boundary_dir.parts[-2:])
        )

        if self.collapsed:
            output_dir = (
                Path("discrete_features_collapsed")
                / output_dir
                / "/".join(self.boundary_dir.parts[-2:])
            )

        print(f"Setting output directory to: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    # ------------------------------------------------------------------
    # Segment deduplication
    # ------------------------------------------------------------------

    def collapse_segment(self, segment_labels):
        """Remove consecutive duplicate labels from a segment (CTC-style).

        For example, ``[3, 3, 7, 7, 7, 2]`` → ``[3, 7, 2]``.

        Args:
            segment_labels (array-like): Integer cluster labels for each frame
                in a segment.

        Returns:
            np.ndarray: Deduplicated label sequence.
        """
        prev_unit = None
        collapsed = []
        for unit in segment_labels:
            if unit != prev_unit:
                collapsed.append(unit)
                prev_unit = unit
        return np.array(collapsed)

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------

    def fit(self):
        """Fit (or load) the MiniBatchKMeans codebook.

        If a serialised model already exists at
        ``<output_dir>/kmeans_k<k>.joblib`` it is loaded and fitting is
        skipped.  Otherwise frames are accumulated from ``train_features_dir``
        up to the ``total_hours`` cap (at 50 Hz) before fitting.

        The fitted model is saved with ``joblib`` for reuse.
        """
        kmeans_path = self.output_dir / f"kmeans_k{self.k}.joblib"

        if kmeans_path.exists():
            print(f"KMeans model already exists at {kmeans_path}, loading model.")
            self.kmeans = joblib.load(kmeans_path)
            return

        feature_files = sorted(self.train_features_dir.rglob("*.npy"))

        total_frames = int(self.total_hours * 3600 * 50)
        print(f"Fitting KMeans with {total_frames:,} frame cap.")

        curr_frames = 0
        all_features = []
        for feature_file in tqdm(feature_files, desc="Loading features for KMeans"):
            features = np.load(feature_file)
            all_features.append(features)
            curr_frames += features.shape[0]
            if curr_frames >= total_frames:
                break

        print(f"Loaded {curr_frames:,} frames for KMeans fitting.")

        all_features = np.vstack(all_features)
        print(
            f"Fitting MiniBatchKMeans with {self.k} clusters "
            f"on {all_features.shape[0]} samples..."
        )
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.k,
            batch_size=self.batch_size,
            random_state=42,
        )
        self.kmeans.fit(all_features)

        joblib.dump(self.kmeans, kmeans_path)
        print(f"KMeans model saved to {kmeans_path}")

    # ------------------------------------------------------------------
    # Quantisation and slicing
    # ------------------------------------------------------------------

    def transform(self):
        """Quantise features and save boundary-aligned discrete unit segments.

        For each utterance in ``feature_dir``:
          1. Predict the nearest centroid for every frame.
          2. Look up its boundary file (matched by stem).
          3. Slice the label sequence at each boundary frame index.
          4. Optionally collapse consecutive identical labels.
          5. Save each segment as ``<stem>_<start>_<end>.npy``.

        Dimension mismatches between the feature file and the fitted KMeans
        are logged and the file is skipped rather than raising an exception.
        """
        feature_files = sorted(self.feature_dir.rglob("*.npy"))

        all_boundaries = {}
        for boundary_file in sorted(self.boundary_dir.rglob("*.list")):
            all_boundaries[boundary_file.stem] = np.loadtxt(boundary_file)

        for feature_file in tqdm(feature_files, desc="Transforming features"):
            features = np.load(feature_file)

            if features.ndim == 1:
                features = features.reshape(-1, 1)
                print(
                    f"Warning: Loaded features from {feature_file.stem} have "
                    f"1 dimension, reshaping to {features.shape}"
                )

            if features.shape[1] != self.kmeans.n_features_in_:
                print(
                    f"Error: Feature dimension {features.shape[1]} does not "
                    f"match KMeans expected dimension "
                    f"{self.kmeans.n_features_in_} for file {feature_file.stem}"
                )
                continue

            quantised = self.kmeans.predict(features)

            segment_name = feature_file.stem
            if segment_name in all_boundaries:
                boundaries = all_boundaries[segment_name]

                if not isinstance(boundaries, np.ndarray):
                    boundaries = np.array(boundaries)
                if boundaries.ndim == 0:
                    boundaries = np.array([boundaries])

                boundaries = [seconds_to_frames(b) for b in boundaries]

                start = 0
                for end in boundaries:
                    out_path = (
                        self.output_dir
                        / feature_file.relative_to(self.feature_dir).parent
                        / f"{feature_file.stem}_{start}_{end}.npy"
                    )

                    if out_path.exists():
                        start = end
                        continue

                    if end == start:
                        continue

                    quantised_segment = quantised[start:end]

                    if self.collapsed:
                        quantised_segment = self.collapse_segment(quantised_segment)

                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(out_path, quantised_segment)
                    start = end


# ---------------------------------------------------------------------------
# Loader utility
# ---------------------------------------------------------------------------


def load_quantised_segments(feature_dir):
    """Load all quantised segment arrays from a directory.

    Files whose stem contains ``"model"`` are skipped (cached KMeans
    artefacts stored alongside the segments).

    Args:
        feature_dir (Path): Directory produced by :meth:`Quantiser.transform`.

    Returns:
        tuple:
            - **segments** (*list[np.ndarray]*): Integer label arrays, one per
              segment (variable length).
            - **filenames** (*list[str]*): Full stem (including
              ``_start_end``) for each segment file.
            - **intervals** (*list[tuple[int,int]]*): ``(start_frame,
              end_frame)`` pair parsed from each filename.
    """
    feature_files = sorted(feature_dir.rglob("*.npy"))
    segments, filenames, intervals = [], [], []

    for feature_file in tqdm(feature_files, desc="Loading quantised segments"):
        # Skip serialised model artefacts.
        if "model" in feature_file.stem:
            continue

        quantised_segment = np.load(feature_file)
        segments.append(quantised_segment)

        filenames.append(feature_file.stem)
        start, end = map(int, feature_file.stem.split("_")[-2:])
        intervals.append((start, end))

    return segments, filenames, intervals


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(args):
    """Instantiate :class:`Quantiser`, fit the codebook, and quantise."""
    quantiser = Quantiser(
        args.train_features_dir,
        args.feature_dir,
        args.boundary_dir,
        args.k,
        batch_size=args.batch_size,
        total_hours=args.total_hours,
        collapsed=args.collapsed,
    )
    quantiser.fit()
    quantiser.transform()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantise features using KMeans")
    parser.add_argument(
        "train_features_dir",
        type=Path,
        help="Directory of *.npy feature files used to fit the KMeans codebook.",
    )
    parser.add_argument(
        "feature_dir",
        type=Path,
        help="Directory of *.npy feature files to quantise.",
    )
    parser.add_argument(
        "boundary_dir",
        type=Path,
        help="Directory of *.list boundary files (timestamps in seconds).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=500,
        help="Number of KMeans clusters / codebook size (default: 500).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1_000,
        help="Mini-batch size for MiniBatchKMeans (default: 1 000).",
    )
    parser.add_argument(
        "--total_hours",
        type=int,
        default=2,
        help="Maximum hours of audio frames to load for KMeans fitting (default: 2).",
    )
    parser.add_argument(
        "--collapsed",
        action="store_true",
        help="Deduplicate consecutive identical units within each segment before saving.",
    )
    args = parser.parse_args()
    main(args)
