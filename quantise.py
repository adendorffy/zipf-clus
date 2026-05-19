from sklearn.cluster import MiniBatchKMeans
from pathlib import Path
import joblib
import numpy as np
from tqdm import tqdm
from utils import seconds_to_frames

class Quantiser:
    def __init__(self, train_features_dir, feature_dir, boundary_dir, k, batch_size=1_000, total_hours=50, collapsed=False):
        self.train_features_dir = train_features_dir
        self.feature_dir = feature_dir
        self.boundary_dir = boundary_dir
        self.k = k
        self.batch_size = batch_size
        self.total_hours = total_hours
        self.collapsed = collapsed
        self.output_dir = self.set_output_dir()
    
    def set_output_dir(self):

        output_dir = "/".join(self.feature_dir.parts[-4:])
        output_dir = Path("discrete_features") / output_dir / "/".join(self.boundary_dir.parts[-2:])
        
        if self.collapsed:
            output_dir = Path("discrete_features_collapsed") / output_dir / "/".join(self.boundary_dir.parts[-2:])

        print(f"Setting output directory to: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def collapse_segment(self, segment_labels):
        prev_unit = None
        collapsed = []
        for unit in segment_labels:
            if unit != prev_unit:
                collapsed.append(unit)
                prev_unit = unit
        return np.array(collapsed)
    
    def fit(self):
        
        kmeans_path = self.output_dir / f"kmeans_k{self.k}.joblib"
        if kmeans_path.exists():
            print(f"KMeans model already exists at {kmeans_path}, loading model.")
            self.kmeans = joblib.load(kmeans_path)
            return
        feature_files = sorted(self.train_features_dir.rglob("*.npy"))
        total_frames = int(self.total_hours * 3600 * 50)
        print(f"Fitting KMeans with {total_frames:,}")
        
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
        print(f"Fitting MiniBatchKMeans with {self.k} clusters on {all_features.shape[0]} samples...")
        self.kmeans = MiniBatchKMeans(n_clusters=self.k, batch_size=self.batch_size, random_state=42)
        self.kmeans.fit(all_features)
        
        joblib.dump(self.kmeans, kmeans_path)
        print(f"KMeans model saved to {kmeans_path}")
    
    def transform(self):

        feature_files = sorted(self.feature_dir.rglob("*.npy"))

        all_boundaries = {}
        for boundary_file in sorted(self.boundary_dir.rglob("*.list")):
            all_boundaries[boundary_file.stem] = np.loadtxt(boundary_file)

        for feature_file in tqdm(feature_files, desc="Transforming features"):
            features = np.load(feature_file)
            if features.ndim == 1:
                features = features.reshape(-1, 1)
                print(f"Warning: Loaded features from {feature_file.stem} have 1 dimension, reshaping to {features.shape}")

            if features.shape[1] != self.kmeans.n_features_in_:
                print(f"Error: Feature dimension {features.shape[1]} does not match KMeans expected dimension {self.kmeans.n_features_in_} for file {feature_file.stem}")
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
                    out_path = self.output_dir / feature_file.relative_to(self.feature_dir).parent / f"{feature_file.stem}_{start}_{end}.npy"
                    
                    if out_path.exists():
                        start = end
                        continue

                    if end == start: continue

                    quantised_segment = quantised[start:end]
                    if self.collapsed:
                        quantised_segment = self.collapse_segment(quantised_segment)

                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(out_path, quantised_segment)
                    start = end

def load_quantised_segments(feature_dir):
    
    feature_files = sorted(feature_dir.rglob("*.npy"))
    segments = []
    filenames = []
    intervals = []
    for feature_file in tqdm(feature_files, desc="Loading quantised segments"):
        if "model" in feature_file.stem: continue

        quantised_segment = np.load(feature_file)
        segments.append(quantised_segment)

        filenames.append(feature_file.stem)
        start, end = map(int, feature_file.stem.split("_")[-2:])
        intervals.append((start, end))

    return segments, filenames, intervals

def main(args):
    quantiser = Quantiser(args.train_features_dir, args.feature_dir, args.boundary_dir, args.k, batch_size=args.batch_size, total_hours=args.total_hours, collapsed=args.collapsed)
    quantiser.fit()
    quantiser.transform()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quantise features using KMeans")
    parser.add_argument("train_features_dir", type=Path, help="Directory containing feature .npy files for training KMeans")
    parser.add_argument("feature_dir", type=Path, help="Directory containing feature .npy files to quantise")
    parser.add_argument("boundary_dir", type=Path, help="Directory containing boundary .list files")
    parser.add_argument("--k", type=int, default=500, help="Number of clusters for KMeans")
    parser.add_argument("--batch_size", type=int, default=1_000, help="Batch size for MiniBatchKMeans")
    parser.add_argument("--total_hours", type=int, default=2, help="Total hours of audio to use for fitting KMeans")
    parser.add_argument("--collapsed", action="store_true", help="Whether to collapse consecutive identical units in the output")
    args = parser.parse_args()
    main(args)

        